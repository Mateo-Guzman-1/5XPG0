"""Phase 3 training of the streaming SNN (snn_stream.py), curriculum of IMPLEMENTATION_PLAN.md.

stage 1  35 words + _unknown_ + _silence_ on 1 s clips (multicorpus.ClipSampler),
         cross-entropy on the readout averaged over the last 0.4 s, plus
         knowledge distillation [13][7] from the BC-ResNet-8 teacher
         (teacher.py) on the same augmented audio.
stage 2  streaming fine-tune [11]: 3 s streams of up to three clips (and
         LibriSpeech speech) at random positions, augmented as a whole.
         Classes _silence_, _unknown_, yes. A "yes" is rewarded at its best
         frame within 0.35 s after the end of the word (max-pooling loss);
         the frames while it is being spoken are "don't care"; every other
         frame is labelled unknown (speech) or silence.
Both stages: surrogate gradients through time [2], learnable time constants
[3][36], a spike-rate penalty [34], DCLS delay width annealed from 16 taps to
0.5 over the first 80% of stage 1 epochs, then delays rounded to one tap
for the rest; stage 2 keeps the rounded delays throughout.
Checkpoints are chosen on validation clips only.

Stage 2 options against false accepts on running speech (JOURNAL entry 19):
--speech-streams f  a fraction f of each batch is 3 s of LibriSpeech only
                    (every frame "unknown"), besides the speech tails above;
--mine-every n      every n steps, score --mine-pool random 3 s LibriSpeech
                    segments with the current model (no gradients, same
                    augmentation) and keep the --mine-keep highest-scoring
                    ones; half of the speech-only streams come from that pool
                    (online hard-negative mining).
--word-mine-share f the same for words: every --mine-every steps the
                    --mine-keep highest-"yes" of --mine-pool random non-"yes"
                    training words (all corpora) are kept, and a fraction f of
                    the word slots in the streams take one of them.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from augment_online import Augmenter
from multicorpus import CLASSES, SILENCE, UNKNOWN, ClipSampler
from snn_stream import MAX_DELAY, StreamSNN

ROOT = Path(__file__).resolve().parent
SR, HOP = 16000, 160
import keyword_config as K
YES = CLASSES.index(K.KEYWORD)   # the keyword (KWS_KEYWORD; "yes" by default)
STREAM_CLASSES = ['_silence_', '_unknown_', K.KEYWORD]


def sigma_at(epoch, epochs):
    f = min(1., epoch / max(1, .8 * epochs))
    return float(np.exp(np.log(MAX_DELAY / 2) * (1 - f) + np.log(.5) * f))


def word_bounds(wave, floor_db=30):
    """First and last 10 ms frame within floor_db of the loudest frame, per clip (in frames)."""
    n = wave.shape[1] // HOP
    e = 10 * torch.log10(wave[:, :n * HOP].reshape(len(wave), n, HOP).square().mean(2) + 1e-10)
    active = e > e.max(1, keepdim=True)[0] - floor_db
    idx = torch.arange(n, device=wave.device)[None].expand_as(active)
    start = torch.where(active, idx, n).min(1)[0]
    end = torch.where(active, idx, -1).max(1)[0] + 1
    return start, end


def rates_penalty(rates, target):
    return sum(F.relu(r - target) ** 2 for r in rates) * 100


def clip_val(model, sampler, aug, device, frontend, corpus='sc', batch=1024):
    """35-word accuracy and yes-vs-rest recall on clean validation clips."""
    clips, rows, cls, _ = sampler.all_clips(corpus)
    order = np.argsort(rows); rows, cls = rows[order], cls[order]
    model.eval()
    pred, margin = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            w = torch.tensor(clips[rows[i:i + batch]].astype(np.float32) / 32768, device=device)
            x = aug.front.frames(aug.front.mel(aug.front.power(w)), frontend)
            o, _, _ = model(x)
            logits = o[:, -40:].mean(1)
            pred.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return float((pred == cls).mean())


def build_streams(sampler, aug, n, seconds, device, rng, yes_share=.25, hard_words=None, hard_share=0.):
    """n waveforms of `seconds` with 3-class frame targets and yes windows.

    yes_share of the clips are replaced by "yes" clips (about 5% at the corpus mix).

    Returns wave (n, L), frame labels (n, T) in {0 silence, 1 unknown, -1 don't care},
    and a list of (stream, first, last) target frame windows for each "yes".
    """
    L = int(seconds * SR)
    T = (L - 400) // HOP + 1
    wave = torch.zeros(n, L, device=device)
    labels = torch.zeros(n, T, dtype=torch.long, device=device)  # silence
    windows = []
    k = 3 * n
    clip, cls, _ = sampler.batch(k, max_shift=0)
    swap = np.flatnonzero(rng.random(k) < yes_share)
    if len(swap):
        clip[torch.tensor(swap, device=device)] = torch.tensor(sampler.yes_batch(len(swap)), device=device)
        cls[torch.tensor(swap, device=device)] = YES
    if hard_words is not None and hard_share > 0:
        # Word slots (not "yes", not silence) take a mined hard negative word.
        free = np.flatnonzero((cls.cpu().numpy() != YES) & (cls.cpu().numpy() != SILENCE) & (rng.random(k) < hard_share))
        if len(free):
            rows = hard_words.draw(len(free))
            clip[torch.tensor(free, device=device)] = torch.tensor(sampler._wave(sampler.row[rows]), device=device)
            cls[torch.tensor(free, device=device)] = torch.tensor(sampler.cls[rows], device=device)
    clip = aug.speed(clip, torch.rand(k, device=device) < aug.cfg['p_speed'])
    start, end = word_bounds(clip)
    slots = rng.integers(0, 3, k)  # 0: empty, 1-2: clip; about one empty slot in three
    for i in range(n):
        t = int(rng.uniform(0, .3) * SR)
        for j in range(3):
            c = 3 * i + j
            if slots[c] == 0 or cls[c] == SILENCE:
                t += int(rng.uniform(.2, .8) * SR)
                continue
            s0, e0 = int(start[c]) * HOP, min(SR, int(end[c]) * HOP + 400)
            seg = clip[c, max(0, s0 - 800):e0 + 800]
            if t + len(seg) > L:
                break
            wave[i, t:t + len(seg)] += seg
            f0 = (t + (s0 - max(0, s0 - 800))) // HOP
            f1 = min(T - 1, (t + (e0 - max(0, s0 - 800))) // HOP)
            if cls[c] == YES:
                labels[i, f0:min(T, f1 + 35)] = -1
                if f1 + 5 < T:
                    windows.append((i, f1, min(T - 1, f1 + 35)))
            else:
                labels[i, max(0, f0 - 2):min(T, f1 + 3)] = torch.where(
                    labels[i, max(0, f0 - 2):min(T, f1 + 3)] == -1, -1, 1)
            t += len(seg) + int(rng.uniform(.1, .6) * SR)
    # LibriSpeech speech in the free tail of a third of the streams.
    if sampler.speech is not None:
        for i in rng.choice(n, n // 3, replace=False):
            busy = torch.nonzero(labels[i] != 0)
            f = int(busy.max()) + 20 if len(busy) else 0
            if f < T - 50:
                s = int(rng.integers(0, len(sampler.speech) - L))
                t0 = f * HOP
                seg = torch.tensor(sampler.speech[s:s + L - t0].astype(np.float32) / 32768, device=device)
                wave[i, t0:t0 + len(seg)] += seg
                labels[i, f:] = torch.where(labels[i, f:] == -1, -1, 1)
    return wave, labels, windows


def speech_streams(sampler, starts, seconds, device):
    """3 s LibriSpeech segments at the given sample offsets: (n, L) waveforms, all frames unknown."""
    L = int(seconds * SR)
    T = (L - 400) // HOP + 1
    wave = torch.tensor(np.stack([sampler.speech[s:s + L] for s in starts]).astype(np.float32) / 32768, device=device)
    return wave, torch.ones(len(starts), T, dtype=torch.long, device=device)


class HardNegatives:
    """Pool of LibriSpeech offsets the model currently scores highest for "yes"."""
    def __init__(self, sampler, seconds, rng, pool=2048, keep=256):
        self.sampler, self.L, self.rng, self.pool, self.keep = sampler, int(seconds * SR), rng, pool, keep
        self.starts = self.rng.integers(0, len(sampler.speech) - self.L, keep)
        self.scores = None

    @torch.no_grad()
    def mine(self, model, aug, frontend, device, chunk=256):
        cand = np.r_[self.starts, self.rng.integers(0, len(self.sampler.speech) - self.L, self.pool)]
        best = []
        was = model.training
        model.eval()
        for i in range(0, len(cand), chunk):
            w, _ = speech_streams(self.sampler, cand[i:i + chunk], self.L / SR, device)
            w, mic = aug.waveform(w)
            o, _, _ = model(aug.spectral(w, mic, frontend))
            best.append(F.log_softmax(o, -1)[..., 2].max(1)[0].float().cpu().numpy())
        model.train(was)
        best = np.concatenate(best)
        top = np.argsort(best)[::-1][:self.keep]
        self.starts, self.scores = cand[top], best[top]
        return float(np.exp(self.scores).mean())

    def draw(self, n):
        return self.starts[self.rng.integers(0, len(self.starts), n)]


class HardWords:
    """Pool of non-"yes" training words (sampler indices) the model currently scores highest for "yes"."""
    def __init__(self, sampler, rng, pool=2048, keep=256):
        self.sampler, self.rng, self.pool, self.keep = sampler, rng, pool, keep
        self.ids = np.flatnonzero((sampler.cls != YES) & (sampler.cls != SILENCE))
        self.best = self.rng.choice(self.ids, keep)

    @torch.no_grad()
    def mine(self, model, aug, frontend, device, chunk=512):
        cand = np.r_[self.best, self.rng.choice(self.ids, self.pool)]
        scores = []
        was = model.training
        model.eval()
        for i in range(0, len(cand), chunk):
            w = torch.tensor(self.sampler._wave(self.sampler.row[cand[i:i + chunk]]), device=device)
            w, mic = aug.waveform(w)
            o, _, _ = model(aug.spectral(w, mic, frontend))
            scores.append((o[..., 2] - o[..., :2].max(-1)[0]).max(1)[0].float().cpu().numpy())
        model.train(was)
        scores = np.concatenate(scores)
        top = np.argsort(scores)[::-1][:self.keep]
        self.best = cand[top]
        return float(scores[top].mean())

    def draw(self, n):
        return self.best[self.rng.integers(0, len(self.best), n)]


def stream_loss(o, labels, windows, pos_weight=4.):
    logp = F.log_softmax(o, -1)
    neg = labels >= 0
    loss_neg = F.nll_loss(logp[neg], labels[neg]) if neg.any() else o.sum() * 0
    if windows:
        pos = torch.stack([logp[i, a:b + 1, 2].max() for i, a, b in windows])
        loss_pos = -pos.mean()
    else:
        loss_pos = o.sum() * 0
    return loss_neg + pos_weight * loss_pos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', type=int, choices=[1, 2], required=True)
    p.add_argument('--init', type=Path, help='stage 2: stage 1 checkpoint')
    p.add_argument('--teacher', type=Path, default=ROOT / 'runs_teacher/bcresnet8.pt')
    p.add_argument('--no-kd', action='store_true')
    p.add_argument('--kd-weight', type=float, default=.5)
    p.add_argument('--kd-temp', type=float, default=2.)
    p.add_argument('--n1', type=int, default=128)
    p.add_argument('--n2', type=int, default=128)
    p.add_argument('--no-delays', action='store_true')
    p.add_argument('--no-recurrence', action='store_true')
    p.add_argument('--lif', action='store_true', help='no threshold adaptation')
    p.add_argument('--frontend', choices=['logmel', 'logmel_w', 'logmel_agc', 'pcen'], default='logmel')
    p.add_argument('--corpora', nargs='+', default=['sc', 'mswc', 'tts'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps-per-epoch', type=int, default=300)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--rate-target', type=float, default=.05)
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mic-ranges', choices=['narrow', 'wide'], default='narrow',
                   help='training microphone corners (channels.RANGES); held-out profiles are unchanged')
    p.add_argument('--speech-streams', type=float, default=0., help='stage 2: fraction of speech-only streams')
    p.add_argument('--mine-every', type=int, default=0, help='stage 2: hard-negative mining period in steps (0: off)')
    p.add_argument('--exclude-words', default=None,
                   help="regex of training words to leave out; 'yes.+' drops every word that begins with a "
                        "complete yes (don't care for a causal detector, JOURNAL entries 20-21)")
    p.add_argument('--word-mine-share', type=float, default=0., help='stage 2: share of word slots taken by mined words')
    p.add_argument('--mine-pool', type=int, default=2048)
    p.add_argument('--mine-keep', type=int, default=256)
    p.add_argument('--qat', action='store_true', help='Phase 4: int8 fake quantization and rounded shifts')
    p.add_argument('--name', default=None)
    p.add_argument('--out', type=Path, default=ROOT / 'runs_stream')
    a = p.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = ClipSampler('train', dev, a.corpora, seed=a.seed, exclude=a.exclude_words)
    val = ClipSampler('validation', dev, a.corpora, seed=a.seed, libri=False)
    aug = Augmenter(dev, np.load(ROOT / 'data/multi/noise_train.npy'), seed=a.seed, mic_ranges=a.mic_ranges)
    config = dict(n1=a.n1, n2=a.n2, classes=len(CLASSES) if a.stage == 1 else 3, delays=not a.no_delays,
                  recurrent=not a.no_recurrence, adaptive=not a.lif, seed=a.seed)
    model = StreamSNN(**config).to(dev)
    if a.stage == 2:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        state = {k: v for k, v in ck['state_dict'].items() if not k.startswith('out.') and k != 'ko'}
        model.load_state_dict(state, strict=False)
    if a.qat:
        model.qat = True
        model.l1.hard = model.l2.hard = True
    teacher = None
    if a.stage == 1 and not a.no_kd:
        from teacher import load_teacher
        teacher, tfront = load_teacher(a.teacher, dev)
    groups = [{'params': [p_ for n, p_ in model.named_parameters() if n != 'pos' and p_.requires_grad], 'lr': a.lr}]
    if model.pos.requires_grad:
        groups.append({'params': [model.pos], 'lr': a.lr * 50, 'weight_decay': 0.})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * a.steps_per_epoch, eta_min=a.lr / 20)
    name = a.name or f"stage{a.stage}_seed{a.seed}"
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    history, best, start = [], -1., time.perf_counter()
    hard = HardNegatives(train, a.seconds, rng, a.mine_pool, a.mine_keep) if a.stage == 2 and a.mine_every else None
    hard_words = HardWords(train, rng, a.mine_pool, a.mine_keep) if a.stage == 2 and a.mine_every and a.word_mine_share else None
    n_speech = int(round(a.speech_streams * a.batch)) if a.stage == 2 else 0
    step = 0
    for epoch in range(a.epochs):
        model.sigma = sigma_at(epoch, a.epochs)
        # Stage 2 starts from stage 1's rounded delays and keeps them single taps.
        model.hard_delays = a.stage == 2 or epoch >= int(.8 * a.epochs)
        model.train()
        sums = dict(loss=0., rate1=0., rate2=0.)
        for _ in range(a.steps_per_epoch):
            with torch.no_grad():
                if a.stage == 1:
                    wave, y, sil = train.batch(a.batch)
                    wave, mic = aug.waveform(wave, sil)
                else:
                    if hard is not None and step % a.mine_every == 0:
                        sums.setdefault('mined_p_yes', []).append(hard.mine(model, aug, a.frontend, dev))
                        if hard_words is not None:
                            sums.setdefault('mined_word_margin', []).append(hard_words.mine(model, aug, a.frontend, dev))
                    wave, labels, windows = build_streams(train, aug, a.batch - n_speech, a.seconds, dev, rng,
                                                          hard_words=hard_words, hard_share=a.word_mine_share)
                    speed = aug.cfg['p_speed']; aug.cfg['p_speed'] = 0.  # clips were sped up before placement
                    wave, mic = aug.waveform(wave)
                    aug.cfg['p_speed'] = speed
                    if n_speech:
                        n_hard = n_speech // 2 if hard is not None else 0
                        starts = np.r_[hard.draw(n_hard) if n_hard else np.zeros(0, np.int64),
                                       rng.integers(0, len(train.speech) - int(a.seconds * SR), n_speech - n_hard)]
                        sw, sl = speech_streams(train, starts, a.seconds, dev)
                        sw, smic = aug.waveform(sw)
                        wave, mic, labels = torch.cat([wave, sw]), torch.cat([mic, smic]), torch.cat([labels, sl])
                x = aug.spectral(wave, mic, a.frontend)
                if teacher is not None:
                    t_logits = teacher(tfront(wave, aug, mic))
            o, _, rates = model(x)
            if a.stage == 1:
                logits = o[:, -40:].mean(1)
                loss = F.cross_entropy(logits, y)
                if teacher is not None:
                    kd = F.kl_div(F.log_softmax(logits / a.kd_temp, 1), F.softmax(t_logits / a.kd_temp, 1),
                                  reduction='batchmean') * a.kd_temp ** 2
                    loss = (1 - a.kd_weight) * loss + a.kd_weight * kd
            else:
                loss = stream_loss(o, labels, windows)
            loss = loss + rates_penalty(rates, a.rate_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step(); sched.step(); step += 1
            sums['loss'] += loss.item(); sums['rate1'] += float(rates[0]); sums['rate2'] += float(rates[1])
        mined = {k: sums.pop(k) for k in ('mined_p_yes', 'mined_word_margin') if k in sums}
        r = {k: v / a.steps_per_epoch for k, v in sums.items()}
        r.update({k: round(float(np.mean(v)), 4) for k, v in mined.items()})
        r.update(epoch=epoch + 1, sigma=model.sigma, hard_delays=model.hard_delays)
        if a.stage == 1:
            r['val_sc_35'] = clip_val(model, val, aug, dev, a.frontend)
            score = r['val_sc_35']
        else:
            score = -r['loss']  # stage 2 checkpoints are ranked by stream_select.py on validation streams
        history.append(r)
        print(json.dumps(r), flush=True)
        if model.hard_delays and score > best:
            best = score
            torch.save({'state_dict': copy.deepcopy(model.state_dict()), 'config': config, 'args': vars(a),
                        'frontend': a.frontend, 'yes_class': YES if a.stage == 1 else 2, 'keyword': K.KEYWORD,
                        'classes': CLASSES if a.stage == 1 else STREAM_CLASSES, 'threshold': 0.,
                        'qat': a.qat, 'hard_shifts': a.qat},
                       out / 'model.pt')
        torch.save({'state_dict': model.state_dict(), 'config': config, 'args': vars(a), 'frontend': a.frontend,
                    'yes_class': YES if a.stage == 1 else 2, 'keyword': K.KEYWORD,
                    'classes': CLASSES if a.stage == 1 else STREAM_CLASSES, 'threshold': 0.,
                        'qat': a.qat, 'hard_shifts': a.qat}, out / 'last.pt')
    summary = dict(vars(a), best=best, training_seconds=time.perf_counter() - start, history=history)
    (out / 'training.json').write_text(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
