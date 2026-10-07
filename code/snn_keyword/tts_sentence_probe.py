"""Probe: the keyword inside sentences (synthetic speech), stage 1 alone and the cascade.

Every positive so far is an isolated Speech Commands word; a user says the
keyword inside sentences. LibriSpeech has one utterance with "sheila", so
the probe uses Piper voices that no model was trained on: libritts_r
speakers of the test split (make_tts_negatives.speaker_splits) and the
held-out voice models (vctk, l2arctic, arctic). Each voice says:
  isolated  "Sheila." / "Sheila?"
  start     "Sheila, turn on the lights." ...
  middle    "Tell Sheila that I called." ...
  end       "Thank you, Sheila." ...
  negative  sentences with near-miss words and no keyword ("She laughed at me.",
            "Zero degrees today.", ...)
Each utterance sits in background noise (0.5 s before, 1 s after, as in the
negatives stream) and goes through the integer stage 1 and the integer
verifier with the cascade's request timing (verifier_cascade.cascade_requests).
A positive counts if the utterance has a detection; a negative sentence
counts as a false accept if it has one. Synthetic speech is not the user:
the isolated rows are the baseline that separates the effect of the sentence
from that of the synthetic voice. Nothing here is used to choose a model.
Writes results/tts_sentence_probe_<tag>.json.
"""
import argparse
import collections
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import keyword_config as K

ROOT = Path(__file__).resolve().parent
SR = 16000

SENTENCES = {
    'isolated': ['Sheila.', 'Sheila?', 'Sheila!'],
    'start': ['Sheila, turn on the lights.', 'Sheila, what time is it?', 'Sheila, play some music please.',
              'Sheila, can you hear me?', 'Sheila, set a timer for ten minutes.'],
    'middle': ['Tell Sheila that I called.', 'I think Sheila is at home.', 'My friend Sheila likes green tea.',
               'Yesterday Sheila bought a new car.', 'Please ask Sheila about the meeting.'],
    'end': ['Thank you, Sheila.', 'Good morning, Sheila.', 'I had lunch with Sheila.',
            'This letter is for Sheila.', 'Where are you going, Sheila?'],
    'negative': ['She laughed at me.', 'She left the house early.', 'She lay down on the sofa.',
                 'The sheep are in the field.', 'He hid behind a shield.', 'Put the book on the shelf.',
                 'Zero degrees today.', 'The score is zero to zero.', 'She will be late.', 'She looked at the sea.',
                 'The ceiling is white.', 'That was a good feeling.', 'Tequila and vanilla.', 'She said hello.'],
}


def jobs(per_model, seed=0):
    from make_tts_negatives import num_speakers, speaker_splits
    rng = np.random.default_rng(seed)
    test_spk = speaker_splits()['test']
    out = []
    for model in ('libritts_r', 'vctk', 'l2arctic', 'arctic'):
        speakers = rng.choice(test_spk, per_model, replace=False) if model == 'libritts_r' \
            else rng.choice(num_speakers(model), min(per_model, num_speakers(model)), replace=False)
        for spk in speakers:
            for kind, sents in SENTENCES.items():
                for i, text in enumerate(sents):
                    out.append(dict(model=model, speaker=int(spk), kind=kind, i=i, text=text,
                                    length_scale=float(rng.uniform(.85, 1.25)), noise_scale=float(rng.uniform(.4, .8)),
                                    noise_w_scale=float(rng.uniform(.5, 1.))))
    return out


_voices = {}


def synth(job):
    """As make_tts_negatives.synth: Piper, resampled to 16 kHz, peak 0.5."""
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly
    from make_tts_negatives import voice_files
    if job['model'] not in _voices:
        _voices[job['model']] = PiperVoice.load(*voice_files(job['model']))
    v = _voices[job['model']]
    cfg = SynthesisConfig(speaker_id=job['speaker'], length_scale=job['length_scale'],
                          noise_scale=job['noise_scale'], noise_w_scale=job['noise_w_scale'])
    audio = np.concatenate([c.audio_float_array for c in v.synthesize(job['text'], cfg)])
    rate = v.config.sample_rate
    g = np.gcd(rate, SR)
    audio = resample_poly(audio, SR // g, rate // g).astype(np.float32)
    return audio / max(1e-6, np.abs(audio).max()) * .5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage1', type=Path, default=Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream')
                   / 'sheila_qat_seed2/int_model.npz')
    p.add_argument('--verifier', type=Path, required=True)
    p.add_argument('--t1', type=int, required=True)
    p.add_argument('--t2', type=int, required=True)
    p.add_argument('--stage1-threshold', type=int, default=18462, help='stage 1 alone')
    p.add_argument('--per-model', type=int, default=12, help='speakers per voice model')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    if K.KEYWORD != 'sheila':
        raise SystemExit('the sentences are written for KWS_KEYWORD=sheila')
    import robust_eval as R
    from features import frame_features
    from model import decision_scores, integer_forward_stream
    from verifier_cascade import cascade_requests
    from verifier_export import load_quantized
    t0 = time.perf_counter()
    js = jobs(a.per_model)
    cache = ROOT / f'data_verifier/tts_probe_audio_{a.per_model}.npz'   # the same utterances for every model
    if cache.exists():
        c = np.load(cache)
        audio = [c['audio'][c['off'][i]:c['off'][i + 1]].astype(np.float32) / 32768 for i in range(len(js))]
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            audio = list(pool.map(synth, js, chunksize=8))
        np.savez(cache, audio=np.concatenate([np.round(x * 32767).astype(np.int16) for x in audio]),
                 off=np.r_[0, np.cumsum([len(x) for x in audio])])
    print(len(js), 'utterances', round(time.perf_counter() - t0), 's', flush=True)
    noise = R.bg_noise(a.data)
    rng = np.random.default_rng(1)
    q1 = dict(np.load(a.stage1))
    K.check_model_keyword(q1.get('keyword'), 'stage 1')
    qv = load_quantized(a.verifier)
    front = str(q1.get('frontend', 'logmel'))
    window = int(q1.get('decision_window', 1))
    rows = []
    for j, x in zip(js, audio):
        buf = np.zeros(len(x) + int(1.5 * SR), np.float32)
        buf[int(.5 * SR):int(.5 * SR) + len(x)] = x
        o = rng.integers(0, len(noise) - len(buf))
        buf = buf + noise[o:o + len(buf)] * rng.uniform(.02, .1)
        fr = frame_features(buf, front)
        s, _, _ = integer_forward_stream(fr[None], q1)
        d = decision_scores(s[0].astype(np.int64), window)
        casc = cascade_requests(fr, q1, qv, a.t1, a.t2, window)
        ran = [r_ for r_ in casc if r_[0] & 4]
        rows.append(dict(j, stage1=bool(d.max() >= a.stage1_threshold), cascade=any(b & 1 for b, *_ in casc),
                         proposed=bool(d.max() >= a.t1), stage1_max=int(d.max()),
                         verifier_max=int(max(r_[3] for r_ in ran)) if ran else None, seconds=len(x) / SR))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r['kind']]['all'].append(r)
        by[r['kind']][r['model']].append(r)
    report = {'stage1': str(a.stage1), 'verifier': str(a.verifier), 't1': a.t1, 't2': a.t2,
              'stage1_threshold': a.stage1_threshold, 'utterances': len(rows), 'per_model': a.per_model,
              'rates': {}}
    for kind in SENTENCES:
        report['rates'][kind] = {m: {'n': len(v), 'stage1': round(float(np.mean([r['stage1'] for r in v])), 4),
                                     'proposed_at_t1': round(float(np.mean([r['proposed'] for r in v])), 4),
                                     'cascade': round(float(np.mean([r['cascade'] for r in v])), 4)}
                                 for m, v in by[kind].items()}
        print(kind, json.dumps(report['rates'][kind]['all']), flush=True)
    report['negative_accepts'] = collections.Counter(r['text'] for r in rows if r['kind'] == 'negative' and r['cascade'])
    report['missed_by_sentence'] = collections.Counter(r['text'] for r in rows if r['kind'] != 'negative' and not r['cascade'])
    report['utterances_detail'] = [{k: r[k] for k in ('model', 'speaker', 'kind', 'text', 'stage1_max', 'verifier_max',
                                                      'proposed', 'cascade')} for r in rows]
    out = ROOT / f'results/tts_sentence_probe{"_" + a.tag if a.tag else ""}.json'
    out.write_text(json.dumps(report, indent=1))
    print('negative accepts', dict(report['negative_accepts']))
    print('written', out, round(time.perf_counter() - t0), 's')


if __name__ == '__main__':
    main()
