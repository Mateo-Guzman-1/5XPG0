"""On-the-fly augmentation and the front end on the GPU (IMPLEMENTATION_PLAN.md, Phase 1).

Per example, in this order (each step with its own probability):
  speed      resampling by 0.9-1.1 (tempo and pitch) [19]
  room       convolution with a *simulated* OpenSLR 28 RIR [17], direct path
             aligned, dry RMS kept (real RIRs are held out: channels.py)
  noise      MUSAN noise/music or OpenSLR 28 point-source noise at 0-30 dB SNR [18]
  level      -20 to +10 dB, then the microphone's preamp gain and tanh clipping
  mic EQ     |H(f)|^2 of a training microphone (even third-octave bands only,
             channels.draw_mic) applied to the power spectrum [22]
  VTLP       warped mel bank, alpha 0.9-1.1 [19]
  SpecAug    up to 2 time masks of <= 3 frames and 2 band masks of <= 3 bands [16]
The EQ is applied to the power spectrum, which is exact for filters much
shorter than the 25 ms frame and cheap on the GPU.

FrontEnd reproduces features.py in torch (float32; matches numpy to rounding):
frame log-mel or PCEN uint8 frames for streaming models, and the pooled
32-bin window features of the release for the dense-model ablation.
"""
from functools import lru_cache
import numpy as np
import torch
from scipy.signal import butter, sosfreqz

import channels
from features import LOGMEL_RANGE, N_MELS, PCEN, SAMPLE_RATE, mel_bank

N_FFT_BINS = 257
DEFAULTS = dict(p_speed=.8, p_room=.4, p_noise=.8, p_mic=.7, p_vtlp=.8, p_specaug=.5,
                snr_db=(0, 30), level_db=(-20, 10), rir_seconds=.5, n_mics=4096, n_rirs=4000,
                mic_ranges='narrow')


def warped_bank(alpha):
    """Mel bank on a VTLP-warped frequency axis (piecewise linear, fixed at Nyquist)."""
    freqs = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    nyq, hi = SAMPLE_RATE / 2, 4800.
    cut = hi * min(alpha, 1) / alpha
    w = np.where(freqs <= cut, freqs * alpha, nyq - (nyq - hi * min(alpha, 1)) / (nyq - cut) * (nyq - freqs))
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    points = 700 * (10 ** (np.linspace(mel(80), mel(7600), N_MELS + 2) / 2595) - 1)
    bank = np.maximum(0, np.minimum((w[None] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None]),
                                    (points[2:, None] - w[None]) / (points[2:, None] - points[1:-1, None])))
    return (bank / np.maximum(bank.sum(axis=1, keepdims=True), 1e-9)).astype(np.float32)


def mic_power_response(mic):
    freqs = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    sos = np.concatenate([butter(2, mic['highpass'], 'highpass', fs=SAMPLE_RATE, output='sos'),
                          butter(4, mic['lowpass'], 'lowpass', fs=SAMPLE_RATE, output='sos')]
                         + [channels.peaking(*p) for p in mic['peaks']])
    _, h = sosfreqz(sos, worN=freqs, fs=SAMPLE_RATE)
    return (np.abs(h) ** 2).astype(np.float32)


@lru_cache(maxsize=None)
def pool_matrix(n_frames, n_time):
    """features.py pooling (np.array_split means) as a (n_time, n_frames) matrix."""
    m = np.zeros((n_time, n_frames), np.float32)
    for i, part in enumerate(np.array_split(np.arange(n_frames), n_time)):
        m[i, part] = 1 / len(part)
    return m


class FrontEnd:
    def __init__(self, device):
        self.device = device
        w = np.hanning(401)[:-1].astype(np.float32)
        self.window = torch.tensor(w, device=device)
        self.norm = float(w.sum())
        self.bank = torch.tensor(mel_bank(), device=device)
        self.alphas = np.linspace(.9, 1.1, 21)
        self.warped = torch.tensor(np.stack([warped_bank(a) for a in self.alphas]), device=device)

    def power(self, wave):
        """(B, L) float waveform -> (B, T, 257) power spectrum; frame k starts at sample 160 k."""
        frames = wave.unfold(1, 400, 160) * self.window
        spec = torch.fft.rfft(frames, n=512) / self.norm
        return spec.real ** 2 + spec.imag ** 2

    def mel(self, power, bank_ids=None):
        if bank_ids is None:
            return power @ self.bank.T
        return torch.einsum('btf,bmf->btm', power, self.warped[bank_ids])

    @staticmethod
    def logmel(mel, frontend='logmel'):
        lo, hi = LOGMEL_RANGE[frontend]
        db = 10 * torch.log10(mel.clamp_min(10 ** (lo / 10)))
        return torch.round(((db - lo) / (hi - lo)).clamp(0, 1) * 255)

    @staticmethod
    def pcen(mel, p=PCEN):
        e = mel.double() * p['scale']
        m = e[:, 0]
        out = torch.empty_like(e)
        for t in range(e.shape[1]):
            m = (1 - p['s']) * m + p['s'] * e[:, t]
            out[:, t] = m
        y = (e / (1e-6 + out) ** p['alpha'] + p['delta']) ** p['r'] - p['delta'] ** p['r']
        return torch.round((y / p['top']).clamp(0, 1) * 255).float()

    def frames(self, mel, frontend='logmel'):
        """uint8-valued float frames (B, T, 24)."""
        return self.pcen(mel) if frontend == 'pcen' else self.logmel(mel, frontend)

    def pooled(self, mel, n_time=32):
        """Release window features (B, 24 * n_time), mel-major, from a 1 s mel sequence."""
        db = 10 * torch.log10(mel.clamp_min(1e-8))
        pm = torch.tensor(pool_matrix(mel.shape[1], n_time), device=mel.device)
        pooled = torch.einsum('btm,nt->bmn', db, pm)
        return torch.round(((pooled + 80) / 80).clamp(0, 1) * 255).reshape(len(mel), -1)


class Augmenter:
    """Waveform and spectral augmentation for batches on one device.

    noise: 1-D array of concatenated noise/music recordings (int16 or float);
    kept on the device as float16.
    """
    def __init__(self, device, noise, seed=0, **cfg):
        self.cfg = dict(DEFAULTS, **cfg)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.front = FrontEnd(device)
        if noise is not None and len(noise):
            noise = torch.tensor(np.asarray(noise))
            self.noise = (noise.float() / 32768 if noise.dtype == torch.int16 else noise.float()).half().to(device)
        else:
            self.noise = None
        mics = [channels.draw_mic(self.rng, heldout=False, ranges=self.cfg['mic_ranges']) for _ in range(self.cfg['n_mics'])]
        self.mic_h = torch.tensor(np.stack([mic_power_response(m) for m in mics]), device=device)
        self.mic_gain = torch.tensor([10 ** (m['gain_db'] / 20) for m in mics], device=device)
        self.mic_drive = torch.tensor([m['drive'] for m in mics], device=device)
        self.rirs = self._load_rirs()

    def _load_rirs(self):
        import soundfile as sf
        files = channels.training_rir_files()
        pick = self.rng.choice(len(files), min(self.cfg['n_rirs'], len(files)), replace=False)
        n = int(self.cfg['rir_seconds'] * SAMPLE_RATE)
        out = np.zeros((len(pick), n), np.float32)
        for i, k in enumerate(pick):
            r = sf.read(files[k], dtype='float32', always_2d=True)[0][:, 0]
            r = r[int(np.argmax(np.abs(r))):][:n]
            out[i, :len(r)] = r
        return torch.tensor(out, device=self.device)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def randint(self, high, shape):
        return torch.randint(high, shape, generator=self.gen, device=self.device)

    def speed(self, wave, keep):
        b, n = wave.shape
        s = torch.where(keep, .9 + .2 * self.rand(b), torch.ones(b, device=self.device))
        pos = (torch.arange(n, device=self.device)[None] * s[:, None]).clamp(max=n - 1)
        i0 = pos.floor().long(); i1 = (i0 + 1).clamp(max=n - 1); f = pos - i0
        out = wave.gather(1, i0) * (1 - f) + wave.gather(1, i1) * f
        # Past the end of a sped-up source there is nothing: silence, not the last sample.
        return torch.where(torch.arange(n, device=self.device)[None] * s[:, None] <= n - 1, out, 0.)

    def room(self, wave, keep):
        idx = torch.nonzero(keep).flatten()
        if not len(idx):
            return wave
        x = wave[idx]
        r = self.rirs[self.randint(len(self.rirs), (len(idx),))]
        n = x.shape[1] + r.shape[1]
        y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(r, n), n)[:, :x.shape[1]]
        rms = lambda v: v.square().mean(1, keepdim=True).sqrt() + 1e-9
        wave = wave.clone()
        wave[idx] = y * rms(x) / rms(y)
        return wave

    def active_rms(self, wave):
        """RMS of the louder half of 20 ms frames: speech level without the silence around it."""
        e = wave[:, :wave.shape[1] // 320 * 320].reshape(len(wave), -1, 320).square().mean(2)
        top = e.sort(1, descending=True)[0][:, :max(1, e.shape[1] // 2)]
        return top.mean(1).sqrt() + 1e-6

    def add_noise(self, wave, keep, speech_rms=None):
        if self.noise is None:
            return wave
        b, n = wave.shape
        start = (self.rand(b) * (len(self.noise) - n - 1)).long()
        seg = self.noise[start[:, None] + torch.arange(n, device=self.device)[None]].float()
        lo, hi = self.cfg['snr_db']
        snr = lo + (hi - lo) * self.rand(b)
        ref = self.active_rms(wave) if speech_rms is None else speech_rms
        scale = ref / (seg.square().mean(1).sqrt() + 1e-6) * 10 ** (-snr / 20)
        return wave + torch.where(keep, scale, 0.)[:, None] * seg

    def __call__(self, wave, labels_silence=None, frontend='logmel', pooled=False, keep_mel=False):
        """(B, L) float waveforms -> features. pooled=True: release window features (1 s input)."""
        wave, mic = self.waveform(wave, labels_silence)
        return self.spectral(wave, mic, frontend, pooled, keep_mel)

    def mic_power(self, power, mic):
        """Apply the training microphones' |H|^2 to a (B, T, 257) power spectrum; mic < 0 = none."""
        has = mic >= 0
        return power * torch.where(has[:, None, None], self.mic_h[mic.clamp_min(0)][:, None], 1.)

    def spectral(self, wave, mic, frontend='logmel', pooled=False, keep_mel=False):
        """Microphone EQ, VTLP, front end and SpecAugment for waveforms from waveform()."""
        b, c = len(wave), self.cfg
        power = self.mic_power(self.front.power(wave), mic)
        vt = self.rand(b) < c['p_vtlp']
        ids = torch.where(vt, self.randint(len(self.front.alphas), (b,)), len(self.front.alphas) // 2)
        mel = self.front.mel(power, ids)
        feats = self.front.pooled(mel) if pooled else self.front.frames(mel, frontend)
        if not pooled:
            feats = self.specaugment(feats, self.rand(b) < c['p_specaug'])
        return (feats, mel) if keep_mel else feats

    def waveform(self, wave, labels_silence=None):
        """Speed, room, noise, level and microphone clipping; returns the waveforms and
        each example's training microphone (-1 = none) for spectral()."""
        b = len(wave)
        c = self.cfg
        wave = self.speed(wave, self.rand(b) < c['p_speed'])
        wave = self.room(wave, self.rand(b) < c['p_room'])
        speech_rms = self.active_rms(wave)
        if labels_silence is not None:  # silence examples: noise only, at a speech-like level
            wave = torch.where(labels_silence[:, None], 0., wave)
            speech_rms = torch.where(labels_silence, .05 * self.rand(b) + .005, speech_rms)
        wave = self.add_noise(wave, (self.rand(b) < c['p_noise']) | (labels_silence if labels_silence is not None else False),
                              speech_rms)
        lo, hi = c['level_db']
        wave = wave * 10 ** ((lo + (hi - lo) * self.rand(b)) / 20)[:, None]
        mic = torch.where(self.rand(b) < c['p_mic'], self.randint(len(self.mic_h), (b,)), -1)
        has = mic >= 0
        g = torch.where(has, self.mic_gain[mic.clamp_min(0)], 1.)
        k = torch.where(has, self.mic_drive[mic.clamp_min(0)], 1.)
        wave = torch.where(has[:, None], torch.tanh(k[:, None] * g[:, None] * wave) / k[:, None], wave)
        return wave, mic

    def specaugment(self, x, keep):
        b, t, m = x.shape
        x = x.clone()
        ar_t = torch.arange(t, device=self.device)[None]
        ar_m = torch.arange(m, device=self.device)[None]
        for _ in range(2):
            w = self.randint(4, (b,)); s = (self.rand(b) * (t - w)).long()
            mask = keep[:, None] & (ar_t >= s[:, None]) & (ar_t < (s + w)[:, None])
            x = torch.where(mask[:, :, None], 0., x)
            w = self.randint(4, (b,)); s = (self.rand(b) * (m - w)).long()
            mask = keep[:, None] & (ar_m >= s[:, None]) & (ar_m < (s + w)[:, None])
            x = torch.where(mask[:, None, :], 0., x)
        return x
