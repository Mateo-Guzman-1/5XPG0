"""Microphone and room channels, split between evaluation and training.

A microphone is a high-pass, a low-pass, three peaking EQs, a level change and
tanh soft clipping. Every corner and centre frequency is assigned to a
third-octave band b = round(3 log2(f / 1 kHz)):

  held out (robust_eval.py)      odd b; N_PROFILES fixed profiles from HELDOUT_SEED
  training (augment_online.py)   even b; drawn fresh per example

The held-out corners stay in RANGES['narrow']. There the only even low-pass
band is b = 8 (5.66-7.13 kHz), so training never sees a microphone rolling off
below 5.66 kHz, and held-out b = 7 (4.5-5.66 kHz, which removes most of the
/s/ of "yes") is an extrapolation. RANGES['wide'] (training only) brackets
every held-out band with even bands on both sides.

Rooms: the real RIRs of OpenSLR 28 (RWCP, AIR, REVERB challenge) are held out
for evaluation; training uses only its simulated RIRs. Speech is convolved
with the RIR aligned at the direct path and rescaled to its dry RMS, so the
test isolates reverberation from level.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

ROOT = Path(__file__).resolve().parent
RIR_ROOT = ROOT / 'data/rirs/RIRS_NOISES'
SAMPLE_RATE = 16000
N_PROFILES = 16
HELDOUT_SEED = 20260925


def band(f):
    return int(round(3 * np.log2(f / 1000)))


def peaking(f, gain_db, q):
    """RBJ cookbook peaking EQ as one second-order section."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f / SAMPLE_RATE
    alpha = np.sin(w) / (2 * q)
    b0, b1, b2 = 1 + alpha * a, -2 * np.cos(w), 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * np.cos(w), 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])


RANGES = {'narrow': dict(highpass=(60, 400), lowpass=(4500, 7600), peaks=(150, 6000)),
          'wide': dict(highpass=(40, 500), lowpass=(3500, 7900), peaks=(100, 7000))}


def draw_mic(rng, heldout, ranges='narrow'):
    """One microphone profile; frequencies restricted to odd (held out) or even (training) bands."""
    r = RANGES['narrow' if heldout else ranges]
    def freq(lo, hi):
        while True:
            f = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            if (band(f) % 2 == 1) == heldout:
                return f
    return dict(highpass=freq(*r['highpass']), lowpass=freq(*r['lowpass']),
                peaks=[(freq(*r['peaks']), float(rng.uniform(-9, 9)), float(rng.uniform(.7, 3))) for _ in range(3)],
                gain_db=float(rng.uniform(-12, 6)), drive=float(rng.uniform(1, 3)))


@lru_cache(maxsize=1)
def heldout_mics():
    rng = np.random.default_rng(HELDOUT_SEED)
    return tuple(draw_mic(rng, True) for _ in range(N_PROFILES))


def apply_mic(audio, mic):
    sos = np.concatenate([butter(2, mic['highpass'], 'highpass', fs=SAMPLE_RATE, output='sos'),
                          butter(4, mic['lowpass'], 'lowpass', fs=SAMPLE_RATE, output='sos')]
                         + [peaking(*p) for p in mic['peaks']])
    y = sosfilt(sos, audio) * 10 ** (mic['gain_db'] / 20)
    k = mic['drive']
    return (np.tanh(k * y) / k).astype(np.float32)


def _rir_files(real):
    if real:
        files = sorted(p for p in (RIR_ROOT / 'real_rirs_isotropic_noises').glob('*.wav') if '_rir_' in p.name)
    else:
        files = sorted((RIR_ROOT / 'simulated_rirs').rglob('*.wav'))
    if not files:
        raise FileNotFoundError(f'No RIRs under {RIR_ROOT}; run: python fetch_corpora.py rirs')
    return files


@lru_cache(maxsize=1)
def heldout_rirs():
    """Real RIRs, first channel, 16 kHz."""
    import soundfile as sf
    out = []
    for p in _rir_files(real=True):
        r, rate = sf.read(p, dtype='float32', always_2d=True)
        if rate == SAMPLE_RATE:
            out.append(r[:, 0])
    return tuple(out)


def training_rir_files():
    return _rir_files(real=False)


def apply_rir(audio, rir):
    """Reverberant copy, same length plus the tail, direct path at the original position, dry RMS."""
    d = int(np.argmax(np.abs(rir)))
    y = fftconvolve(audio, rir[d:])
    rms = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-9
    return (y * rms(audio) / rms(y[:len(audio)])).astype(np.float32)
