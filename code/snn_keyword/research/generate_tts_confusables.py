"""Generate a voice-disjoint Kokoro Sheila/confusable corpus (mono 16 kHz)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import wave

import numpy as np

VOICES = {
    "train": ["af_heart", "af_sarah", "af_bella", "af_nicole", "am_adam", "am_michael", "bf_emma", "bm_george"],
    "val": ["af_sky", "am_fenrir"],
    "test": ["af_river", "am_puck", "bf_isabella", "bm_lewis"],
}
WORDS = ["Sheila", "she", "sheep", "shield", "she'll", "Shelley", "she likes", "shilling", "shelter", "sealer"]
SPEEDS = [0.85, 1.0, 1.15]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(samples, -0.999, 0.999) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, len(pcm), "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())


def fit_speech(audio: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    # Anti-alias before resampling the model's 24 kHz output to 16 kHz.
    axis = np.arange(-48, 49)
    cutoff = 7400 / rate
    kernel = 2 * cutoff * np.sinc(2 * cutoff * axis) * np.hanning(len(axis))
    kernel /= kernel.sum()
    filtered = np.convolve(audio, kernel, mode="same")
    target = np.arange(round(len(audio) * 16000 / rate)) * rate / 16000
    resampled = np.interp(target, np.arange(len(audio)), filtered)
    active = np.flatnonzero(np.abs(resampled) > max(0.001, 0.015 * np.max(np.abs(resampled))))
    if not active.size:
        raise ValueError("TTS produced empty speech")
    speech = resampled[max(0, active[0] - 480): min(len(resampled), active[-1] + 481)]
    duration = len(speech) / 16000
    # Preserve the whole phrase rather than cutting off a disambiguating suffix.
    if len(speech) > 14080:
        speech = np.interp(np.linspace(0, len(speech) - 1, 14080), np.arange(len(speech)), speech)
    return speech.astype(np.float32), duration


def main() -> None:
    from kokoro_onnx import Kokoro
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.assets / "kokoro-v1.0.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    engine = Kokoro.from_session(session, str(args.assets / "voices-v1.0.bin"))
    rows = []
    for split, voices in VOICES.items():
        for voice in voices:
            for word_id, word in enumerate(WORDS):
                for speed_id, speed in enumerate(SPEEDS):
                    stem = f"{voice}_{word_id}_{speed_id}"
                    raw_path = args.output / "raw" / f"{stem}.wav"
                    if raw_path.exists():
                        with wave.open(str(raw_path), "rb") as stream:
                            rate = stream.getframerate()
                            audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
                    else:
                        audio, rate = engine.create(word + ".", voice=voice, speed=speed, lang="en-gb" if voice.startswith("b") else "en-us")
                        write_wav(raw_path, audio, rate)
                        print(f"generated {split} {stem} {word}", flush=True)
                    # Process the persisted PCM in both fresh and resumed runs.
                    with wave.open(str(raw_path), "rb") as stream:
                        rate = stream.getframerate()
                        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
                    speech, original_duration = fit_speech(audio, rate)
                    for variant, (rms, offset, snr) in enumerate([(0.012, -640, None), (0.035, 0, 30), (0.09, 640, 20)]):
                        frame = np.zeros(16000, np.float32)
                        start = (16000 - len(speech)) // 2 + offset
                        start = int(np.clip(start, 0, 16000 - len(speech)))
                        frame[start:start + len(speech)] = speech
                        frame *= rms / max(1e-8, np.sqrt(np.mean(frame ** 2)))
                        seed = int(hashlib.sha256(f"{stem}_{variant}".encode()).hexdigest()[:8], 16)
                        if snr is not None:
                            noise = np.random.default_rng(seed).normal(size=16000).astype(np.float32)
                            noise *= rms / (10 ** (snr / 20) * np.sqrt(np.mean(noise ** 2)))
                            frame += noise
                        path = args.output / "clips" / split / f"{stem}_{variant}.wav"
                        write_wav(path, frame, 16000)
                        rows.append({"path": path.relative_to(args.output).as_posix(), "raw_path": raw_path.relative_to(args.output).as_posix(), "voice": voice, "split": split, "text": word, "label": int(word == "Sheila"), "speed": speed, "variant": variant, "rms": rms, "snr_db": snr, "source_speech_duration_s": original_duration, "duration_compressed": original_duration > 0.88, "sha256": sha(path)})
    manifest = {"engine": "kokoro-onnx", "engine_version": importlib.metadata.version("kokoro-onnx"), "model_source": "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1", "model_sha256": sha(args.assets / "kokoro-v1.0.onnx"), "voices_sha256": sha(args.assets / "voices-v1.0.bin"), "voice_splits": VOICES, "words": WORDS, "clips": rows, "limitations": ["Synthetic held-out voices are not independent real speakers.", "Long utterances are compressed to 880 ms before centering; raw audio is retained.", "Procedural gain, shift and white noise variants share their source utterance."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} clips to {args.output}", flush=True)


if __name__ == "__main__":
    main()
