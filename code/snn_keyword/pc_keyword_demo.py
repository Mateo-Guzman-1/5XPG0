#!/usr/bin/env python3
"""Capture audio, create a fixed spectrogram, and classify it on the board.

The board-side ``keyword_bridge.py`` owns TCP port 5556. Each request carries
one 16x16 uint8 feature map; the reply contains the PicoRV32 decision, spike
counts, and measured inference cycles.

Examples:
    python pc_keyword_demo.py 192.168.2.99
    python pc_keyword_demo.py 192.168.2.99 --wav sample.wav
"""

from __future__ import annotations

import argparse
import json
import queue
import time

import numpy as np
import zmq

from audio_features import (
    N_MELS,
    N_TIME,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    extract_features,
    pack_frame,
    read_wav,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None


PORT = 5556
HOP_MS = 250


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """Compatibility name retained for notebooks using the old skeleton."""
    return extract_features(wav)


def make_socket(ctx: zmq.Context, board_ip: str, port: int) -> zmq.Socket:
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, 5_000)
    sock.setsockopt(zmq.RCVTIMEO, 5_000)
    sock.connect(f"tcp://{board_ip}:{port}")
    return sock


def classify(sock: zmq.Socket, wav: np.ndarray, sequence: int) -> dict:
    feature = extract_features(wav)
    sock.send(pack_frame(feature, sequence))
    try:
        return json.loads(sock.recv().decode("utf-8"))
    except zmq.Again as exc:
        raise TimeoutError("board did not answer within 5 seconds") from exc


def print_result(result: dict, quiet_negatives: bool = False) -> None:
    if "error" in result:
        raise RuntimeError(f"board rejected the frame: {result['error']}")
    if quiet_negatives and not result.get("keyword"):
        return
    label = "KEYWORD" if result.get("keyword") else "not-keyword"
    print(
        f"frame={result.get('sequence')}  {label:11s}  "
        f"spikes=[{result.get('score_not')},{result.get('score_keyword')}]  "
        f"cycles={result.get('cycles')}  ms={result.get('inference_ms'):.2f}  "
        f"led={result.get('led')}"
    )


def wav_mode(sock: zmq.Socket, path: str) -> int:
    result = classify(sock, read_wav(path), 1)
    print_result(result)
    return 0


def microphone_mode(
    sock: zmq.Socket,
    duration: float | None,
    quiet_negatives: bool,
) -> int:
    if sd is None:
        print("sounddevice is not installed; run setup_venv.ps1 or use --wav")
        return 1

    blocks: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
    hop = int(SAMPLE_RATE * HOP_MS / 1000)

    def callback(indata, frames, timing, status):
        if status:
            print(f"audio status: {status}")
        blocks.put(indata[:, 0].copy())

    samples = np.zeros(0, dtype=np.float32)
    sequence = 0
    started = time.monotonic()
    print(
        f"listening at {SAMPLE_RATE} Hz; sending {N_MELS}x{N_TIME} frames "
        f"to the board every {HOP_MS} ms (Ctrl-C to stop)"
    )

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=hop,
            dtype="float32",
            callback=callback,
        ):
            while duration is None or time.monotonic() - started < duration:
                samples = np.append(samples, blocks.get())
                if samples.size < WINDOW_SAMPLES:
                    continue
                samples = samples[-WINDOW_SAMPLES:]
                sequence += 1
                print_result(classify(sock, samples, sequence), quiet_negatives)
    except KeyboardInterrupt:
        print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("board_ip")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--wav", help="classify one WAV file instead of the microphone")
    p.add_argument("--duration", type=float, help="stop microphone mode after N seconds")
    p.add_argument("--quiet-negatives", action="store_true")
    args = p.parse_args()

    ctx = zmq.Context()
    sock = make_socket(ctx, args.board_ip, args.port)
    try:
        if args.wav:
            return wav_mode(sock, args.wav)
        return microphone_mode(sock, args.duration, args.quiet_negatives)
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    raise SystemExit(main())
