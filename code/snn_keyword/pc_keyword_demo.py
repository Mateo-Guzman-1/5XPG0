#!/usr/bin/env python3
"""pc_keyword_demo.py — SKELETON for GROUP 2 (PC side).

Captures microphone audio, turns each short window into a small mel
spectrogram "frame" and publishes it over Ethernet with ZeroMQ. The board
(see the README) is meant to subscribe, run the SNN and flash an LED.

This is an un-finished skeleton: it captures + encodes + sends, but the
packet format, the frame rate, the size and the board-side handling are all
DESIGN CHOICES left to you.

Needs on the PC:  pip install sounddevice numpy pyzmq

Run:  python pc_keyword_demo.py <board-ip>
"""

import sys
import time

import numpy as np
import zmq

try:
    import sounddevice as sd
except ImportError:
    sd = None

SAMPLE_RATE = 16000
WIN_MS = 500            # 0.5 s window
HOP_MS = 250            # send a frame every 0.25 s
N_MELS = 16             # keep it small: the board is a tiny CPU
PORT = 5556


def mel_spectrogram(wav):
    """Very small hand-rolled mel-ish spectrogram -> [N_MELS, n_frames], [0,1]."""
    win = int(SAMPLE_RATE * 0.025)
    hop = int(SAMPLE_RATE * 0.010)
    frames = [wav[i:i + win] for i in range(0, len(wav) - win, hop)]
    if not frames:
        return np.zeros((N_MELS, 1), dtype=np.float32)
    spec = np.abs(np.fft.rfft(np.stack(frames) * np.hanning(win), axis=1))
    # crude band grouping instead of a real mel filterbank (a design choice!)
    spec = spec[:N_MELS * 8].reshape(len(frames), N_MELS, 8).mean(axis=2)
    spec = np.log1p(spec).T                     # [N_MELS, n_frames]
    m = spec.max()
    return (spec / m).astype(np.float32) if m > 0 else spec.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print("usage: pc_keyword_demo.py <board-ip>")
        return 2
    board_ip = sys.argv[1]
    if sd is None:
        print("sounddevice not installed: pip install sounddevice")
        return 1

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{PORT}")
    print(f"publishing spectrogram frames on tcp://*:{PORT} "
          f"(board {board_ip} should SUB to it)")

    win = int(SAMPLE_RATE * WIN_MS / 1000)
    hop = int(SAMPLE_RATE * HOP_MS / 1000)

    def callback(indata, frames, t, status):
        callback.buf = np.append(callback.buf, indata[:, 0])
        if len(callback.buf) >= win:
            frame = mel_spectrogram(callback.buf[-win:])
            # payload = int32 shape (mels, n_frames) + float32 data, little endian
            hdr = np.array(frame.shape, dtype="<i4").tobytes()
            sock.send(hdr + frame.astype("<f4").tobytes())

    callback.buf = np.zeros(0, dtype=np.float32)

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=hop, callback=callback):
        while True:
            time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
