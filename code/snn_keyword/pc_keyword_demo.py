#!/usr/bin/env python3
"""pc_keyword_demo.py — GROUP 2, PC side of the "sheila" demo.

Captures microphone audio, turns each 1 s window into a 16x16 spectrogram
frame (features.py, same as training) every 0.25 s and publishes it over
Ethernet with ZeroMQ. On the board, host/keyword_bridge.py receives and
hands the frames to the RISC-V SNN, which lights the LEDs on "sheila".

Packet: int32 shape (mels, frames) + float32 data, little endian.

Needs on the PC:  pip install sounddevice numpy pyzmq

Run:  python pc_keyword_demo.py <board-ip>
      python pc_keyword_demo.py --local     # no board: run the integer
                                            # firmware model on the PC
"""

import sys
import time

import numpy as np
import zmq

try:
    import sounddevice as sd
except ImportError:
    sd = None

# same front-end as training, so the board sees what the SNN was trained on
from features import CLIP_SAMPLES, SAMPLE_RATE, clip_features

HOP_MS = 250            # send a frame every 0.25 s
PORT = 5556
KW_CONSECUTIVE = 2      # same rule as firmware/main.c (only for --local)


def local_classifier():
    """Bit-exact PC copy of the firmware SNN, for testing without a board."""
    import os
    import torch
    from export_weights import int_forward, quantize, quantize_input
    from train_keyword_snn import OUT_DIR, SpikeMLP

    net = SpikeMLP()
    net.load_state_dict(torch.load(os.path.join(OUT_DIR, "keyword_snn.pt")))
    p = quantize(net)
    state = {"streak": 0}

    def classify(frame):
        c = int_forward(p, quantize_input(frame.reshape(1, -1)))[0]
        state["streak"] = state["streak"] + 1 if c[1] > c[0] else 0
        bar = "#" * int(c[1])
        print(f"keyword={c[1]:2d} other={c[0]:2d} {bar}")
        if state["streak"] == KW_CONSECUTIVE:
            print(">>> sheila! (LED would light for 1 s)")

    return classify


def main():
    if len(sys.argv) < 2:
        print("usage: pc_keyword_demo.py <board-ip> | --local")
        return 2
    if sd is None:
        print("sounddevice not installed: pip install sounddevice")
        return 1

    if sys.argv[1] == "--local":
        handle = local_classifier()
        print("local mode: say 'sheila' (Ctrl-C to stop)")
    else:
        # the board binds (host/keyword_bridge.py), the PC connects: this
        # needs no inbound firewall rule on the PC
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.connect(f"tcp://{sys.argv[1]}:{PORT}")
        print(f"sending spectrogram frames to tcp://{sys.argv[1]}:{PORT} "
              f"- say 'sheila' (Ctrl-C to stop)")

        def handle(frame):
            hdr = np.array(frame.shape, dtype="<i4").tobytes()
            sock.send(hdr + frame.astype("<f4").tobytes())

    win = CLIP_SAMPLES      # 1 s sliding window, like the training clips
    hop = int(SAMPLE_RATE * HOP_MS / 1000)

    # the audio callback only buffers; frames are built in the main loop
    buf = {"wav": np.zeros(0, dtype=np.float32), "new": False}

    def callback(indata, frames, t, status):
        buf["wav"] = np.append(buf["wav"], indata[:, 0])[-win:]
        buf["new"] = True

    # a muted mic still delivers audio, just ~1e-4 full scale: every frame
    # then looks like silence and nothing is ever detected. Warn about it.
    quiet_since = time.time()
    warned = False
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                            blocksize=hop, callback=callback):
            while True:
                time.sleep(0.01)
                if buf["new"] and len(buf["wav"]) >= win:
                    buf["new"] = False
                    handle(clip_features(buf["wav"]))   # [N_MELS, N_TIME]
                    if np.abs(buf["wav"]).max() > 1e-3:
                        quiet_since = time.time()
                        warned = False
                    elif time.time() - quiet_since > 5 and not warned:
                        warned = True
                        print("WARNING: microphone is (almost) silent - is it "
                              "muted? Check the mic-mute key and Windows "
                              "Settings > System > Sound > Input volume.")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    raise SystemExit(main())
