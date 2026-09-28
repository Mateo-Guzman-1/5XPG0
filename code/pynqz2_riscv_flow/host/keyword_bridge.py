#!/usr/bin/env python3
"""keyword_bridge.py — PC spectrogram stream -> RISC-V frame buffer.

Runs ON THE PYNQ-Z2 (PYNQ venv python, needs sudo for /dev/mem), after the
bitstream and firmware are loaded and the CPU is started.

Listens (ZeroMQ SUB, bound on the board) for the frames that
code/snn_keyword/pc_keyword_demo.py on the PC publishes to it. The board
binds and the PC connects, so no inbound firewall rule is needed on the PC.
Every frame is quantized to uint8 exactly like
code/snn_keyword/export_weights.py (min(255, round(x*256))), written into
the BRAM frame buffer, and the RISC-V result + console output is printed.

Usage:  keyword_bridge.py [--port 5556] [-v]
"""

import argparse
import struct
import sys

import numpy as np
import zmq

from spike_pynq import FRAME_BYTES, Bram


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print the result of every frame")
    a = p.parse_args()

    bram = Bram()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)          # always take the newest frame
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.bind(f"tcp://*:{a.port}")
    print(f"waiting for frames on tcp://*:{a.port} - start "
          f"pc_keyword_demo.py <board-ip> on the PC, then say 'sheila'")

    n = 0
    try:
        while True:
            if not sock.poll(2000):
                print("... no frames from the PC (is pc_keyword_demo.py running?)")
                continue
            msg = sock.recv()
            mels, frames = struct.unpack_from("<ii", msg)
            x = np.frombuffer(msg, dtype="<f4", offset=8)
            if x.size != FRAME_BYTES or mels * frames != FRAME_BYTES:
                print(f"skipping frame of shape {mels}x{frames} "
                      f"(firmware expects {FRAME_BYTES} values)")
                continue
            xq = np.minimum(255, np.round(x * 256)).astype(np.uint8)

            r = bram.frame_result(bram.write_frame(xq.tobytes()))
            n += 1
            if a.verbose or r["led"]:
                ms = r["cycles"] / 100e3
                print(f"[{n:5d}] keyword={r['keyword']:2d} other={r['other']:2d}"
                      f"  {ms:5.1f} ms" + ("  -> LED" if r["led"] else ""))

            out = bram.console_read()
            if out:
                sys.stdout.write(out.decode("ascii", "replace"))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
