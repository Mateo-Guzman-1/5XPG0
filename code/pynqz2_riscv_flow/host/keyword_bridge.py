#!/usr/bin/env python3
"""ZeroMQ-to-BRAM bridge for the Group-2 keyword detector.

Run this on the PYNQ Linux processing system as root.  It receives a compact
spectrogram from the PC, copies it into dual-port BRAM, asks the PicoRV32 to
classify it through the mailbox, and returns the result to the PC.
"""

from __future__ import annotations

import argparse
import json
import sys

import zmq

# audio_features.py is deployed alongside this script.
from audio_features import N_INPUTS, unpack_frame
from spike_pynq import Bram, Regs


INPUT_FRAME_BASE = 0x22000
MB_CMD_CLASSIFY = 8
CLK_HZ = 100_000_000


def signed32(value: int) -> int:
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


def classify(bram: Bram, regs: Regs, packet: bytes) -> dict:
    sequence, frame = unpack_frame(packet)
    payload = frame.tobytes(order="C")
    if len(payload) != N_INPUTS:
        raise ValueError("unexpected feature payload size")

    bram.m[INPUT_FRAME_BASE:INPUT_FRAME_BASE + len(payload)] = payload
    result = bram.mb_cmd(
        MB_CMD_CLASSIFY,
        [INPUT_FRAME_BASE, len(payload), sequence],
    )
    decision, score_not, score_keyword, cycles = result
    if decision == 0xFFFFFFFF:
        raise RuntimeError(
            f"firmware rejected frame address/size: {score_not:#x}/{score_keyword:#x}"
        )
    return {
        "sequence": sequence,
        "keyword": bool(decision),
        "score_not": signed32(score_not),
        "score_keyword": signed32(score_keyword),
        "cycles": cycles,
        "inference_ms": cycles * 1000.0 / CLK_HZ,
        "led": regs.dump()["LED"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="tcp://*:5556")
    p.add_argument("--once", action="store_true", help="serve one frame, then exit")
    args = p.parse_args()

    regs = Regs()
    status = regs.dump()
    if status["MAGIC"] != "0x534b454c" or status["STATUS"] != "0x1":
        raise RuntimeError(f"keyword firmware is not running: {status}")
    bram = Bram()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.bind)
    print(f"keyword bridge listening on {args.bind}", flush=True)

    try:
        while True:
            packet = sock.recv()
            try:
                response = classify(bram, regs, packet)
                print(json.dumps(response, sort_keys=True), flush=True)
            except Exception as exc:
                response = {"error": str(exc)}
                print(f"frame error: {exc}", file=sys.stderr, flush=True)
            sock.send_json(response)
            if args.once:
                break
    finally:
        sock.close()
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
