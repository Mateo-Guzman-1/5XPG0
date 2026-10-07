#!/usr/bin/env python3
"""Replay exported verification vectors through the Ethernet PYNQ service."""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import zmq


HEADER = struct.Struct("<4sIHH")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board")
    parser.add_argument("vectors")
    parser.add_argument("--bands", type=int, required=True)
    parser.add_argument("--time-bins", type=int, default=16)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    vectors = np.load(args.vectors)
    x = vectors["x"]
    expected = vectors["counts"]
    if x.shape[1] != args.bands * args.time_bins:
        raise ValueError("vector shape does not match bands * time bins")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10_000)
    socket.connect(f"tcp://{args.board}:5556")
    actual = []
    cycles = []
    roundtrip_ms = []
    try:
        for index, frame in enumerate(x):
            packet = HEADER.pack(
                b"KWS1", index, args.bands, args.time_bins
            ) + frame.tobytes()
            started = time.perf_counter()
            socket.send(packet)
            response = socket.recv_json()
            roundtrip_ms.append(1000.0 * (time.perf_counter() - started))
            if "error" in response:
                raise RuntimeError(response["error"])
            actual.append([response["score_not"], response["score_keyword"]])
            cycles.append(response["cycles"])
    finally:
        socket.close()
        context.term()

    actual_array = np.asarray(actual)
    cycle_array = np.asarray(cycles)
    roundtrip_array = np.asarray(roundtrip_ms)
    result = {
        "label": args.label,
        "vectors": len(x),
        "score_mismatches": int(np.any(actual_array != expected, axis=1).sum()),
        "cycles_min": int(cycle_array.min()),
        "cycles_mean": float(cycle_array.mean()),
        "cycles_max": int(cycle_array.max()),
        "milliseconds_mean_at_100mhz": float(cycle_array.mean() / 100_000),
        "milliseconds_max_at_100mhz": float(cycle_array.max() / 100_000),
        "ethernet_roundtrip_ms_mean": float(roundtrip_array.mean()),
        "ethernet_roundtrip_ms_max": float(roundtrip_array.max()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["score_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
