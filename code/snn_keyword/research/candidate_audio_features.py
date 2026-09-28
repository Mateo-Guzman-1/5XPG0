"""Board-side packet definition for the selected 8-by-16 Sheila candidate.

Deploy this file as ``audio_features.py`` beside ``keyword_bridge.py`` while
benchmarking the candidate firmware.  Feature extraction remains on the PC;
the board only validates and unpacks the uint8 feature frame.
"""

from __future__ import annotations

import struct

import numpy as np


N_MELS = 8
N_TIME = 16
N_INPUTS = N_MELS * N_TIME
PACKET_MAGIC = b"KWS1"
PACKET_HEADER = struct.Struct("<4sIHH")


def unpack_frame(packet: bytes) -> tuple[int, np.ndarray]:
    if len(packet) < PACKET_HEADER.size:
        raise ValueError("packet is shorter than the keyword header")
    magic, sequence, rows, cols = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC:
        raise ValueError(f"bad packet magic {magic!r}")
    if (rows, cols) != (N_MELS, N_TIME):
        raise ValueError(f"unsupported feature shape {(rows, cols)}")
    payload = packet[PACKET_HEADER.size:]
    if len(payload) != N_INPUTS:
        raise ValueError(f"expected {N_INPUTS} payload bytes, got {len(payload)}")
    return sequence, np.frombuffer(payload, dtype=np.uint8).reshape(rows, cols)
