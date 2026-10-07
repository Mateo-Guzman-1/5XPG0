"""Bounded request/reply TCP framing. No pickle or arbitrary shape allocation.

ABI v2 (window models, firmware magic "KWS1"): request magic KWS1, one
N_INPUT-byte window per request, modes single (0) and stream "2 of 3" (1).

ABI v3 (streaming models, firmware magic "KWS3", IMPLEMENTATION_PLAN.md
Phase 5): request magic KWS3 and one of
  MODE_INFO    no payload; reply INFO: ABI, frame bytes, threshold, max frames
  MODE_FRAMES  k whole frames of FRAME_BYTES (the 250 ms hop is 25); the
               network state persists on the core between requests
  MODE_RESET   no payload; clears the network state
A request for the other ABI than the loaded firmware gets status 2.
"""
import struct
from features import N_INPUT
MAGIC = b'KWS1'
MAGIC3 = b'KWS3'
# magic, request ID, payload byte count, mode. Clients predating the mode field
# sent the count as uint32, which reads as mode 0 (single window).
HEADER = struct.Struct('<4sIHH')
RESPONSE = struct.Struct('<4sIIiiIII')  # magic, id, status, scores, cycles, spikes, detected
MODE_SINGLE, MODE_STREAM = 0, 1
# Stream mode: detected bit0 = confirmed ("2 of 3" windows), bit1 = this window alone.
MODE_INFO, MODE_FRAMES, MODE_RESET = 2, 3, 4
FRAME_BYTES = 24
# magic, id, status, abi (2 or 3), input bytes (v2) or frame bytes (v3), threshold, max frames (v3)
INFO = struct.Struct('<4sIIIIiI')
# magic, id, status, best score, last score, cycles, spikes, detected, synaptic events, frame of detection (-1: none)
RESPONSE3 = struct.Struct('<4sIIiiIIIIi')
STATUS_OK, STATUS_BAD, STATUS_WRONG_ABI = 0, 1, 2


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError('Connection closed during packet')
        data.extend(chunk)
    return bytes(data)


def request(sock, sequence, frame, mode=MODE_SINGLE):
    payload = bytes(frame)
    if len(payload) != N_INPUT:
        raise ValueError(f'Expected {N_INPUT} feature bytes')
    sock.sendall(HEADER.pack(MAGIC, sequence, len(payload), mode) + payload)
    magic, seq, status, s0, s1, cycles, spikes, detected = RESPONSE.unpack(recv_exact(sock, RESPONSE.size))
    if magic != MAGIC or seq != sequence:
        raise ValueError('Mismatched response')
    if status:
        raise RuntimeError(f'Board returned error {status}')
    window = detected >> 1 & 1 if mode == MODE_STREAM else detected & 1
    return dict(sequence=seq, score0=s0, score1=s1, cycles=cycles, spikes=spikes,
                detected=bool(detected & 1), window=bool(window))


def info(sock, sequence):
    """ABI of the loaded model: {'abi': 2 or 3, 'input_bytes' or 'frame_bytes', 'threshold', 'max_frames'}."""
    sock.sendall(HEADER.pack(MAGIC3, sequence, 0, MODE_INFO))
    magic, seq, status, abi, size, threshold, max_frames = INFO.unpack(recv_exact(sock, INFO.size))
    if magic != MAGIC3 or seq != sequence or status:
        raise ValueError('Bad INFO response')
    key = 'frame_bytes' if abi == 3 else 'input_bytes'
    return {'abi': abi, key: size, 'threshold': threshold, 'max_frames': max_frames}


def request_frames(sock, sequence, frames, mode=MODE_FRAMES):
    """Send whole frames (or a reset with no payload); returns the hop result."""
    payload = bytes(frames)
    if mode == MODE_FRAMES and (not payload or len(payload) % FRAME_BYTES):
        raise ValueError(f'Expected whole {FRAME_BYTES}-byte frames')
    sock.sendall(HEADER.pack(MAGIC3, sequence, len(payload), mode) + payload)
    magic, seq, status, best, last, cycles, spikes, detected, events, at = RESPONSE3.unpack(
        recv_exact(sock, RESPONSE3.size))
    if magic != MAGIC3 or seq != sequence:
        raise ValueError('Mismatched response')
    if status:
        raise RuntimeError(f'Board returned error {status}')
    return dict(sequence=seq, best=best, last=last, cycles=cycles, spikes=spikes, events=events,
                detected=bool(detected & 1), above=bool(detected & 2), at_frame=at)
