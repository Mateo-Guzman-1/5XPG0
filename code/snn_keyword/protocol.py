"""Bounded request/reply TCP framing. No pickle or arbitrary shape allocation."""
import struct
from features import N_INPUT
MAGIC = b'KWS1'
HEADER = struct.Struct('<4sII')  # magic, request ID, feature byte count
RESPONSE = struct.Struct('<4sIIiiIII')  # magic, id, status, scores, cycles, spikes, detected


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError('Connection closed during packet')
        data.extend(chunk)
    return bytes(data)


def request(sock, sequence, frame):
    payload = bytes(frame)
    if len(payload) != N_INPUT:
        raise ValueError(f'Expected {N_INPUT} feature bytes')
    sock.sendall(HEADER.pack(MAGIC, sequence, len(payload)) + payload)
    magic, seq, status, s0, s1, cycles, spikes, detected = RESPONSE.unpack(recv_exact(sock, RESPONSE.size))
    if magic != MAGIC or seq != sequence:
        raise ValueError('Mismatched response')
    if status:
        raise RuntimeError(f'Board returned error {status}')
    return dict(sequence=seq, score0=s0, score1=s1, cycles=cycles, spikes=spikes, detected=bool(detected))
