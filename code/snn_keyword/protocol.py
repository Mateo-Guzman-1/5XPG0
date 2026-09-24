"""Bounded request/reply TCP framing. No pickle or arbitrary shape allocation."""
import struct
from features import N_INPUT
MAGIC = b'KWS1'
# magic, request ID, feature byte count, mode. Clients predating the mode field
# sent the count as uint32, which reads as mode 0 (single window).
HEADER = struct.Struct('<4sIHH')
RESPONSE = struct.Struct('<4sIIiiIII')  # magic, id, status, scores, cycles, spikes, detected
MODE_SINGLE, MODE_STREAM = 0, 1
# Stream mode: detected bit0 = confirmed ("2 of 3" windows), bit1 = this window alone.


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
