"""Board test over the TCP protocol: board_server.py on PYNQ (Ethernet) or jtag_server.py.

Asks the server which ABI is loaded (INFO), then
  ABI v2: sends the 40 verification windows (results/verification_vectors.npz)
          and checks scores, spikes and the decision against the integer oracle;
  ABI v3: resets and streams the 40 test streams of jtag/make_stream_vectors.py
          in 25-frame hops and checks every hop against the oracle, then checks
          that a request for the other ABI is refused.
Writes a CSV of the per-request results (with cycles and round-trip time).
"""
import argparse
import csv
from pathlib import Path
import socket
import sys
import time

import numpy as np

import protocol
from model import integer_forward

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'jtag'))


def run_v2(sock, q, writer):
    x = np.load(ROOT / 'results/verification_vectors.npz')['x']
    scores, spikes = integer_forward(x, q)
    th = int(q['decision_threshold'])
    for i, (v, (s0, s1), k) in enumerate(zip(x, scores, spikes)):
        t = time.perf_counter()
        r = protocol.request(sock, 1000 + i, v.tobytes())
        rtt = time.perf_counter() - t
        got, want = (r['score0'], r['score1'], r['spikes'], r['detected']), (int(s0), int(s1), int(k), bool(s1 - s0 >= th))
        if got != want:
            raise AssertionError(f'vector {i}: got {got} expected {want}')
        writer.writerow([i, 0, r['score0'], r['score1'], r['spikes'], r['cycles'], int(r['detected']), round(rtt * 1e3, 2)])
    return len(x)


def prepare_v3(q, data, streams, hop):
    from make_stream_vectors import expected_hops, test_streams
    return [(x, expected_hops(x, q, hop)) for x in test_streams(q, data, streams)]


def run_v3(sock, writer, prepared):
    seq, n = 2000, 0
    for record, (x, expected) in enumerate(prepared):
        protocol.request_frames(sock, seq, b'', protocol.MODE_RESET); seq += 1
        for i, h in enumerate(expected):
            t = time.perf_counter()
            r = protocol.request_frames(sock, seq, x[h['start']:h['start'] + h['frames']].tobytes()); seq += 1
            rtt = time.perf_counter() - t
            at = -1 if h['at'] == 0xffffffff else h['at']
            got = (r['best'], r['last'], r['spikes'], r['detected'], r['above'], r['at_frame'])
            want = (h['best'], h['last'], h['spikes'], bool(h['detected'] & 1), bool(h['detected'] & 2), at)
            if got != want:
                raise AssertionError(f'stream {record} hop {i}: got {got} expected {want}')
            writer.writerow([record, i, r['best'], r['last'], r['spikes'], r['cycles'], int(r['detected']), round(rtt * 1e3, 2)])
            n += 1
    # A v2 window request to a v3 backend is refused (the server then closes the connection).
    sock.sendall(protocol.HEADER.pack(protocol.MAGIC, seq, 768, protocol.MODE_SINGLE) + bytes(768))
    status = protocol.RESPONSE.unpack(protocol.recv_exact(sock, protocol.RESPONSE.size))[2]
    if status != protocol.STATUS_WRONG_ABI:
        raise AssertionError(f'v2 request to v3 firmware returned status {status}')
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('host')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--model', type=Path, help='default: deploy/model.npz (v2, the "yes" release) or '
                   'results/models/sheila_stream_int8.npz (v3)')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--streams', type=int, default=40)
    p.add_argument('--hop', type=int, default=25)
    p.add_argument('--out', type=Path, default=ROOT / 'build/net_board.csv')
    a = p.parse_args()
    with socket.create_connection((a.host, a.port), timeout=30) as sock:
        info = protocol.info(sock, 1)
    print('INFO', info, flush=True)
    default = ROOT / ('deploy/model.npz' if info['abi'] == 2 else 'results/models/sheila_stream_int8.npz')
    q = dict(np.load(a.model or default))
    threshold = int(q['decision_threshold'] if info['abi'] == 2 else q['stream_threshold'])
    if info['threshold'] != threshold:
        raise AssertionError(f"board threshold {info['threshold']} != model {threshold}")
    # Expectations first: the server drops a connection that stays idle for 10 s.
    prepared = prepare_v3(q, a.data, a.streams, a.hop) if info['abi'] == 3 else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with socket.create_connection((a.host, a.port), timeout=30) as sock:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            if info['abi'] == 2:
                w.writerow(['index', 'hop', 'score0', 'score1', 'spikes', 'cycles', 'detected', 'rtt_ms'])
                n = run_v2(sock, q, w)
            else:
                w.writerow(['record', 'hop', 'best', 'last', 'spikes', 'cycles', 'detected', 'rtt_ms'])
                n = run_v3(sock, w, prepared)
    rows = np.genfromtxt(a.out, delimiter=',', names=True)
    print(f"PASS: {n} requests (ABI v{info['abi']}) match the integer oracle; "
          f"cycles max {int(rows['cycles'].max())}, round trip median {np.median(rows['rtt_ms']):.1f} ms, "
          f"max {rows['rtt_ms'].max():.1f} ms -> {a.out}")


if __name__ == '__main__':
    main()
