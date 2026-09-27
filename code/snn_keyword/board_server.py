"""PC feature receiver; use --simulate for local integer model replay.

Serves ABI v2 (window models, firmware magic KWS1) and ABI v3 (streaming
models, magic KWS3; protocol.py). The backend reports which one is loaded.
"""
import argparse
import mmap
import os
from pathlib import Path
import socket
import struct
import time
from protocol import (HEADER, RESPONSE, MAGIC, MODE_SINGLE, MODE_STREAM, recv_exact, MAGIC3, INFO, RESPONSE3,
                      MODE_INFO, MODE_FRAMES, MODE_RESET, FRAME_BYTES, STATUS_BAD, STATUS_WRONG_ABI)
from features import N_INPUT

ROOT = Path(__file__).resolve().parent
# Firmware magic (mailbox word 12) -> ABI; firmware/main.c is v2, firmware/stream_main.c v3.
FIRMWARE_ABI = {0x4b575331: 2, 0x4b575333: 3}
MAX_FRAMES = 100       # firmware/stream_main.c
HOLDOFF_FRAMES = 100   # 1 s, as robust_eval.py and the firmware


class Board:
    def __init__(self, firmware):
        from pynq import Clocks
        fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        try:
            self.ram = mmap.mmap(fd, 0x40000, offset=0x40000000)
            self.reg = mmap.mmap(fd, 4096, offset=0x40040000)
        finally:
            os.close(fd)
        if self.read_reg(0x14) != 0x534b454c:
            raise RuntimeError('Expected spike SoC bitstream is not loaded')
        abi = self.read_reg(0x1c)
        if abi & 0xffff0000 != 0x00020000:
            raise RuntimeError('Load deploy/keyword.bit: this bitstream lacks the timed LED ABI')
        # ABI bit0: kdot PCPI coprocessor present (deploy/keyword_kdot.bit).
        if 'kdot' in Path(firmware).name and not abi & 1:
            raise RuntimeError('kdot firmware needs deploy/keyword_kdot.bit (ABI bit0)')
        if self.read_reg(0x10) != 100000000:
            raise RuntimeError('Expected 100 MHz fabric clock')
        image = Path(firmware).read_bytes()
        if not 0 < len(image) <= 0x3c000:
            raise ValueError('Invalid firmware image size')
        struct.pack_into('<I', self.reg, 0, 1)
        # Configure the physical PS clock while the core is held in reset.
        Clocks.fclk0_mhz = 100.0
        if abs(Clocks.fclk0_mhz - 100.0) > .01:
            raise RuntimeError('Could not configure the physical FCLK0 to 100 MHz')
        self.ram[:] = bytes(0x40000)
        self.ram[:len(image)] = image
        struct.pack_into('<I', self.reg, 0, 0)
        deadline = time.monotonic() + 5
        while self.read(0x10430) not in FIRMWARE_ABI:
            if time.monotonic() > deadline:
                raise TimeoutError('Firmware did not initialize')
            time.sleep(.001)
        self.abi = FIRMWARE_ABI[self.read(0x10430)]

    def read_reg(self, offset):
        return struct.unpack_from('<I', self.reg, offset)[0]

    def read(self, offset):
        return struct.unpack_from('<I', self.ram, offset)[0]

    def write(self, offset, value):
        struct.pack_into('<I', self.ram, offset, value)

    def info(self):
        threshold = struct.unpack('<i', struct.pack('<I', self.read(0x10434)))[0]
        size, max_frames = self.read(0x10438) or N_INPUT, self.read(0x1043c)
        return self.abi, size, threshold, max_frames if self.abi == 3 else 0

    def infer(self, payload, opcode=1):
        if len(payload) != N_INPUT:
            raise ValueError('Incorrect feature payload size')
        self._command(opcode, payload)
        return struct.unpack_from('<IiiIII', self.ram, 0x10418)

    def frames(self, payload):
        """ABI v3 command 4: status, best, last, cycles, spikes, detected, events, frame of detection."""
        self._command(4, payload)
        status, best, last, cycles, spikes, detected = struct.unpack_from('<IiiIII', self.ram, 0x10418)
        events, at = struct.unpack_from('<Ii', self.ram, 0x10440)
        return status, best, last, cycles, spikes, detected, events, at

    def reset(self):
        self._command(5, b'')
        return (self.read(0x10418),) + (0,) * 6 + (-1,)

    def _command(self, opcode, payload):
        if self.read_reg(0) & 1:
            raise RuntimeError('Core is stopped; restart the board server')
        seq = (self.read(0x10400) + 1) & 0xffffffff
        self.ram[0x10800:0x10800+len(payload)] = payload
        self.write(0x10408, opcode); self.write(0x1040c, len(payload))
        self.write(0x10400, seq)  # publish last
        deadline = time.monotonic() + 5
        while self.read(0x10404) != seq:
            if self.read_reg(4) & 2:
                struct.pack_into('<I', self.reg, 0, 1)
                raise RuntimeError('RISC-V trap')
            if time.monotonic() >= deadline:
                struct.pack_into('<I', self.reg, 0, 1)
                raise RuntimeError('RISC-V inference timed out; core stopped, restart server')
            time.sleep(.001)


class SimulatedBoard:
    def __init__(self, model):
        import numpy as np
        from model import integer_forward
        self.np, self.forward = np, integer_forward
        self.q = dict(np.load(model))
        self.abi = 3 if str(self.q.get('kind', 'window')) == 'stream' else 2
        self.led_until = 0.
        self.history = [(False, 0.), (False, 0.)]   # previous two stream windows
        self.state, self.frames_done, self.last_event = None, 0, None
        self.window = int(self.q.get('decision_window', 1))
        self.recent = self.np.zeros(0, self.np.int64)   # raw scores for the moving-sum decision

    def info(self):
        if self.abi == 3:
            return 3, FRAME_BYTES, int(self.q['stream_threshold']), MAX_FRAMES
        return 2, N_INPUT, int(self.q['decision_threshold']), 0

    def frames(self, payload):
        """Mirrors firmware/stream_main.c command 4 (hold-off counted in frames)."""
        from model import decision_scores, integer_forward_stream
        x = self.np.frombuffer(payload, self.np.uint8).reshape(1, -1, FRAME_BYTES)
        scores, self.state, spikes = integer_forward_stream(x, self.q, self.state)
        th, detected, at = int(self.q['stream_threshold']), 0, -1
        decision = decision_scores(scores[0], self.window, self.recent if self.window > 1 else None)
        self.recent = self.np.r_[self.recent, scores[0]][-max(1, self.window - 1):]
        for f, sc in enumerate(decision):
            if sc >= th:
                detected |= 2
                if self.last_event is None or self.frames_done + f - self.last_event >= HOLDOFF_FRAMES:
                    self.last_event = self.frames_done + f
                    if not detected & 1:
                        at = f
                    detected |= 1
        self.frames_done += x.shape[1]
        if detected & 1:
            self.led_until = time.monotonic() + 1
        # Cycles and synaptic events exist only in RTL or hardware; never fabricated here.
        return 0, int(scores[0].max()), int(scores[0, -1]), 0, int(spikes.sum()), detected, 0, at

    def reset(self):
        self.state, self.frames_done, self.last_event = None, 0, None
        self.recent = self.np.zeros(0, self.np.int64)
        return (0,) * 7 + (-1,)

    def infer(self, payload, opcode=1):
        scores, spikes = self.forward(self.np.frombuffer(payload, self.np.uint8)[None], self.q)
        s0, s1 = map(int, scores[0])
        detected = int(s1 - s0 >= int(self.q['decision_threshold']))
        if opcode == 3:   # mirrors firmware/main.c "2 of 3" confirmation
            stream_threshold = int(self.q.get('stream_threshold', self.q['decision_threshold']))
            now, window = time.monotonic(), int(s1 - s0 >= stream_threshold)
            detected = int(window and any(hit and now - t < .75 for hit, t in self.history))
            self.history = [(bool(window), now), self.history[0]]
            detected |= window << 1
        if detected & 1:
            self.led_until = time.monotonic() + 1
        # CPU cycle count is only available in RTL or hardware, never fabricated here.
        return 0, s0, s1, 0, int(spikes[0]), detected


def backend_info(backend):
    """(abi, size, threshold, max frames); backends without info() serve ABI v2 only."""
    return backend.info() if hasattr(backend, 'info') else (2, N_INPUT, 0, 0)


def handle(client, backend):
    client.settimeout(10)
    while True:
        try:
            magic, seq, length, mode = HEADER.unpack(recv_exact(client, HEADER.size))
        except EOFError:
            return
        if magic == MAGIC3:
            if mode == MODE_INFO and length == 0:
                client.sendall(INFO.pack(MAGIC3, seq, 0, *backend_info(backend)))
                continue
            abi, _, _, max_frames = backend_info(backend)
            ok = (mode == MODE_RESET and length == 0) or (
                mode == MODE_FRAMES and 0 < length <= max_frames * FRAME_BYTES and length % FRAME_BYTES == 0)
            if abi != 3 or not ok:
                status = STATUS_WRONG_ABI if abi != 3 else STATUS_BAD
                client.sendall(RESPONSE3.pack(MAGIC3, seq, status, 0, 0, 0, 0, 0, 0, -1))
                return
            payload = recv_exact(client, length)
            result = backend.frames(payload) if mode == MODE_FRAMES else backend.reset()
            client.sendall(RESPONSE3.pack(MAGIC3, seq, *result))
            continue
        if magic == MAGIC and backend_info(backend)[0] != 2:
            client.sendall(RESPONSE.pack(MAGIC, seq, STATUS_WRONG_ABI, 0, 0, 0, 0, 0))
            return
        if magic != MAGIC or length != N_INPUT or mode not in (MODE_SINGLE, MODE_STREAM):
            client.sendall(RESPONSE.pack(MAGIC, seq, 1, 0, 0, 0, 0, 0))
            return
        payload = recv_exact(client, length)
        response = backend.infer(payload, 3 if mode == MODE_STREAM else 1)
        client.sendall(RESPONSE.pack(MAGIC, seq, *response))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--model', type=Path, default=ROOT / 'deploy/model.npz')
    p.add_argument('--firmware', type=Path, default=ROOT / 'deploy/keyword.bin')
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    a = p.parse_args()
    backend = SimulatedBoard(a.model) if a.simulate else Board(a.firmware)
    with socket.create_server((a.bind, a.port)) as server:
        print(f'Listening on {a.bind}:{a.port}; backend={"simulation" if a.simulate else "PicoRV32"}', flush=True)
        while True:
            client, address = server.accept()
            with client:
                try:
                    handle(client, backend)
                except (ConnectionError, EOFError, TimeoutError, ValueError) as e:
                    print(f'{address}: {e}', flush=True)


if __name__ == '__main__':
    main()
