"""PC feature receiver; use --simulate for local integer model replay."""
import argparse
import mmap
import os
from pathlib import Path
import socket
import struct
import time
from protocol import HEADER, RESPONSE, MAGIC, recv_exact
from features import N_INPUT

ROOT = Path(__file__).resolve().parent


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
        while self.read(0x10430) != 0x4b575331:
            if time.monotonic() > deadline:
                raise TimeoutError('Firmware did not initialize')
            time.sleep(.001)

    def read_reg(self, offset):
        return struct.unpack_from('<I', self.reg, offset)[0]

    def read(self, offset):
        return struct.unpack_from('<I', self.ram, offset)[0]

    def write(self, offset, value):
        struct.pack_into('<I', self.ram, offset, value)

    def infer(self, payload):
        if len(payload) != N_INPUT:
            raise ValueError('Incorrect feature payload size')
        if self.read_reg(0) & 1:
            raise RuntimeError('Core is stopped; restart the board server')
        seq = (self.read(0x10400) + 1) & 0xffffffff
        self.ram[0x10800:0x10800+N_INPUT] = payload
        self.write(0x10408, 1); self.write(0x1040c, N_INPUT)
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
        return struct.unpack_from('<IiiIII', self.ram, 0x10418)


class SimulatedBoard:
    def __init__(self, model):
        import numpy as np
        from model import integer_forward
        self.np, self.forward = np, integer_forward
        self.q = dict(np.load(model))
        self.led_until = 0.

    def infer(self, payload):
        scores, spikes = self.forward(self.np.frombuffer(payload, self.np.uint8)[None], self.q)
        s0, s1 = map(int, scores[0])
        detected = s1 - s0 >= int(self.q['decision_threshold'])
        if detected:
            self.led_until = time.monotonic() + 1
        # CPU cycle count is only available in RTL or hardware, never fabricated here.
        return 0, s0, s1, 0, int(spikes[0]), int(detected)


def handle(client, backend):
    client.settimeout(10)
    while True:
        try:
            magic, seq, length = HEADER.unpack(recv_exact(client, HEADER.size))
        except EOFError:
            return
        if magic != MAGIC or length != N_INPUT:
            client.sendall(RESPONSE.pack(MAGIC, seq, 1, 0, 0, 0, 0, 0))
            return
        payload = recv_exact(client, length)
        response = backend.infer(payload)
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
