"""R3 board kit (research/PLAN_R3_GATING.md 2F): replay R3 request files on the PYNQ-Z2. Written in R3, run later.

Runs on the board (PYNQ Linux, as root for /dev/mem):
    sudo env XILINX_XRT=/usr BOARD=Pynq-Z2 python3 r3_board_cascade.py --bit r3_gated.bit --fw cascade.bin \
        --req v3a.req --out v3a.board.txt [--idle-seconds 60]

- Loads the bitstream (pynq.Bitstream(...).download()), checks the SoC magic, ABI bits 0-1 (and bit 2 for a gated
  bitstream) and the 100 MHz clock, then loads the firmware like board_server.py (core in reset, RAM cleared, image
  written, reset released) and waits for the cascade magic "KWC1" in MB[12].
- Replays the request file of r3_sim.py (same format; testbench-only flags are ignored): payload to 0x10800, MB[2],
  MB[3], then MB[0] last; MB[1] polled every 1 ms; MB[6..35] read after each reply.
- Writes REQ lines in the xsim log format (activity counters 0), so that on the PC
      r3_sim.py check <name>.board.txt <set>.expect.json
  compares the board with the oracle and the simulated cycles; MB[34..35] hold the POWER counters (clocks asleep,
  sleeps) on a gated bitstream.
- --idle-seconds N: afterwards the core stays idle (asleep on a gated bitstream) for N s and the asleep share is
  printed (for the USB-meter comparison of busy-polling and gated idle).
board_server.py, start_board.sh, run_board.ps1 and deploy/ are not used or changed; restore the demo with run_board.ps1.
"""
import argparse
import mmap
import os
from pathlib import Path
import time

MB, INPUT = 0x10400, 0x10800
MAGIC_SOC, MAGIC_CASCADE = 0x534B454C, 0x4B574331


class Soc:
    def __init__(self, bit, fw, gated):
        from pynq import Bitstream, Clocks
        Bitstream(str(bit)).download()
        fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        try:
            self.ram = mmap.mmap(fd, 0x40000, offset=0x40000000)
            self.reg = mmap.mmap(fd, 4096, offset=0x40040000)
        finally:
            os.close(fd)
        self.ram32, self.reg32 = memoryview(self.ram).cast('I'), memoryview(self.reg).cast('I')   # one access per word
        if self.reg32[0x14 // 4] != MAGIC_SOC:
            raise RuntimeError('spike SoC bitstream not loaded')
        self.abi = self.reg32[0x1c // 4]
        if self.abi & 3 != 3 or (gated and not self.abi & 4):
            raise RuntimeError(f'unexpected ABI {self.abi:#x}')
        if self.reg32[0x10 // 4] != 100_000_000:
            raise RuntimeError('expected a 100 MHz fabric clock')
        image = Path(fw).read_bytes()
        self.reg32[0] = 1
        Clocks.fclk0_mhz = 100.0
        self.ram[:] = bytes(0x40000)
        self.ram[:len(image)] = image
        self.reg32[0] = 0
        deadline = time.monotonic() + 5
        while self.ram32[(MB + 48) // 4] != MAGIC_CASCADE:
            if time.monotonic() > deadline:
                raise TimeoutError('firmware did not initialize')
            time.sleep(.001)

    def request(self, opcode, payload, length):
        seq = (self.ram32[MB // 4] + 1) & 0xFFFFFFFF
        self.ram[INPUT:INPUT + len(payload)] = payload
        self.ram32[(MB + 8) // 4] = opcode
        self.ram32[(MB + 12) // 4] = length
        self.ram32[MB // 4] = seq                     # publish last
        t0 = time.monotonic()
        while self.ram32[(MB + 4) // 4] != seq:
            if self.reg32[1] & 2:
                raise RuntimeError('RISC-V trap')
            if time.monotonic() - t0 > 5:
                raise TimeoutError('no reply within 5 s')
            time.sleep(.001)
        return seq, [self.ram32[MB // 4 + i] for i in range(6, 36)], time.monotonic() - t0


def requests(path):
    tok = Path(path).read_text().split()
    i = 0
    while i < len(tok):
        op, length, gap, flags, n = (int(x) for x in tok[i:i + 5])
        words = [int(x, 16) for x in tok[i + 5:i + 5 + n]]
        i += 5 + n
        yield op, length, b''.join(w.to_bytes(4, 'little') for w in words)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bit', type=Path, required=True)
    p.add_argument('--fw', type=Path, required=True)
    p.add_argument('--req', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--idle-seconds', type=float, default=0)
    a = p.parse_args()
    gated = 'gated' in a.bit.name
    soc = Soc(a.bit, a.fw, gated)
    lines = [f'CONFIG board ABI={soc.abi:#x} bit={a.bit.name} fw={a.fw.name}']
    for n, (op, length, payload) in enumerate(requests(a.req)):
        seq, mb, dt = soc.request(op, payload, length)
        lines.append(f'REQ n={n} op={op} len={length} flags=0 seq={seq} svc=0 pb_re=0 pb_we=0 pa_re=0 pa_we=0 '
                     f'clk_on=0 sleep=0 tb_asleep=0 tb_sleeps=0 awake=0 race=none host_ms={1000 * dt:.2f} mb=' +
                     ','.join(f'{w:08x}' for w in mb))
        print(lines[-1][:120], flush=True)
    if a.idle_seconds:
        s0, t0 = soc.ram32[(MB + 136) // 4], soc.reg32[2]          # MB[34] (asleep counter snapshot), TIME
        time.sleep(a.idle_seconds)
        seq, mb, _ = soc.request(2, b'', 0)
        asleep = (mb[34 - 6] - s0) & 0xFFFFFFFF
        cycles = (soc.reg32[2] - t0) & 0xFFFFFFFF
        lines.append(f'IDLE seconds={a.idle_seconds} asleep_cycles={asleep} timer_cycles={cycles} '
                     f'asleep_share={asleep / max(cycles, 1):.6f}')
        print(lines[-1])
    lines += ['DONE']
    a.out.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
