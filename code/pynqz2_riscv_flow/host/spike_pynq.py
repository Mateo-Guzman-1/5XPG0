#!/usr/bin/env python3
"""spike_pynq.py — host-side tool for the minimal RISC-V / SNN design (PYNQ-Z2).

Runs ON THE PYNQ-Z2 (needs /dev/mem and the FPGA manager). Use the PYNQ venv
python: /usr/local/share/pynq-venv/bin/python3

Memory layout (must match firmware/board.h):
    0x4000_0000  BRAM (program image, console, mailbox, spike log) via AXI
    0x4004_0000  syscon (CTRL, STATUS, TIME, SCRATCH, CLK_HZ, MAGIC, LED)

Subcommands:
    load-bit <bit>       download the bitstream (pynq Overlay)
    load-elf <bin>       hold CPU in reset, write program image, leave stopped
    start / stop         release / hold CPU reset
    status               syscon dump
    console              print new console output from the ring
    console -f           follow console (Ctrl-C to stop)
    spikes               print new output-spike records
    spikes -c            ... as csv
    cmd <name> [args]    mailbox command (echo, status)
    infer [file]         classify one 256-byte uint8 frame (default: silence)

Memory-map constants are duplicated from firmware/board.h — keep in sync.
"""

import argparse
import mmap
import os
import struct
import sys
import time

# ---------------------------------------------------------------- regs
BRAM_BASE   = 0x40000000
SYSCON_BASE = 0x40040000
SYSCON_SIZE = 0x1000
BRAM_SIZE   = 0x40000           # 256 KB

CTRL, STATUS, TIME, SCRATCH, CLK_HZ, MAGIC, LED = range(0x00, 0x1C, 4)

# BRAM regions (mirror of board.h)
CONSOLE_BASE = 0x10000
MAILBOX_BASE = 0x10400
SPIKE_LOG_BASE = 0x11000
SPIKE_LOG_DATA = SPIKE_LOG_BASE + 16
SPIKE_LOG_NWORDS = 16384
SPIKE_LOG_VERSION = 0x5A110001
FRAME_BASE = 0x22000            # seq_in, seq_done, result, cycles, data[256]
FRAME_DATA = FRAME_BASE + 16
FRAME_BYTES = 256

# mailbox opcodes (mirror of board.h)
MB = {"nop": 0, "echo": 1, "status": 2}
MB_ARGC = {"nop": 0, "echo": 3, "status": 0}

_devmem = None


def devmem():
    global _devmem
    if _devmem is None:
        _devmem = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    return _devmem


def mmap_region(base, size):
    return mmap.mmap(devmem(), size, mmap.MAP_SHARED,
                     mmap.PROT_READ | mmap.PROT_WRITE, offset=base)


class Regs:
    def __init__(self):
        self.m = mmap_region(SYSCON_BASE, SYSCON_SIZE)

    def _rd(self, off):
        return struct.unpack_from("<I", self.m, off)[0]

    def _wr(self, off, val):
        struct.pack_into("<I", self.m, off, val & 0xFFFFFFFF)

    def ctrl_reset(self, hold):
        self._wr(CTRL, 1 if hold else 0)

    def dump(self):
        return {
            "CTRL": f"{self._rd(CTRL):#x}",
            "STATUS": f"{self._rd(STATUS):#x}",
            "TIME": self._rd(TIME),
            "SCRATCH": f"{self._rd(SCRATCH):#x}",
            "CLK_HZ": self._rd(CLK_HZ),
            "MAGIC": f"{self._rd(MAGIC):#x}",
            "LED": f"{self._rd(LED):#x}",
        }


class Bram:
    def __init__(self):
        self.m = mmap_region(BRAM_BASE, BRAM_SIZE)

    def rd32(self, off):
        return struct.unpack_from("<I", self.m, off)[0]

    def wr32(self, off, val):
        struct.pack_into("<I", self.m, off, val & 0xFFFFFFFF)

    def write_image(self, image):
        if len(image) > BRAM_SIZE:
            raise ValueError("image too big")
        self.m[0:len(image)] = image
        self.m[len(image):CONSOLE_BASE] = b"\x00" * (CONSOLE_BASE - len(image))
        # console + mailbox + spike-log header + frame buffer get a clean start
        self.m[CONSOLE_BASE:SPIKE_LOG_DATA + 4 * 64] = (
            b"\x00" * (SPIKE_LOG_DATA + 4 * 64 - CONSOLE_BASE))
        self.m[FRAME_BASE:FRAME_DATA + FRAME_BYTES] = (
            b"\x00" * (FRAME_DATA + FRAME_BYTES - FRAME_BASE))

    def write_frame(self, data):
        """Hand one uint8 spectrogram frame to the CPU; returns its seq."""
        if len(data) != FRAME_BYTES:
            raise ValueError(f"frame must be {FRAME_BYTES} bytes")
        self.m[FRAME_DATA:FRAME_DATA + FRAME_BYTES] = bytes(data)
        seq = (self.rd32(FRAME_BASE) + 1) & 0xFFFFFFFF
        self.wr32(FRAME_BASE, seq)
        return seq

    def frame_result(self, seq, timeout=2.0):
        """Wait for the CPU to finish frame `seq` -> dict."""
        t0 = time.time()
        # read seq_done twice: a read that collides with the CPU's write of
        # the same word returns garbage (dual-port BRAM, see firmware)
        while not (self.rd32(FRAME_BASE + 4) == seq ==
                   self.rd32(FRAME_BASE + 4)):
            if time.time() - t0 > timeout:
                raise TimeoutError("frame not processed - CPU not running?")
            time.sleep(0.001)
        r = self.rd32(FRAME_BASE + 8)
        return {"other": r & 0xFF, "keyword": (r >> 8) & 0xFF,
                "detected": (r >> 16) & 1, "led": (r >> 17) & 1,
                "cycles": self.rd32(FRAME_BASE + 12)}

    def console_read(self):
        head = self.rd32(CONSOLE_BASE)
        tail = self.rd32(CONSOLE_BASE + 4)
        n = head - tail
        if n <= 0:
            return b""
        if n > 512:
            n = 512
        out = b""
        for i in range(n):
            off = (tail + i) & 0x1FF
            out += self.m[CONSOLE_BASE + 8 + off:CONSOLE_BASE + 8 + off + 1]
        self.wr32(CONSOLE_BASE + 4, tail + n)
        return out

    def mb_cmd(self, opcode, args=()):
        a = (list(args) + [0, 0, 0])[:3]
        seq = self.rd32(MAILBOX_BASE)
        self.wr32(MAILBOX_BASE + 8, opcode)
        self.wr32(MAILBOX_BASE + 12, a[0])
        self.wr32(MAILBOX_BASE + 16, a[1])
        self.wr32(MAILBOX_BASE + 20, a[2])
        self.wr32(MAILBOX_BASE, seq + 1)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if self.rd32(MAILBOX_BASE + 4) == seq + 1:
                return [self.rd32(MAILBOX_BASE + 24 + 4 * i) for i in range(4)]
            time.sleep(0.001)
        raise TimeoutError("mailbox command timed out - CPU not running?")

    def spikes_read(self):
        """returns (new_spikes, dropped); each spike is a dict."""
        head = self.rd32(SPIKE_LOG_BASE)
        tail = self.rd32(SPIKE_LOG_BASE + 4)
        dropped = self.rd32(SPIKE_LOG_BASE + 8)
        version = self.rd32(SPIKE_LOG_BASE + 12)
        if version != SPIKE_LOG_VERSION:
            raise RuntimeError(f"spike log not initialised "
                               f"(version {version:#x}, CPU not started?)")
        out = []
        while tail != head:
            w = self.rd32(SPIKE_LOG_DATA + 4 * (tail & (SPIKE_LOG_NWORDS - 1)))
            out.append({"out_id": w & 0xFF, "ts": (w >> 8) & 0xFFFFFF})
            tail += 1
        self.wr32(SPIKE_LOG_BASE + 4, tail)
        return out, dropped


# ---------------------------------------------------------------- cmds

def load_bit(path):
    from pynq import Bitstream
    Bitstream(path).download()
    print(f"bitstream loaded: {path}")


def cmd_loadelf(bram, path, regs):
    with open(path, "rb") as f:
        image = f.read()
    regs.ctrl_reset(True)
    time.sleep(0.01)
    bram.write_image(image)
    print(f"program written: {len(image)} bytes (CPU held in reset)")


def follow_console(bram):
    print("--- console -f (Ctrl-C to stop) ---")
    try:
        while True:
            data = bram.console_read()
            if data:
                sys.stdout.write(data.decode("ascii", "replace"))
                sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sp = sub.add_parser("load-bit"); sp.add_argument("bit")
    sp = sub.add_parser("load-elf"); sp.add_argument("bin")
    sub.add_parser("start")
    sub.add_parser("stop")
    sp = sub.add_parser("console"); sp.add_argument("-f", "--follow", action="store_true")
    sp = sub.add_parser("spikes"); sp.add_argument("-c", "--csv", action="store_true")
    sp = sub.add_parser("infer"); sp.add_argument("file", nargs="?")
    sp = sub.add_parser("cmd")
    sp.add_argument("name", choices=sorted(MB))
    sp.add_argument("args", nargs="*", type=lambda x: int(x, 0))

    a = p.parse_args()
    regs = Regs()
    bram = Bram()

    if a.cmd == "status":
        for k, v in regs.dump().items():
            print(f"{k:10s} {v}")
    elif a.cmd == "load-bit":
        load_bit(a.bit)
    elif a.cmd == "load-elf":
        cmd_loadelf(bram, a.bin, regs)
    elif a.cmd == "start":
        regs.ctrl_reset(False)
        print("CPU started")
    elif a.cmd == "stop":
        regs.ctrl_reset(True)
        print("CPU held in reset")
    elif a.cmd == "console":
        if a.follow:
            follow_console(bram)
        else:
            sys.stdout.write(bram.console_read().decode("ascii", "replace"))
    elif a.cmd == "spikes":
        evs, dropped = bram.spikes_read()
        if a.csv:
            print("out_id,ts")
            for e in evs:
                print(f"{e['out_id']},{e['ts']}")
        else:
            for e in evs:
                print(f"out={e['out_id']} ts={e['ts']}")
        if dropped:
            print(f"# dropped: {dropped}", file=sys.stderr)
    elif a.cmd == "infer":
        data = open(a.file, "rb").read() if a.file else bytes(FRAME_BYTES)
        r = bram.frame_result(bram.write_frame(data))
        print(f"result other={r['other']} keyword={r['keyword']} "
              f"detected={r['detected']} cycles={r['cycles']}")
    elif a.cmd == "cmd":
        name = a.name
        if len(a.args) != MB_ARGC[name]:
            p.error(f"{name} takes {MB_ARGC[name]} argument(s)")
        resp = bram.mb_cmd(MB[name], a.args)
        print(" ".join(f"{r:#x}" for r in resp))


if __name__ == "__main__":
    main()
