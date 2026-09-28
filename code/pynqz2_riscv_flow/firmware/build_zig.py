#!/usr/bin/env python3
"""build_zig.py — build spike.bin without a RISC-V GCC toolchain.

Uses Zig's bundled clang as the cross compiler (pip install ziglang), so it
works on Windows/macOS/Linux alike. Same flags as the Makefile.

Run:  python build_zig.py        (from this folder)
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZIG = [sys.executable, "-m", "ziglang"]
CFLAGS = ["-target", "riscv32-freestanding-none", "-mcpu=generic_rv32+m",
          "-mabi=ilp32", "-mno-relax", "-O2", "-Wall", "-Wextra", "-Werror",
          "-ffreestanding", "-fno-builtin", "-nostdlib", "-I."]
LDFLAGS = ["-T", "linker.ld", "-Wl,--gc-sections", "-Wl,--build-id=none"]
SRCS = ["start.S", "main.c", "snn.c"]


def main():
    os.chdir(HERE)
    subprocess.run(ZIG + ["cc"] + CFLAGS + LDFLAGS + SRCS + ["-o", "spike.elf"],
                   check=True)
    subprocess.run(ZIG + ["objcopy", "-O", "binary", "spike.elf", "spike.bin"],
                   check=True)
    print(f"spike.bin: {os.path.getsize('spike.bin')} bytes (max 65280)")


if __name__ == "__main__":
    main()
