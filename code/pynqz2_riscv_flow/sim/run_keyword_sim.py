#!/usr/bin/env python3
"""Run keyword firmware on the full SoC RTL in xsim and check it like the board.

Simulates rtl/*.v (PicoRV32 + BRAM + peripherals) with sim/tb_keyword.v,
classifies the first N board vectors and compares both output spike counts
with board_vectors.npz. Cycle counts come from the same TIMER the board uses,
so they should match the board numbers (research/results/quant/board/).

  python run_keyword_sim.py firmware/keyword.bin \\
      ../snn_keyword/research/results/quant/export_int4/board_vectors.npz -n 4

Needs Vivado 2024.1 (found through vivado/vivado2024.sh) and numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FLOW = HERE.parent
WRAPPER = FLOW / "vivado" / "vivado2024.sh"
MEM_WORDS = 65536
FRAME_BYTES = 128
RESULT = re.compile(r"^RESULT (\d+) (\d+) (-?\d+) (-?\d+) (\d+) (\d+)$")


def tool(name: str, args: list[str], cwd: Path) -> str:
    env = dict(os.environ, VIVADO_TOOL=name)
    proc = subprocess.run(
        [str(WRAPPER), *args], cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        raise RuntimeError(f"{name} failed (exit {proc.returncode})")
    return proc.stdout


def write_words(path: Path, data: bytes) -> None:
    data = data + b"\0" * (-len(data) % 4)
    words = np.frombuffer(data, dtype="<u4")
    path.write_text("".join(f"{w:08x}\n" for w in words))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("binary", type=Path, help="firmware image (keyword.bin)")
    p.add_argument("vectors", type=Path, help="board_vectors.npz of the same model")
    p.add_argument("-n", "--frames", type=int, default=4, help="frames to simulate")
    p.add_argument("--workdir", type=Path, default=HERE / "work")
    p.add_argument("--label", default=None)
    p.add_argument("--output", type=Path, help="write the summary JSON here")
    args = p.parse_args()

    image = args.binary.read_bytes()
    if len(image) > MEM_WORDS * 4:
        raise ValueError("firmware image larger than the BRAM")
    vectors = np.load(args.vectors)
    x = vectors["x"][: args.frames]
    expected = vectors["counts"][: args.frames]
    if x.shape[1] != FRAME_BYTES:
        raise ValueError(f"testbench expects {FRAME_BYTES}-byte frames")

    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    write_words(work / "prog.hex", image)
    write_words(work / "frames.hex", x.astype(np.uint8).tobytes())

    sources = sorted(str(s) for s in (FLOW / "rtl").glob("*.v"))
    tool("xvlog", ["--nolog", *sources, str(HERE / "tb_keyword.v")], work)
    tool("xelab", ["--nolog", "-O3", "--timescale", "1ns/1ps",
                   "-debug", "off", "tb_keyword", "-s", "tb_keyword"], work)
    out = tool("xsim", ["tb_keyword", "--nolog", "-R",
                        "-testplusarg", "prog=prog.hex",
                        "-testplusarg", "frames=frames.hex",
                        "-testplusarg", f"nframes={len(x)}"], work)

    rows = [RESULT.match(line.strip()) for line in out.splitlines()]
    rows = [tuple(int(v) for v in m.groups()) for m in rows if m]
    if len(rows) != len(x) or "DONE" not in out:
        sys.stdout.write(out)
        raise RuntimeError("simulation did not classify every frame")

    actual = np.array([[r[2], r[3]] for r in rows])
    cycles = np.array([r[4] for r in rows])
    layer1 = np.array([r[5] for r in rows])
    for (index, decision, s0, s1, total, l1), want in zip(rows, expected):
        ok = "ok" if [s0, s1] == list(want) else f"MISMATCH (want {list(want)})"
        print(f"frame {index:2d}: scores [{s0},{s1}] {ok}  "
              f"cycles {total}  layer1 {l1}")

    result = {
        "label": args.label or args.binary.stem,
        "vectors": len(x),
        "score_mismatches": int(np.any(actual != expected, axis=1).sum()),
        "cycles_min": int(cycles.min()),
        "cycles_mean": float(cycles.mean()),
        "cycles_max": int(cycles.max()),
        "layer1_cycles_min": int(layer1.min()),
        "layer1_cycles_mean": float(layer1.mean()),
        "layer1_cycles_max": int(layer1.max()),
        "layer1_fraction_mean": float(layer1.mean() / cycles.mean()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["score_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
