#!/usr/bin/env python3
"""Check every firmware variant natively on the laptop before it goes to the board.

Builds keyword_main.c for x86 with the host gcc (firmware_host_check.c), feeds it
the 40 board vectors of the matching model and compares both output spike counts
with the expected counts from the integer model. A variant only goes to the board
if all 40 frames match exactly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FW = HERE.parent.parent / "pynqz2_riscv_flow" / "firmware"
QUANT = HERE / "results" / "quant"
OUT = QUANT / "build"

# name -> (KW_VARIANT or None, exported model or None, vectors)
VARIANTS = {
    "default": (None, None, QUANT / "export_int4/board_vectors.npz"),
    "v0_damien": (None, "damien", HERE / "results/sheila_v2_deployment/board_vectors.npz"),
    "v0_int8": (0, "int8", QUANT / "export_int8/board_vectors.npz"),
    "v1_int8": (1, "int8", QUANT / "export_int8/board_vectors.npz"),
    "v2_int8": (2, "int8", QUANT / "export_int8/board_vectors.npz"),
    "v0_int4": (0, "int4", QUANT / "export_int4/board_vectors.npz"),
    "v3_int4": (3, "int4", QUANT / "export_int4/board_vectors.npz"),
    "v4_int8": (4, "int8", QUANT / "export_int8/board_vectors.npz"),
    "v5_int4": (5, "int4", QUANT / "export_int4/board_vectors.npz"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (variant, model, vectors_path) in VARIANTS.items():
        binary = OUT / f"host_{name}"
        command = ["gcc", "-O2", "-w", f"-I{FW}", str(HERE / "firmware_host_check.c"),
                   "-o", str(binary)]
        if variant is not None:
            command.insert(1, f"-DKW_VARIANT={variant}")
        if model is not None:
            header = (HERE / "results/sheila_v2_deployment/keyword_model.h" if model == "damien"
                      else QUANT / f"export_{model}" / "keyword_model.h")
            command.insert(1, f'-DKW_MODEL_HEADER="{header}"')
        subprocess.run(command, check=True)

        vectors = np.load(vectors_path)
        output = subprocess.run([str(binary)], input=vectors["x"].astype(np.uint8).tobytes(),
                                capture_output=True, check=True).stdout.decode()
        counts = np.array([[int(v) for v in line.split()] for line in output.splitlines()])
        mismatches = int(np.any(counts != vectors["counts"], axis=1).sum())
        results[name] = {"vectors": len(counts), "mismatches": mismatches}
        print(f"{name:<11} {len(counts) - mismatches}/{len(counts)} frames match the integer model")

    (OUT / "host_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["mismatches"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
