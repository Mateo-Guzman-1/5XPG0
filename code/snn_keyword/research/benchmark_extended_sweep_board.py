#!/usr/bin/env python3
"""Compile, deploy, and measure every extended encoding candidate on PYNQ."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import extended_encoding_sweep as sweep
from export_extended_candidate import export_candidate


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def msys_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix().split(":", 1)[1]
    return f"/{drive}{remainder}"


def build_firmware(firmware: Path, bash: Path) -> dict[str, int]:
    command = (
        "export PATH=/mingw64/bin:$PATH; "
        f"make -C '{msys_path(firmware)}' -B keyword.bin"
    )
    completed = run([str(bash), "-lc", command], capture_output=True)
    print(completed.stdout, end="", flush=True)
    match = re.search(
        r"\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[0-9a-fA-F]+\s+keyword\.elf",
        completed.stdout,
    )
    if not match:
        raise RuntimeError("could not parse riscv64-unknown-elf-size output")
    text_bytes, data_bytes, bss_bytes, linked_bytes = map(int, match.groups())
    return {
        "elf_text_bytes": text_bytes,
        "elf_data_bytes": data_bytes,
        "elf_bss_bytes": bss_bytes,
        "elf_linked_bytes": linked_bytes,
        "firmware_binary_bytes": (firmware / "keyword.bin").stat().st_size,
    }


def host_feature_latency(
    audio: np.ndarray, cfg: sweep.EncodingConfig, repeats: int = 3
) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        sweep.feature_batch(audio, cfg)
        samples.append(1000.0 * (time.perf_counter() - started) / len(audio))
    return {
        "host_feature_ms_per_clip_mean": float(np.mean(samples)),
        "host_feature_ms_per_clip_min": float(np.min(samples)),
        "host_feature_ms_per_clip_max": float(np.max(samples)),
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    flow = project.parent / "pynqz2_riscv_flow"
    firmware = flow / "firmware"
    host = flow / "host"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="192.168.2.99")
    parser.add_argument("--sweep", type=Path, default=Path(__file__).parent / "results" / "extended_encoding_sweep_sheila_v2.json")
    parser.add_argument("--dataset", type=Path, default=project / "data" / "speech_commands_v2_sheila_subset")
    parser.add_argument("--cache-dir", type=Path, default=project / "data" / "extended_encoding_cache_sheila_v2")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "extended_encoding_board_sheila_v2.json")
    parser.add_argument("--exports", type=Path, default=Path(__file__).parent / "results" / "extended_encoding_board_exports")
    parser.add_argument("--bash", type=Path, default=Path(r"C:\msys64\usr\bin\bash.exe"))
    args = parser.parse_args()

    report = json.loads(args.sweep.read_text(encoding="utf-8"))
    screening = report["screening"]
    model_dir = args.sweep.parent / f"{args.sweep.stem}_models"
    remote = "/home/xilinx/snn_keyword"
    board_login = f"xilinx@{args.board}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", board_login]

    sample_paths = sorted(args.dataset.glob("*/*.wav"))[:128]
    sample_audio = np.stack([sweep.read_pcm16(path) for path in sample_paths])

    original_header = (firmware / "keyword_model.h").read_bytes()
    original_binary = (firmware / "keyword.bin").read_bytes()
    board_results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="snn-encoding-board-") as temporary:
        backup = Path(temporary)
        (backup / "keyword_model.h").write_bytes(original_header)
        (backup / "keyword.bin").write_bytes(original_binary)
        try:
            run([*ssh, f"mkdir -p '{remote}'"])
            run([
                "scp", "-q",
                str(flow / "vivado" / "spike_top.bit"),
                str(host / "spike_pynq.py"),
                str(host / "keyword_bridge.py"),
                str(host / "keyword_service.sh"),
                f"{board_login}:{remote}/",
            ])
            run([*ssh, f"chmod +x '{remote}/keyword_service.sh' '{remote}/keyword_bridge.py'; sudo -n systemctl stop snn-keyword.service >/dev/null 2>&1 || true; cd '{remote}' && ./keyword_service.sh stop >/dev/null 2>&1 || true"])
            run([*ssh, f"sudo -n bash -lc \"source /etc/profile.d/xrt_setup.sh; cd '{remote}'; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-bit spike_top.bit\""])

            for index, item in enumerate(screening, 1):
                cfg = sweep.EncodingConfig(**item["config"])
                key = cfg.key
                print(f"board {index}/{len(screening)} {key}", flush=True)
                checkpoint = model_dir / f"{key}_screen_seed0.pt"
                candidate_dir = args.exports / key
                export = export_candidate(
                    checkpoint,
                    args.dataset,
                    "sheila",
                    args.cache_dir / f"{key}.npy",
                    candidate_dir,
                )
                shutil.copyfile(candidate_dir / "keyword_model.h", firmware / "keyword_model.h")
                sizes = build_firmware(firmware, args.bash)
                shutil.copyfile(firmware / "keyword.bin", candidate_dir / "keyword.bin")
                run([
                    "scp", "-q",
                    str(candidate_dir / "keyword.bin"),
                    str(candidate_dir / "audio_features.py"),
                    f"{board_login}:{remote}/",
                ])
                run([*ssh, f"cd '{remote}' && ./keyword_service.sh stop >/dev/null 2>&1 || true; sudo -n bash -lc \"source /etc/profile.d/xrt_setup.sh; cd '{remote}'; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py stop; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-elf keyword.bin; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py start\"; cd '{remote}' && ./keyword_service.sh start"])

                benchmark_path = candidate_dir / "board.json"
                benchmark = run([
                    sys.executable,
                    str(Path(__file__).parent / "benchmark_candidate_board.py"),
                    args.board,
                    str(candidate_dir / "board_vectors.npz"),
                    "--bands", str(cfg.board_rows),
                    "--time-bins", str(cfg.time_bins),
                    "--label", key,
                    "--output", str(benchmark_path),
                ], capture_output=True)
                print(benchmark.stdout, end="", flush=True)
                measured = json.loads(benchmark_path.read_text(encoding="utf-8"))
                board_results[key] = {
                    "config": item["config"],
                    "validation": item["validation"],
                    "memory": item["memory"],
                    "integer_validation": export["integer_validation"],
                    **host_feature_latency(sample_audio, cfg),
                    **sizes,
                    **measured,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"board": args.board, "candidates": board_results}, indent=2),
                    encoding="utf-8",
                )
        finally:
            (firmware / "keyword_model.h").write_bytes(original_header)
            (firmware / "keyword.bin").write_bytes(original_binary)
            build_firmware(firmware, args.bash)
            run([
                "scp", "-q",
                str(flow / "vivado" / "spike_top.bit"),
                str(firmware / "keyword.bin"),
                str(host / "spike_pynq.py"),
                str(host / "keyword_bridge.py"),
                str(host / "keyword_service.sh"),
                str(host / "keyword_boot.sh"),
                str(host / "snn-keyword.service"),
                str(project / "audio_features.py"),
                f"{board_login}:{remote}/",
            ])
            run([*ssh, f"chmod +x '{remote}/keyword_service.sh' '{remote}/keyword_bridge.py' '{remote}/keyword_boot.sh'; cd '{remote}' && ./keyword_service.sh stop >/dev/null 2>&1 || true; sudo -n install -m 0644 '{remote}/snn-keyword.service' /etc/systemd/system/snn-keyword.service; sudo -n systemctl daemon-reload; sudo -n systemctl enable snn-keyword.service; sudo -n systemctl restart snn-keyword.service"])

    report["physical_board"] = {
        "board": args.board,
        "clock_hz": 100_000_000,
        "candidate_count": len(board_results),
        "all_reference_vectors_matched": all(
            item["score_mismatches"] == 0 for item in board_results.values()
        ),
        "candidates": board_results,
    }
    args.sweep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and updated {args.sweep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
