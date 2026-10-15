#!/usr/bin/env python3
"""Compile and physically verify the paired-study shortlist on the PYNQ-Z2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import benchmark_extended_sweep_board as board
import extended_encoding_sweep as sweep
from export_extended_candidate import export_candidate


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    flow = project.parent / "pynqz2_riscv_flow"
    firmware = flow / "firmware"
    host = flow / "host"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", default="192.168.2.99")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "encoding_reproducibility_board.json",
    )
    parser.add_argument(
        "--exports",
        type=Path,
        default=Path(__file__).parent / "results" / "encoding_reproducibility_board_exports",
    )
    parser.add_argument(
        "--bash", type=Path, default=Path(r"C:\msys64\usr\bin\bash.exe")
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    summaries = report["paired_ten_seed"]["summaries"]
    remote = "/home/xilinx/snn_keyword"
    login = f"xilinx@{args.board}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", login]
    sample_paths = sorted(args.dataset.glob("*/*.wav"))[:128]
    sample_audio = np.stack([sweep.read_pcm16(path) for path in sample_paths])

    original_header = (firmware / "keyword_model.h").read_bytes()
    original_binary = (firmware / "keyword.bin").read_bytes()
    measured: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="snn-repro-board-") as temporary:
        backup = Path(temporary)
        (backup / "keyword_model.h").write_bytes(original_header)
        (backup / "keyword.bin").write_bytes(original_binary)
        try:
            board.run([*ssh, f"mkdir -p '{remote}'"])
            board.run(
                [
                    "scp",
                    "-q",
                    str(flow / "vivado" / "spike_top.bit"),
                    str(host / "spike_pynq.py"),
                    str(host / "keyword_bridge.py"),
                    str(host / "keyword_service.sh"),
                    f"{login}:{remote}/",
                ]
            )
            board.run(
                [
                    *ssh,
                    f"chmod +x '{remote}/keyword_service.sh' '{remote}/keyword_bridge.py'; "
                    "sudo -n systemctl stop snn-keyword.service >/dev/null 2>&1 || true; "
                    f"cd '{remote}' && ./keyword_service.sh stop >/dev/null 2>&1 || true",
                ]
            )
            board.run(
                [
                    *ssh,
                    "sudo -n bash -lc \"source /etc/profile.d/xrt_setup.sh; "
                    f"cd '{remote}'; /usr/local/share/pynq-venv/bin/python3 "
                    "spike_pynq.py load-bit spike_top.bit\"",
                ]
            )

            for index, (label, summary) in enumerate(summaries.items(), 1):
                cfg = sweep.EncodingConfig(**summary["config"])
                key = cfg.key
                print(f"board {index}/{len(summaries)} {label}", flush=True)
                checkpoint = args.models / f"{key}_seed0.pt"
                cache = args.cache_dir / f"{key}.npy"
                candidate = args.exports / label
                exported = export_candidate(
                    checkpoint, args.dataset, "sheila", cache, candidate
                )
                shutil.copyfile(candidate / "keyword_model.h", firmware / "keyword_model.h")
                sizes = board.build_firmware(firmware, args.bash)
                shutil.copyfile(firmware / "keyword.bin", candidate / "keyword.bin")
                board.run(
                    [
                        "scp",
                        "-q",
                        str(candidate / "keyword.bin"),
                        str(candidate / "audio_features.py"),
                        f"{login}:{remote}/",
                    ]
                )
                board.run(
                    [
                        *ssh,
                        f"cd '{remote}' && ./keyword_service.sh stop >/dev/null 2>&1 || true; "
                        "sudo -n bash -lc \"source /etc/profile.d/xrt_setup.sh; "
                        f"cd '{remote}'; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py stop; "
                        "/usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-elf keyword.bin; "
                        "/usr/local/share/pynq-venv/bin/python3 spike_pynq.py start\"; "
                        f"cd '{remote}' && ./keyword_service.sh start",
                    ]
                )
                benchmark_path = candidate / "board.json"
                completed = board.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "benchmark_candidate_board.py"),
                        args.board,
                        str(candidate / "board_vectors.npz"),
                        "--bands",
                        str(cfg.board_rows),
                        "--time-bins",
                        str(cfg.time_bins),
                        "--label",
                        label,
                        "--output",
                        str(benchmark_path),
                    ],
                    capture_output=True,
                )
                print(completed.stdout, end="", flush=True)
                physical = json.loads(benchmark_path.read_text(encoding="utf-8"))
                measured[label] = {
                    "config": summary["config"],
                    "ten_seed": {
                        "accuracy": summary["accuracy"],
                        "f1": summary["f1"],
                    },
                    "integer_validation": exported["integer_validation"],
                    "integer_test": exported["integer_test"],
                    **board.host_feature_latency(sample_audio, cfg),
                    **sizes,
                    **physical,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {"board": args.board, "candidates": measured}, indent=2
                    ),
                    encoding="utf-8",
                )
        finally:
            (firmware / "keyword_model.h").write_bytes(original_header)
            (firmware / "keyword.bin").write_bytes(original_binary)
            board.build_firmware(firmware, args.bash)
            board.run(
                [
                    "scp",
                    "-q",
                    str(flow / "vivado" / "spike_top.bit"),
                    str(firmware / "keyword.bin"),
                    str(host / "spike_pynq.py"),
                    str(host / "keyword_bridge.py"),
                    str(host / "keyword_service.sh"),
                    str(host / "keyword_boot.sh"),
                    str(host / "snn-keyword.service"),
                    str(project / "audio_features.py"),
                    f"{login}:{remote}/",
                ]
            )
            board.run(
                [
                    *ssh,
                    f"chmod +x '{remote}/keyword_service.sh' '{remote}/keyword_bridge.py' "
                    f"'{remote}/keyword_boot.sh'; cd '{remote}' && "
                    "./keyword_service.sh stop >/dev/null 2>&1 || true; "
                    f"sudo -n install -m 0644 '{remote}/snn-keyword.service' "
                    "/etc/systemd/system/snn-keyword.service; "
                    "sudo -n systemctl daemon-reload; "
                    "sudo -n systemctl enable snn-keyword.service; "
                    "sudo -n systemctl restart snn-keyword.service",
                ]
            )

    physical_summary = {
        "board": args.board,
        "clock_hz": 100_000_000,
        "candidate_count": len(measured),
        "all_reference_vectors_matched": all(
            item["score_mismatches"] == 0 for item in measured.values()
        ),
        "candidates": measured,
    }
    report["physical_board_shortlist"] = physical_summary
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and updated {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
