#!/usr/bin/env python3
"""Download Speech Commands v0.02 and extract a reproducible KWS subset.

Every target-keyword clip is retained.  Negative clips are sampled separately
inside the official train, validation, and test partitions, approximately
uniformly across the remaining 34 command labels.  Speaker grouping and the
official manifests are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath


URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# SHA-256 published by torchaudio for the v0.02 archive.  Older torchaudio
# releases documented the shorter MD5 value 6b74f3901214cb2c2934e98196829835.
ARCHIVE_SHA256 = "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"


def normalized(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def stable_key(path: str) -> str:
    return hashlib.sha256(("group2-sheila-v2:" + path).encode()).hexdigest()


def download(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 2048 == 0:
            done = min(100.0, blocks * block_size * 100.0 / total)
            print(f"download {done:5.1f}%", flush=True)

    urllib.request.urlretrieve(URL, partial, reporthook=progress)
    partial.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def member_text(archive: tarfile.TarFile, name: str) -> str:
    member = next(
        member for member in archive.getmembers() if normalized(member.name) == name
    )
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"could not read {name}")
    return source.read().decode("utf-8")


def balanced_sample(paths: list[str], count: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        grouped[PurePosixPath(path).parts[0]].append(path)
    for values in grouped.values():
        values.sort(key=stable_key)
    selected: list[str] = []
    labels = sorted(grouped)
    cursor = 0
    while len(selected) < count:
        made_progress = False
        for label in labels:
            if cursor < len(grouped[label]) and len(selected) < count:
                selected.append(grouped[label][cursor])
                made_progress = True
        if not made_progress:
            break
        cursor += 1
    return selected


def safe_destination(root: Path, member_name: str) -> Path:
    relative = Path(*PurePosixPath(normalized(member_name)).parts)
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError(f"unsafe archive member: {member_name}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default="sheila")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-negatives", type=int, default=8000)
    parser.add_argument("--validation-negatives", type=int, default=2500)
    parser.add_argument("--test-negatives", type=int, default=3000)
    args = parser.parse_args()

    download(args.archive)
    digest = sha256(args.archive)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA256 mismatch: {digest}")

    with tarfile.open(args.archive, "r:gz") as archive:
        validation = {
            normalized(line.strip())
            for line in member_text(archive, "validation_list.txt").splitlines()
            if line.strip()
        }
        testing = {
            normalized(line.strip())
            for line in member_text(archive, "testing_list.txt").splitlines()
            if line.strip()
        }
        members = {
            normalized(member.name): member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".wav")
            and not normalized(member.name).startswith("_background_noise_/")
        }
        all_paths = sorted(members)
        positives = [
            path for path in all_paths if PurePosixPath(path).parts[0] == args.keyword
        ]
        if not positives:
            raise RuntimeError(f"keyword {args.keyword!r} is absent from the archive")
        negatives = [path for path in all_paths if path not in set(positives)]

        split_of = lambda path: "validation" if path in validation else "test" if path in testing else "train"
        negatives_by_split = {
            split: [path for path in negatives if split_of(path) == split]
            for split in ("train", "validation", "test")
        }
        selected_negatives = {
            "train": balanced_sample(negatives_by_split["train"], args.train_negatives),
            "validation": balanced_sample(
                negatives_by_split["validation"], args.validation_negatives
            ),
            "test": balanced_sample(negatives_by_split["test"], args.test_negatives),
        }
        selected = set(positives)
        for values in selected_negatives.values():
            selected.update(values)

        args.output.mkdir(parents=True, exist_ok=True)
        # Read compressed members in their physical archive order.  Seeking
        # to alphabetically sorted names in a .tar.gz repeatedly replays the
        # gzip stream and makes selective extraction prohibitively slow.
        ordered_members = [
            member
            for member in archive.getmembers()
            if normalized(member.name) in selected
        ]
        for index, member in enumerate(ordered_members, 1):
            path = normalized(member.name)
            destination = safe_destination(args.output, path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not extract {path}")
            destination.write_bytes(source.read())
            if index % 1000 == 0 or index == len(selected):
                print(f"extract {index}/{len(selected)}", flush=True)

    retained_validation = sorted(path for path in selected if path in validation)
    retained_testing = sorted(path for path in selected if path in testing)
    (args.output / "validation_list.txt").write_text(
        "\n".join(retained_validation) + "\n", encoding="utf-8"
    )
    (args.output / "testing_list.txt").write_text(
        "\n".join(retained_testing) + "\n", encoding="utf-8"
    )
    summary = {
        "source_url": URL,
        "source_sha256": digest,
        "keyword": args.keyword,
        "positive_clips": len(positives),
        "negative_limits": {
            "train": args.train_negatives,
            "validation": args.validation_negatives,
            "test": args.test_negatives,
        },
        "selected_clips": len(selected),
        "split_counts": {
            split: {
                "positive": sum(
                    1 for path in positives if split_of(path) == split
                ),
                "negative": len(selected_negatives[split]),
            }
            for split in ("train", "validation", "test")
        },
        "selection": "all keyword clips plus SHA256-ordered round-robin negatives per label and official split",
    }
    (args.output / "subset_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
