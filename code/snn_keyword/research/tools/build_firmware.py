"""Build the RISC-V firmware images for one integer model (no make needed).

Runs export_model.py, then compiles keyword.bin (pure RV32IM) and
keyword_kdot.bin (-DUSE_KDOT) with the Vitis RISC-V gcc and exactly the flags
of firmware/Makefile. Reports the .model section size and refuses models that
do not fit the 144 KiB model memory (0x18000-0x3bfff).

    build_firmware.py code/snn_keyword/deploy/model.npz --out code/snn_keyword/build/release --compare code/snn_keyword/deploy

--compare checks that the new images are byte-identical to those in a folder
(the release model must rebuild into deploy/keyword*.bin exactly).
Set RISCV_CROSS to use another toolchain prefix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

KWS = Path(__file__).resolve().parents[2]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from export_model import export

FIRMWARE = KWS / 'firmware'
CROSS = os.environ.get('RISCV_CROSS', 'C:/AMDDesignTools/2026.1/gnu/riscv/nt/bin/riscv64-unknown-elf-')
# firmware/Makefile FLAGS (without -I), plus a linker flag that only silences the RWX segment warning.
FLAGS = ['-march=rv32im', '-mabi=ilp32', '-O3', '-Wall', '-Wextra', '-Werror', '-ffreestanding', '-fno-builtin',
         '-fno-tree-loop-distribute-patterns', '-nostdlib', '-msmall-data-limit=0']
SOURCES = ['../../pynqz2_riscv_flow/firmware/start.S', 'main.c', 'inference.c', 'runtime.c']
IMAGES = {'keyword': [], 'keyword_kdot': ['-DUSE_KDOT']}
# Research images (research/PLAN_SYNAPSE.md, S2-S4), built only when asked for; never part of deploy/.
EXTRA_IMAGES = {'keyword_packed': ['-DUSE_PACKED'],         # n-bit packed input weights (RV32IM)
                'keyword_sparse': ['-DUSE_SPARSE'],         # fixed fan-in synapse list (RV32IM)
                'keyword_lut': ['-DUSE_LIF_TABLE'],         # exact LIF threshold count (RV32IM)
                'keyword_kdot_lut': ['-DUSE_KDOT', '-DUSE_LIF_TABLE'],
                # S6: sparse kdot (kxload/ksdot); needs the experimental ksdot bitstream (ABI bit1)
                'keyword_ksdot_lut': ['-DUSE_KSDOT', '-DUSE_LIF_TABLE']}
MODEL_LIMIT = 144 * 1024


def run(*args):
    # Run from firmware/ with relative paths, as make does; this also keeps the
    # space in "Semiconductor Technology" out of every compiler argument.
    return subprocess.run(args, cwd=FIRMWARE, check=True, capture_output=True, text=True).stdout


def shown(path):
    """Repository-relative path when possible, for the report."""
    path = Path(path).resolve()
    return path.relative_to(KWS.parents[1]).as_posix() if path.is_relative_to(KWS.parents[1]) else path.as_posix()


def section_sizes(elf):
    """Section name -> bytes, from `size -A`."""
    rows = (line.split() for line in run(CROSS + 'size', '-A', elf).splitlines())
    return {r[0]: int(r[1]) for r in rows if len(r) == 3 and r[0].startswith('.') and r[1].isdigit()}


def build(model, out, images=tuple(IMAGES)):
    out = Path(out).resolve()
    flags = {**IMAGES, **EXTRA_IMAGES}
    # The research arrays are only written when an image needs them, so the release headers (and
    # binaries) stay byte-identical; GCC drops arrays that an image does not reference.
    export(model, out, packed='keyword_packed' in images,
           sparse=any(name in ('keyword_sparse', 'keyword_ksdot_lut') for name in images),
           lif_table=any(name.endswith('_lut') for name in images))
    rel = os.path.relpath(out, FIRMWARE).replace('\\', '/')
    report = {'model': shown(model), 'out': shown(out),
              'compiler': run(CROSS + 'gcc', '--version').splitlines()[0], 'images': {}}
    for name in images:
        defines = flags[name]
        elf = f'{rel}/{name}.elf'
        try:
            run(CROSS + 'gcc', *FLAGS, f'-I{rel}', '-I.', *defines, '-T', 'linker.ld',
                f'-Wl,--build-id=none,--no-warn-rwx-segments,-Map={rel}/{name}.map', *SOURCES, '-o', elf, '-lgcc')
        except subprocess.CalledProcessError as e:
            raise SystemExit(f'{name}: compilation failed\n{e.stderr}')
        run(CROSS + 'objcopy', '-O', 'binary', elf, f'{rel}/{name}.bin')
        sizes = section_sizes(elf)
        if sizes.get('.model', 0) > MODEL_LIMIT:
            raise SystemExit(f'{name}: .model is {sizes[".model"]} bytes, more than the {MODEL_LIMIT}-byte model memory')
        image = (out / f'{name}.bin').read_bytes()
        report['images'][name] = {'bin_bytes': len(image), 'sha256': hashlib.sha256(image).hexdigest(),
                                  'text_bytes': sizes.get('.text', 0), 'model_bytes': sizes.get('.model', 0),
                                  'model_limit_bytes': MODEL_LIMIT}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path, help='integer model .npz')
    p.add_argument('--out', type=Path, required=True, help='output folder, e.g. code/snn_keyword/build/<name>')
    p.add_argument('--compare', type=Path, help='folder whose keyword*.bin must be byte-identical')
    p.add_argument('--images', nargs='+', choices=[*IMAGES, *EXTRA_IMAGES], default=list(IMAGES),
                   help='images to build (default: the two release images)')
    a = p.parse_args()
    report = build(a.model, a.out, a.images)
    if a.compare:
        # The release since 2026-10-17 ships the LIF-table images under the release names
        # (deploy/manifest.json "firmware_build"); compare each built image with its release name.
        release = {'keyword': 'keyword', 'keyword_kdot': 'keyword_kdot', 'keyword_lut': 'keyword',
                   'keyword_kdot_lut': 'keyword_kdot'}
        report['identical_to'] = {name: (a.compare / f'{release[name]}.bin').read_bytes() == (a.out / f'{name}.bin').read_bytes()
                                  for name in a.images if name in release}
    print(json.dumps(report, indent=2))
    if a.compare and not all(report['identical_to'].values()):
        raise SystemExit('Images differ from ' + str(a.compare))


if __name__ == '__main__':
    main()
