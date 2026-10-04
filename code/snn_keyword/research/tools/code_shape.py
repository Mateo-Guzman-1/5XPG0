"""How the compiler shaped keyword_infer for each network size (E2 helper).

GCC knows MODEL_HIDDEN at compile time. For small networks it unrolls the neuron loop
and keeps the neuron state in registers, which changes the cycles per neuron. For every
firmware that bench_board.py built (build/bench/h<H>_seed<s>/), this counts the
instructions of keyword_infer and the copies of the membrane update (the arithmetic
shift by 3 of "7 * v / 8"): 1 copy = a loop over the neurons, H copies = fully unrolled.

    code_shape.py --out research/results/E2/compiled_code.json
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

KWS = Path(__file__).resolve().parents[2]   # code/snn_keyword
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firmware import CROSS


def shape(elf):
    listing = subprocess.run([CROSS + 'objdump', '-d', '--no-show-raw-insn', str(elf)], check=True,
                             capture_output=True, text=True).stdout
    body = listing.split('<keyword_infer>:\n', 1)[1].split('\n\n', 1)[0].splitlines()
    return {'instructions': len(body),
            'membrane_update_copies': sum(bool(re.search(r'\tsrai\t\w+,\w+,0x3$', line)) for line in body)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--builds', type=Path, default=KWS / 'build/bench')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = {}
    for folder in sorted(a.builds.glob('h*_seed0'), key=lambda f: int(f.name[1:].split('_')[0])):
        rows[folder.name.split('_')[0]] = {image: shape(folder / f'{image}.elf') for image in ('keyword', 'keyword_kdot')}
    a.out.write_text(json.dumps(rows, indent=2))
    for size, images in rows.items():
        print(size, json.dumps(images))


if __name__ == '__main__':
    main()
