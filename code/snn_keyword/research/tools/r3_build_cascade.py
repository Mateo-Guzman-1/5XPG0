"""R3: build the cascade firmware (window SNN stage 1 + GRU verifier) for the B1 and B2 designs.

    r3_build_cascade.py                 # both designs -> code/snn_keyword/build/r3_fw/<b1|b2>/
    r3_build_cascade.py --designs b1

Per design:
  - stage 1 = results/S3/models/h48_k64_seed2.npz, exported by build_firmware's export (sparse + LIF table);
  - verifier = results/R2/verifier/sheila_head_r2_s1.pt, exported by the vendored
    verifier_export.export(q, out, frames=150, warmup=5, boundary=10, thresholds=(t2, NEG), cascade_t1=t1);
  - t1, t2 read from the representative R2 configuration (keys cascade.t1, cascade.t2):
      B1 results/R2/B/v1/h48_k64_seed2.json, B2 results/R2/B/v1/budget2150/h48_k64_seed2.json;
  - compiled from firmware/ with build_firmware's compiler and FLAGS plus -DUSE_KSDOT -DUSE_LIF_TABLE -DUSE_KDOT:
    start.S cascade_main.c inference.c verifier.c runtime.c.
Writes cascade.{elf,bin,map} and build.json (hashes, section sizes, the F1 memory check) in the design folder.
F1: .model <= 144 KiB (stage 1 + verifier) and .bss ends below 0x10000 (the mailbox).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_firmware import CROSS, FIRMWARE, FLAGS, KWS, MODEL_LIMIT, export, run, section_sizes, shown  # noqa: E402

os.environ['KWS_KEYWORD'] = 'sheila'
sys.path.append(str(TOOLS / 'mateo_eval'))   # appended: code/snn_keyword's features/model stay first
import verifier_export  # noqa: E402
from verifier_model import NEG  # noqa: E402

RES = TOOLS.parent / 'results'
STAGE1 = RES / 'S3/models/h48_k64_seed2.npz'
VERIFIER = RES / 'R2/verifier/sheila_head_r2_s1.pt'
DESIGNS = {'b1': RES / 'R2/B/v1/h48_k64_seed2.json', 'b2': RES / 'R2/B/v1/budget2150/h48_k64_seed2.json'}
SOURCES = ['../../pynqz2_riscv_flow/firmware/start.S', 'cascade_main.c', 'inference.c', 'verifier.c', 'runtime.c']
DEFINES = ['-DUSE_KSDOT', '-DUSE_LIF_TABLE', '-DUSE_KDOT']
FRAMES, WARMUP, BOUNDARY = 150, 5, 10
BSS_LIMIT = 0x10000
OUT_ROOT = KWS / 'build/r3_fw'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thresholds(design):
    """(t1, t2) of the representative configuration; also checks that it used our two models."""
    j = json.loads(DESIGNS[design].read_text(encoding='utf-8'))
    if Path(j['stage1']).name != STAGE1.name or Path(j['verifier'].replace('\\', '/')).name != VERIFIER.name:
        raise SystemExit(f'{DESIGNS[design]}: made with {j["stage1"]} / {j["verifier"]}, not {STAGE1.name} / {VERIFIER.name}')
    return int(j['cascade']['t1']), int(j['cascade']['t2'])


def symbols(elf):
    out = {}
    for line in run(CROSS + 'nm', elf).splitlines():
        parts = line.split()
        if len(parts) == 3:
            out[parts[2]] = int(parts[0], 16)
    return out


def build(design, q, out_root=OUT_ROOT):
    t1, t2 = thresholds(design)
    out = Path(out_root).resolve() / design
    export(STAGE1, out, sparse=True, lif_table=True)
    vinfo = verifier_export.export(q, out, frames=FRAMES, warmup=WARMUP, boundary=BOUNDARY,
                                   thresholds=(t2, NEG), cascade_t1=t1)
    rel = os.path.relpath(out, FIRMWARE).replace('\\', '/')
    elf = f'{rel}/cascade.elf'
    try:
        run(CROSS + 'gcc', *FLAGS, f'-I{rel}', '-I.', *DEFINES, '-T', 'linker.ld',
            f'-Wl,--build-id=none,--no-warn-rwx-segments,-Map={rel}/cascade.map', *SOURCES, '-o', elf, '-lgcc')
    except Exception as e:   # CalledProcessError carries the compiler output
        raise SystemExit(f'{design}: compilation failed\n{getattr(e, "stderr", e)}')
    run(CROSS + 'objcopy', '-O', 'binary', elf, f'{rel}/cascade.bin')
    sizes, sym = section_sizes(elf), symbols(elf)
    image = (out / 'cascade.bin').read_bytes()
    f1 = dict(model_bytes=sizes.get('.model', 0), model_limit_bytes=MODEL_LIMIT,
              bss_start=hex(sym['__bss_start']), bss_end=hex(sym['__bss_end']), bss_limit=hex(BSS_LIMIT),
              bss_bytes=sym['__bss_end'] - sym['__bss_start'])
    f1['pass'] = f1['model_bytes'] <= MODEL_LIMIT and sym['__bss_end'] < BSS_LIMIT
    report = dict(design=design.upper(), t1=t1, t2=t2, thresholds_from=shown(DESIGNS[design]),
                  stage1=dict(model=shown(STAGE1), sha256=sha256(STAGE1)),
                  verifier=dict(checkpoint=shown(VERIFIER), sha256=sha256(VERIFIER), frames=FRAMES, warmup=WARMUP,
                                boundary=BOUNDARY, export_model_bytes=vinfo['model_bytes'],
                                range_checks=vinfo['range_checks']),
                  compiler=run(CROSS + 'gcc', '--version').splitlines()[0], flags=FLAGS + DEFINES, sources=SOURCES,
                  out=shown(out), bin_bytes=len(image), sha256=hashlib.sha256(image).hexdigest(),
                  sections=sizes, text_bytes=sizes.get('.text', 0),
                  symbols={k: hex(sym[k]) for k in ('main', 'verifier_run', 'keyword_infer', 'vhist', 'vlogits')
                           if k in sym},
                  f1_memory=f1)
    (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--designs', nargs='+', choices=sorted(DESIGNS), default=sorted(DESIGNS))
    p.add_argument('--out-root', type=Path, default=OUT_ROOT)
    a = p.parse_args()
    q = verifier_export.load_quantized(VERIFIER)
    reports = [build(d, q, a.out_root) for d in a.designs]
    print(json.dumps([{k: r[k] for k in ('design', 't1', 't2', 'sha256', 'bin_bytes', 'text_bytes', 'f1_memory')}
                      for r in reports], indent=1))
    if not all(r['f1_memory']['pass'] for r in reports):
        raise SystemExit('F1 (memory) failed')


if __name__ == '__main__':
    main()
