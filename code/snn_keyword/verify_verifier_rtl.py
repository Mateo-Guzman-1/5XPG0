"""Verifier track: cycles of the phoneme verifier on the PicoRV32 SoC RTL (Verilator), bit-exact.

Builds the stage-1 headers (export_model.export_stream, the keyword's stage-1
model of the main worktree, read only: "yes" W=20, "sheila" the QAT candidate), the verifier headers (verifier_export.export), the
firmware with -DUSE_VERIFIER (kdot + neuron engine, and pure RV32IM with
--rv32im), and sim/soc_verifier.cpp against rtl/spike_soc.v. Records are
validation windows: each record is streamed through stage 1 in 250 ms
requests, then command 6 verifies the last 150 frames. The oracle is
verifier_model.integer_forward + keyword_score on the same zero-padded window.
Cycles of the verifier do not depend on the weights' values (no data-dependent
branches except LUT clamping), so a randomly initialised model measures the
same cycles as a trained one.
Writes results/rtl_verifier_<tag>.csv and results/verification_verifier_rtl_<tag>.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import keyword_config as K
from export_model import export_stream
from verifier_export import export, load_quantized, oracle, records
from verify import command

ROOT = Path(__file__).resolve().parent
RTL = ROOT.parent / 'pynqz2_riscv_flow' / 'rtl'
STAGE1_W20 = Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream') / \
    {'yes': 's2_nokd_qat/int_last_w20.npz', 'sheila': 'sheila_qat_seed2/int_model.npz'}[K.KEYWORD]


def section_sizes(mapfile):
    txt = Path(mapfile).read_text()
    out = {}
    for sec in ('.text', '.data', '.bss', '.model'):
        m = re.search(rf'^\{sec}\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)', txt, re.M)
        out[sec] = int(m.group(2), 16) if m else 0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--stage1', type=Path, default=STAGE1_W20)
    p.add_argument('--frames', type=Path, help='cache npz (verifier_cascade.py cache): live clips as records')
    p.add_argument('--records', type=int, default=12)
    p.add_argument('--rv32im', action='store_true', help='no kdot, no engine')
    p.add_argument('--tag', default='')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--boundary', type=int, default=10)
    a = p.parse_args()
    tag = ('rv32im' if a.rv32im else 'engine') + (f'_{a.tag}' if a.tag else '')
    q = load_quantized(a.checkpoint)
    info = export(q, ROOT / 'build/verifier', warmup=a.warmup, boundary=a.boundary)
    export_stream(a.stage1, ROOT / 'build/stream')
    command(['make', '-C', 'firmware', 'verifier'])
    fw = f"build/keyword_stream_verifier{'' if a.rv32im else '_engine'}"
    command(['bash', '-lc', ' '.join([
        'verilator --cc --exe --build -j 8 -Wno-fatal --top-module spike_soc', '--Mdir build/obj_verifier',
        "-CFLAGS '-O3'", '../pynqz2_riscv_flow/rtl/spike_soc.v', '../pynqz2_riscv_flow/rtl/picorv32.v',
        '../pynqz2_riscv_flow/rtl/poisson.v', '../pynqz2_riscv_flow/rtl/kdot_pcpi.v',
        '../pynqz2_riscv_flow/rtl/neuron_engine.v', '"$PWD/sim/soc_verifier.cpp"'])])
    if a.frames:
        c = np.load(a.frames)
        off = c['live_off']
        rng = np.random.default_rng(0)
        pick = rng.choice(len(off) - 1, a.records, replace=False)
        seqs = [c['live_frames'][off[i]:off[i + 1]] for i in pick]
    else:
        rng = np.random.default_rng(0)
        seqs = [rng.integers(0, 256, (int(rng.integers(100, 260)), 24)).astype(np.uint8) for _ in range(a.records)]
    wins = np.stack([np.concatenate([np.zeros((150, 24), np.uint8), s])[-150:] for s in seqs])
    lg, res = oracle(wins, q, a.warmup, a.boundary)
    flat = lg.reshape(len(wins), -1)
    hashes = (flat.astype(np.int64) * np.arange(1, flat.shape[1] + 1)).sum(1) & 0xffffffff
    exp = np.c_[res, hashes.astype(np.uint32).view(np.int32)].astype('<i4')
    (ROOT / 'build/verifier_rtl_in.bin').write_bytes(records(seqs))
    exp.tofile(ROOT / 'build/verifier_rtl_expected.bin')
    t0 = time.perf_counter()
    r = command([f'./build/obj_verifier/Vspike_soc', fw + '.bin', 'build/verifier_rtl_in.bin',
                 'build/verifier_rtl_expected.bin', f'results/rtl_verifier_{tag}.csv', 25], capture_output=True, text=True)
    print(r.stdout.strip())
    rows = np.genfromtxt(ROOT / f'results/rtl_verifier_{tag}.csv', delimiter=',', names=True, dtype=None, encoding=None)
    cyc = rows['cycles'].astype(np.int64)
    sizes = section_sizes(ROOT / (fw + '.map'))
    summary = {'checkpoint': a.checkpoint.as_posix(), 'firmware': fw + '.bin', 'records': len(seqs),
               'bit_exact': 'PASS' in r.stdout, 'cycles': {'max': int(cyc.max()), 'mean': float(cyc.mean()),
                                                          'min': int(cyc.min())},
               'ms_at_100MHz': float(cyc.max()) / 1e5, 'verifier_export': info,
               'sections_bytes': sizes, 'model_region_used': f"{sizes['.model']} of {144 * 1024}",
               'code_region_used': f"{sizes['.text'] + sizes['.data'] + sizes['.bss']} of {64 * 1024}",
               'simulation_seconds': round(time.perf_counter() - t0, 1), 'harness': r.stdout.strip()}
    (ROOT / f'results/verification_verifier_rtl_{tag}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'harness'}, indent=2))


if __name__ == '__main__':
    main()
