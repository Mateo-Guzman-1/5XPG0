"""The size benchmark (E2): train, tune, evaluate and measure the dense network at several sizes.

Per hidden size H (--hidden) and seed (--seeds), with the deployed model's recipe (release
recipe plus Sheila hard negatives at fraction 0.3, JOURNAL.md entries 10-11):
  train  train_keyword_snn.py -> runs_e2/h<H>/current_seed<s>/ (ignored by git)
  tune   tune_stream.py (validation windows only) writes the stream threshold; the model and
         its histories are copied to research/results/E2/models/ and training/
  eval   eval_model.py (all seeds of a size) -> research/results/E2/quality/h<H>.json
  bench  bench_board.py (RV32IM and kdot on the board) -> research/results/E2/board/h<H>_seed<s>.json
Steps whose output exists are skipped (--force redoes them), so an interrupted sweep
resumes. The first run stores the fingerprint of the fixed inputs in benchmark_inputs.json
and later runs stop if the inputs changed. pc_timing.json times the PC frontend once.
The board server restarts for every model: do not run the live demo meanwhile, and run
run_board.ps1 afterwards.

    size_sweep.py --hidden 8 16 32 48 64 80 95
    size_sweep.py --hidden 32 --steps eval bench --force
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np

TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from benchmark_inputs import describe
from features import SAMPLE_RATE, features, read_wav
from model import keyword_of
from tune_stream import HOP, live_windows

# Deployed model's recipe; everything else is train_keyword_snn.py's default (release recipe).
RECIPE = ['--encodings', 'current', '--aug-fraction', '0.3']
STEPS = ['train', 'tune', 'eval', 'bench']


def run(*args, log=None):
    """Run one of the project's scripts with this Python, from the repository root."""
    print('>', ' '.join(Path(str(x)).name if i == 0 else str(x) for i, x in enumerate(args)), flush=True)
    r = subprocess.run([sys.executable, *map(str, args)], cwd=KWS.parents[1], text=True,
                       stdout=subprocess.PIPE if log else None)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(r.stdout)
    if r.returncode:
        raise SystemExit(f'{Path(str(args[0])).name} failed (exit status {r.returncode})')


def pc_timing(out):
    """Time the PC frontend (features.py) on 300 one-second windows of real audio, 250 ms apart."""
    audio = read_wav(sorted((KWS / 'data/speech_commands_v0.02/_background_noise_').glob('*.wav'))[0])
    windows = [audio[i:i + SAMPLE_RATE] for i in range(0, 300 * HOP, HOP)]
    features(windows[0])   # builds the cached mel filter bank
    ms = []
    for w in windows:
        start = time.perf_counter()
        features(w)
        ms.append(1000 * (time.perf_counter() - start))
    out.write_text(json.dumps({'what': 'features.py on one 1 s window: the PC work for every live window',
                               'windows': len(ms), 'median_ms': round(float(np.median(ms)), 3),
                               'p95_ms': round(float(np.percentile(ms, 95)), 3), 'max_ms': round(max(ms), 3),
                               'cpu': platform.processor(), 'python': platform.python_version(),
                               'numpy': np.__version__}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hidden', nargs='+', type=int, required=True, help='hidden-layer sizes')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS)
    p.add_argument('--out', type=Path, default=TOOLS.parent / 'results/E2')
    p.add_argument('--runs', type=Path, default=KWS / 'runs_e2', help='training runs (ignored by git)')
    p.add_argument('--board', default='pynq', help='ssh host for bench_board.py')
    p.add_argument('--host', default='10.43.0.1', help='board address for bench_board.py')
    p.add_argument('--force', action='store_true', help='redo steps whose output exists')
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    data = KWS / 'data'
    keyword = keyword_of(np.load(data / 'features.npz'))
    # Make sure both live-window caches exist before they are fingerprinted.
    live_windows(data, 3000)
    live_windows(data, 3000, seed=1, split=2)
    bench = describe(keyword)
    stored = out / 'benchmark_inputs.json'
    if stored.exists() and json.loads(stored.read_text())['fingerprint'] != bench['fingerprint']:
        raise SystemExit(f'The benchmark inputs differ from {stored}; results would not be comparable')
    stored.write_text(json.dumps(bench, indent=2))
    if not (out / 'pc_timing.json').exists():
        pc_timing(out / 'pc_timing.json')
    for h in a.hidden:
        runs = a.runs / f'h{h}'
        models = {s: out / 'models' / f'h{h}_seed{s}.npz' for s in a.seeds}
        if 'train' in a.steps:
            todo = [s for s in a.seeds if a.force or not (runs / f'current_seed{s}/model.npz').exists()]
            if todo:
                run(KWS / 'train_keyword_snn.py', *RECIPE, '--hidden', h, '--seeds', *todo, '--out', runs,
                    log=runs / 'train.log')
        if 'tune' in a.steps:
            for s, model in models.items():
                if model.exists() and not a.force:
                    continue
                src = runs / f'current_seed{s}'
                run(KWS / 'tune_stream.py', src / 'model.npz', log=out / 'training' / f'h{h}_seed{s}_stream.json')
                shutil.copyfile(src / 'training.json', out / 'training' / f'h{h}_seed{s}_training.json')
                model.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / 'model.npz', model)
        if 'eval' in a.steps:
            quality = out / 'quality' / f'h{h}.json'
            if a.force or not quality.exists():
                run(TOOLS / 'eval_model.py', *[f'h{h}_seed{s}={m}' for s, m in models.items()], '--out', quality)
        if 'bench' in a.steps:
            for s, model in models.items():
                board = out / 'board' / f'h{h}_seed{s}.json'
                if a.force or not board.exists():
                    run(TOOLS / 'bench_board.py', model, '--board', a.board, '--host', a.host, '--out', board)
    print('Done. Restore the demo with run_board.ps1.', flush=True)


if __name__ == '__main__':
    main()
