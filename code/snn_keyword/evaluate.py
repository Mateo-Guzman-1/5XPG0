"""Freeze deployment selection on validation, then evaluate the held-out test set."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from model import SpikeMLP, integer_forward, metrics
from train_keyword_snn import evaluate
from export_model import export

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT / 'runs')
    a = p.parse_args()
    torch.set_num_threads(8)
    folders = sorted(a.runs.glob('*_seed*'))
    training = {f: json.loads((f / 'training.json').read_text()) for f in folders}
    if len(training) != 6:
        raise ValueError('Expected the six preregistered runs; finish training first')
    winner = max(folders, key=lambda f: (training[f]['validation']['f1'], training[f]['encoding'] == 'current'))
    deploy, results = ROOT / 'deploy', ROOT / 'results'
    deploy.mkdir(exist_ok=True); results.mkdir(exist_ok=True)
    shutil.copyfile(winner / 'model.npz', deploy / 'model.npz')
    data = np.load(ROOT / 'data/features.npz')
    mask = data['split'] == 2
    x, y, names = data['x'][mask], data['y'][mask], data['names'][mask]
    reports = []
    for folder in folders:
        q = dict(np.load(folder / 'model.npz'))
        scores, spikes = integer_forward(x, q)
        pred = scores[:, 1] - scores[:, 0] >= int(q['decision_threshold'])
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = SpikeMLP(checkpoint['hidden'], checkpoint['steps'], checkpoint['encoding'])
        model.load_state_dict(checkpoint['state_dict'])
        logits = evaluate(model, torch.tensor(x, dtype=torch.float32) / 255)
        float_pred = (logits[:, 1] - logits[:, 0]) * 1024 >= int(q['decision_threshold'])
        report = {k:v for k,v in training[folder].items() if k != 'history'}
        report.update(test=metrics(y, pred), float_test_at_same_threshold=metrics(y, float_pred),
                      quantization_decision_disagreements=int((float_pred != pred).sum()),
                      hidden_spikes_mean=float(spikes.mean()),
                      parameter_bytes=sum(q[k].nbytes for k in ['w1','b1','w2','b2']),
                      input_bytes=x.shape[1], selected=folder == winner)
        reports.append(report)
        shutil.copyfile(folder / 'training.json', results / f'{folder.name}_training.json')
        print(folder.name, json.dumps(report['test']), flush=True)
        if folder == winner:
            # Fixed stratified test examples, boundary cases, repeated data, and stress cases.
            ids = np.r_[np.where(y == 1)[0][:12], np.where(y == 0)[0][:12],
                        np.argsort(np.abs(scores[:,1]-scores[:,0]-int(q['decision_threshold'])))[:8]]
            rng = np.random.default_rng(2026)
            vectors = np.concatenate([x[ids], x[ids[:2]], np.zeros((1,768),np.uint8),
                                      np.full((1,768),255,np.uint8), rng.integers(0,256,(4,768),dtype=np.uint8)])
            expected, events = integer_forward(vectors, q)
            np.savez_compressed(results / 'verification_vectors.npz', x=vectors, scores=expected, spikes=events)
            (results / 'vector_sources.json').write_text(json.dumps(names[ids].tolist(), indent=2))
            (ROOT / 'build').mkdir(exist_ok=True)
            vectors.tofile(ROOT / 'build/vectors.bin')
            x.tofile(ROOT / 'build/test_features.bin')
            np.c_[scores, spikes].astype('<i4').tofile(ROOT / 'build/test_expected.bin')
    summary = {'selected_run': winner.name, 'selection_rule': 'maximum integer validation F1; no test tuning',
               'model_sha256': hashlib.sha256((deploy / 'model.npz').read_bytes()).hexdigest(),
               'test_samples': len(y), 'runs': reports}
    (results / 'evaluation.json').write_text(json.dumps(summary, indent=2))
    shutil.copyfile(ROOT / 'data/manifest.json', results / 'dataset.json')
    export(deploy / 'model.npz', ROOT / 'build')
    print('Selected:', winner.name, flush=True)


if __name__ == '__main__':
    main()
