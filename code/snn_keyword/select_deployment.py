"""Apply the real-time constraint using measured RTL latency and validation F1."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from model import integer_forward
from export_model import export

ROOT=Path(__file__).resolve().parent


def main():
    results=ROOT/'results'
    report=json.loads((results/'evaluation.json').read_text())
    # Preserve the rate experiment before the final deployment verification replaces it.
    if report['selected_run'].startswith('rate_'):
        shutil.copyfile(results/'rtl.csv',results/'rtl_rate.csv')
        shutil.copyfile(results/'verification.json',results/'verification_rate.json')
    observed={}
    for encoding in ['current','rate']:
        rows=list(csv.DictReader((results/f'rtl_{encoding}.csv').open()))
        if len(rows)!=40:
            raise ValueError('Complete all 40 RTL vectors for each encoding before selecting')
        observed[encoding]=max(int(r['cycles']) for r in rows)
    feasible=[r for r in report['runs'] if observed[r['encoding']] <= 25000000]
    if not feasible:
        raise RuntimeError('Neither encoding meets the 250 ms hop constraint')
    winner=max(feasible,key=lambda r:r['validation']['f1'])
    selected=f'{winner["encoding"]}_seed{winner["seed"]}'
    report['unconstrained_validation_winner']=report.get('unconstrained_validation_winner',report['selected_run'])
    report['selected_run']=selected
    report['selection_rule']='maximum integer validation F1 among encodings with measured RTL max <= 25,000,000 cycles (250 ms at 100 MHz)'
    report['latency_constraint_cycles']=25000000
    report['observed_encoding_max_cycles']=observed
    for r in report['runs']:
        r['selected']=r is winner
    shutil.copyfile(ROOT/f'runs/{selected}/model.npz',ROOT/'deploy/model.npz')
    report['model_sha256']=hashlib.sha256((ROOT/'deploy/model.npz').read_bytes()).hexdigest()
    (results/'evaluation.json').write_text(json.dumps(report,indent=2))
    q=dict(np.load(ROOT/'deploy/model.npz'))
    # Retain exactly the same stress vectors for both encoding experiments.
    vectors=np.load(results/'verification_vectors.npz')['x']
    scores,spikes=integer_forward(vectors,q)
    np.savez_compressed(results/'verification_vectors.npz',x=vectors,scores=scores,spikes=spikes)
    data=np.load(ROOT/'data/features.npz')
    x=data['x'][data['split']==2]
    scores,spikes=integer_forward(x,q)
    np.c_[scores,spikes].astype('<i4').tofile(ROOT/'build/test_expected.bin')
    export(ROOT/'deploy/model.npz',ROOT/'build')
    print('Real-time deployment:',selected,'; observed max cycles:',observed,flush=True)


if __name__=='__main__': main()
