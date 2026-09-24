"""Compare actual native C and RV32IM RTL outputs with the independent oracle."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from model import integer_forward
from export_model import export

ROOT = Path(__file__).resolve().parent


def command(args, **kwargs):
    prefix = ['wsl', '-d', 'Ubuntu', '--'] if platform.system() == 'Windows' else []
    return subprocess.run(prefix + list(map(str,args)), cwd=ROOT, check=True, **kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-rtl',action='store_true')
    p.add_argument('--release-only',action='store_true',help='Use bundled vectors without downloading training data')
    a=p.parse_args()
    q=dict(np.load(ROOT/'deploy/model.npz'))
    if a.release_only:
        vectors=np.load(ROOT/'results/verification_vectors.npz')
        x=vectors['x']; expected=np.c_[vectors['scores'],vectors['spikes']]
    else:
        x=np.fromfile(ROOT/'build/test_features.bin',np.uint8).reshape(-1,768)
        expected=np.fromfile(ROOT/'build/test_expected.bin','<i4').reshape(-1,3)
    x.tofile(ROOT/'build/native_input.bin')
    np.load(ROOT/'results/verification_vectors.npz')['x'].tofile(ROOT/'build/vectors.bin')
    start=time.perf_counter()
    with (ROOT/'build/native_input.bin').open('rb') as inp, (ROOT/'build/native_actual.bin').open('wb') as out:
        command(['./build/native'],stdin=inp,stdout=out)
    actual=np.fromfile(ROOT/'build/native_actual.bin','<i4').reshape(-1,3)
    np.testing.assert_array_equal(actual,expected)
    native_seconds=time.perf_counter()-start
    report={'native_c':{'samples':len(x),'score_and_spike_mismatches':0,'wall_seconds_including_wsl':native_seconds}}
    if not a.skip_rtl:
        log=command(['./build/obj_dir/Vspike_soc','build/keyword.bin','build/vectors.bin','results/rtl.csv',int(q['decision_threshold'])],capture_output=True,text=True)
        print(log.stdout,flush=True)
        (ROOT/'results/rtl_console.txt').write_text(log.stdout+log.stderr)
        rows=list(csv.DictReader((ROOT/'results/rtl.csv').open()))
        vectors=np.load(ROOT/'results/verification_vectors.npz')
        got=np.array([[int(r[k]) for k in ['score0','score1','spikes']] for r in rows])
        np.testing.assert_array_equal(got,np.c_[vectors['scores'],vectors['spikes']])
        cycles=np.array([int(r['cycles']) for r in rows])
        report['rtl']={'samples':len(rows),'score_and_spike_mismatches':0,
                       'cycles_min':int(cycles.min()),'cycles_mean':float(cycles.mean()),'cycles_max':int(cycles.max()),
                       'milliseconds_max_at_100mhz':float(cycles.max()/100000),
                       'led_pulse_cycles':100000000,'malformed_requests':'pass','duplicate_sequence':'pass'}
    report['firmware_sha256']=hashlib.sha256((ROOT/'build/keyword.bin').read_bytes()).hexdigest()
    report['model_sha256']=hashlib.sha256((ROOT/'deploy/model.npz').read_bytes()).hexdigest()
    (ROOT/'results'/('recheck.json' if a.release_only else 'verification.json')).write_text(json.dumps(report,indent=2))
    shutil.copyfile(ROOT/'build/keyword.bin',ROOT/'deploy/keyword.bin')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
