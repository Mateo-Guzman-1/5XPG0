"""Additional held-out background-noise probes; never used to tune the model."""
import argparse
import json
from pathlib import Path
import numpy as np
from features import read_wav,features
from model import integer_forward

ROOT=Path(__file__).resolve().parent
PURPOSE='out-of-training background probes, nonoverlapping 1 s clips; not continuous-speech FAR'


def background_windows():
    """(recording name, nonoverlapping 1 s feature windows) for every background-noise recording."""
    windows=[]
    for path in sorted((ROOT/'data/speech_commands_v0.02/_background_noise_').glob('*.wav')):
        audio=read_wav(path)
        windows.append((path.name,np.stack([features(audio[i:i+16000]) for i in range(0,len(audio)-15999,16000)])))
    return windows


def background_probe(q,windows=None):
    """Detections at the decision threshold per recording; pass windows to reuse them across models."""
    rows=[]
    for name,x in windows or background_windows():
        scores,_=integer_forward(x,q)
        detections=scores[:,1]-scores[:,0]>=int(q['decision_threshold'])
        rows.append(dict(recording=name,windows=len(x),false_detections=int(detections.sum())))
    return {'purpose':PURPOSE,'recordings':rows,'windows':sum(r['windows'] for r in rows),
            'false_detections':sum(r['false_detections'] for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=ROOT/'deploy/model.npz')
    p.add_argument('--out',type=Path,default=ROOT/'results/background_noise.json')
    a=p.parse_args()
    result=background_probe(dict(np.load(a.model)))
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
