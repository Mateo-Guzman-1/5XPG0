"""Additional held-out background-noise probes; never used to tune the model."""
import json
from pathlib import Path
import numpy as np
from features import read_wav,features
from model import integer_forward

ROOT=Path(__file__).resolve().parent


def main():
    q=dict(np.load(ROOT/'deploy/model.npz'))
    rows=[]
    for path in sorted((ROOT/'data/speech_commands_v0.02/_background_noise_').glob('*.wav')):
        audio=read_wav(path)
        x=np.stack([features(audio[i:i+16000]) for i in range(0,len(audio)-15999,16000)])
        scores,_=integer_forward(x,q)
        detections=scores[:,1]-scores[:,0]>=int(q['decision_threshold'])
        rows.append(dict(recording=path.name,windows=len(x),false_detections=int(detections.sum())))
    result={'purpose':'out-of-training background probes, nonoverlapping 1 s clips; not continuous-speech FAR',
            'recordings':rows,'windows':sum(r['windows'] for r in rows),
            'false_detections':sum(r['false_detections'] for r in rows)}
    (ROOT/'results/background_noise.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
