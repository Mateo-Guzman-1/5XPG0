"""Export a QAT-trained StreamSNN checkpoint to the integer model file (Phase 4).

The .npz holds the int8 weights, integer thresholds, shifts, delays and the
decision threshold (the float one converted; stream_select.py then re-chooses
it on validation with the integer oracle). robust_eval.py loads it as an
IntegerStreamDetector, i.e. through model.integer_forward_stream.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import quantize_stream
from snn_stream import StreamSNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = StreamSNN(**ck['config'])
    model.load_state_dict(ck['state_dict'])
    model.hard_delays = True
    q = quantize_stream(model, float(ck['threshold']))
    q['yes_class'] = np.array(ck['yes_class'])
    q['frontend'] = np.array(ck.get('frontend', 'logmel'))
    q['keyword'] = np.array(ck.get('keyword', 'yes'))
    q['source'] = np.array(str(a.checkpoint.as_posix()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **q)
    n_syn = q['w1'].size + q['w12'].size + q['rec'].size + q['wo'].size
    print(f'{a.out}: {n_syn} synapses, int8 weights {n_syn} bytes, threshold {int(q["stream_threshold"])}')


if __name__ == '__main__':
    main()
