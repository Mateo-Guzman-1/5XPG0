#!/usr/bin/env python3
"""mic_check.py — live microphone level meter. Speak and watch the bar.

Normal speech should reach a peak of ~0.05 or more. If it stays around
0.0002 while you talk, the mic is muted (mute key / Windows input volume)
and the keyword demo will only ever see silence.

Run:  python mic_check.py      (Ctrl-C to stop)
"""

import numpy as np
import sounddevice as sd

print(f"input device: {sd.query_devices(kind='input')['name']}  (Ctrl-C to stop)")
try:
    with sd.InputStream(channels=1, samplerate=16000, blocksize=4000) as s:
        while True:
            a = s.read(4000)[0][:, 0]
            peak = float(np.abs(a).max())
            bar = "#" * min(60, int(peak * 300))
            state = "MUTED?" if peak < 1e-3 else ("ok" if peak > 0.02 else "quiet")
            print(f"peak {peak:.4f} {state:6s} {bar}")
except KeyboardInterrupt:
    print()
