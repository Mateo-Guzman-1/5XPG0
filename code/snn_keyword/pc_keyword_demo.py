"""Microphone or WAV -> shared mel frontend -> Ethernet -> PicoRV32 reply."""
import argparse
import json
import queue
import socket
from pathlib import Path
import numpy as np
from features import SAMPLE_RATE, features, read_wav
from protocol import MODE_SINGLE, MODE_STREAM, request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', nargs='?', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--wav', type=Path, help='Replay one mono PCM16 16 kHz WAV')
    p.add_argument('--device', help='sounddevice input name or ID')
    p.add_argument('--single', action='store_true',
                   help='Live: decide on each window alone instead of requiring 2 of 3 consecutive windows')
    a = p.parse_args()
    with socket.create_connection((a.host, a.port), timeout=10) as sock:
        if a.wav:
            print(json.dumps(request(sock, 1, features(read_wav(a.wav)))))
            return
        import sounddevice as sd
        chunks = queue.Queue(maxsize=8)
        def callback(indata, frames, timing, status):
            if status:
                # Signal a discontinuity; the main thread clears its rolling buffer.
                try: chunks.put_nowait(None)
                except queue.Full: pass
            try:
                chunks.put_nowait(indata[:, 0].copy())
            except queue.Full:
                # Bounded memory, discard backlog and reset the audio window.
                while True:
                    try: chunks.get_nowait()
                    except queue.Empty: break
                chunks.put_nowait(None)
        buffer = np.empty(0, dtype=np.float32)
        seq = 0
        mode = MODE_SINGLE if a.single else MODE_STREAM
        rule = 'each window' if a.single else '2 of 3 consecutive windows'
        print(f'Listening for "yes"; 1 s windows / 250 ms hop; detection needs {rule}. Ctrl-C stops.')
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=4000,
                            dtype='float32', device=a.device, callback=callback):
            while True:
                chunk = chunks.get()
                if chunk is None:
                    buffer = np.empty(0, dtype=np.float32)
                    continue
                buffer = np.concatenate((buffer, chunk))[-SAMPLE_RATE:]
                if len(buffer) < SAMPLE_RATE:
                    continue
                seq = (seq + 1) & 0xffffffff
                result = request(sock, seq, features(buffer), mode)
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
