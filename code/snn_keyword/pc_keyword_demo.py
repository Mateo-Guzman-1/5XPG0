"""Microphone or WAV -> shared mel frontend -> Ethernet -> PicoRV32 reply.

The client asks the server which model is loaded (protocol.info):
  ABI v2 (window model): 1 s windows every 250 ms, "2 of 3" confirmation.
  ABI v3 (streaming SNN): 10 ms frames sent in 250 ms hops of 25 frames;
  the network state lives on the core; detection with a 1 s hold-off.
Servers that predate the INFO request are treated as ABI v2.
"""
import argparse
import json
import queue
import socket
from pathlib import Path
import numpy as np
import keyword_config as K
from features import SAMPLE_RATE, features, frame_power, logmel_agc_frames, logmel_frames, pcen_frames, read_wav
from protocol import MODE_RESET, MODE_SINGLE, MODE_STREAM, info, request, request_frames

HOP_FRAMES = 25


class FrameStream:
    """Continuous 10 ms frames from audio chunks; frame k starts at sample 160 k of the stream."""
    def __init__(self, frontend='logmel'):
        self.pending = np.empty(0, np.float32)
        self.frontend, self.pcen_state, self.agc_state = frontend, None, None

    def push(self, chunk):
        self.pending = np.concatenate((self.pending, np.asarray(chunk, np.float32)))
        if len(self.pending) < 400:
            return np.zeros((0, 24), np.uint8)
        n = (len(self.pending) - 400) // 160 + 1
        power = frame_power(self.pending[:(n - 1) * 160 + 400])
        self.pending = self.pending[n * 160:]
        if self.frontend == 'pcen':
            frames, self.pcen_state = pcen_frames(power, self.pcen_state)
            return frames
        if self.frontend == 'logmel_agc':
            frames, self.agc_state = logmel_agc_frames(power, self.agc_state)
            return frames
        return logmel_frames(power, self.frontend)


def connect(host, port):
    return socket.create_connection((host, port), timeout=10)


def model_abi(host, port):
    try:
        with connect(host, port) as sock:
            return info(sock, 0)
    except (ValueError, EOFError, ConnectionError):
        return {'abi': 2}


def run_stream(sock, frames_iter, max_frames):
    """Send frame blocks in hops; print one JSON line per hop."""
    seq = 1
    request_frames(sock, seq, b'', MODE_RESET)
    buffered = np.zeros((0, 24), np.uint8)
    for block in frames_iter:
        buffered = np.concatenate((buffered, block))
        while len(buffered) >= HOP_FRAMES:
            hop, buffered = buffered[:min(HOP_FRAMES, max_frames)], buffered[min(HOP_FRAMES, max_frames):]
            seq = (seq + 1) & 0xffffffff
            print(json.dumps(request_frames(sock, seq, hop.tobytes())), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('host', nargs='?', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--wav', type=Path, help='Replay one mono PCM16 16 kHz WAV')
    p.add_argument('--device', help='sounddevice input name or ID')
    p.add_argument('--single', action='store_true',
                   help='ABI v2 live: decide on each window alone instead of requiring 2 of 3 consecutive windows')
    p.add_argument('--frontend', choices=['logmel', 'logmel_w', 'logmel_agc', 'pcen'], default='logmel', help='ABI v3: the model\'s front end')
    p.add_argument('--keyword', help='name shown in the messages; default: KWS_KEYWORD ("sheila") for the streaming '
                   'model (ABI v3), "yes" for ABI v2, which is the window-model release')
    a = p.parse_args()
    abi = model_abi(a.host, a.port)
    stream = abi['abi'] == 3
    keyword = a.keyword or (K.KEYWORD if stream else 'yes')
    with connect(a.host, a.port) as sock:
        if a.wav:
            audio = read_wav(a.wav)
            if stream:
                # Half a second of silence after the file lets a final keyword reach the readout.
                fs = FrameStream(a.frontend)
                run_stream(sock, [fs.push(np.r_[audio, np.zeros(SAMPLE_RATE // 2, np.float32)])], abi['max_frames'])
            else:
                print(json.dumps(request(sock, 1, features(audio))))
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
        if stream:
            print(f'Listening for "{keyword}"; streaming SNN, 10 ms frames in 250 ms hops, 1 s hold-off. Ctrl-C stops.')
        else:
            rule = 'each window' if a.single else '2 of 3 consecutive windows'
            print(f'Listening for "{keyword}"; 1 s windows / 250 ms hop; detection needs {rule}. Ctrl-C stops.')
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=4000,
                            dtype='float32', device=a.device, callback=callback):
            if stream:
                def frames_iter():
                    fs = FrameStream(a.frontend)
                    while True:
                        chunk = chunks.get()
                        # A discontinuity only restarts the framing; the network state carries on.
                        if chunk is None:
                            fs = FrameStream(a.frontend)
                            continue
                        yield fs.push(chunk)
                run_stream(sock, frames_iter(), abi['max_frames'])
                return
            buffer = np.empty(0, dtype=np.float32)
            seq = 0
            mode = MODE_SINGLE if a.single else MODE_STREAM
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
