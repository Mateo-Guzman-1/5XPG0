import socket
import sys
import threading
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import features, mel_bank, N_INPUT
from model import integer_forward, trunc_div, metrics, choose_threshold
from protocol import request, recv_exact, HEADER, RESPONSE, MAGIC, MODE_STREAM
from board_server import handle
from board_server import SimulatedBoard
from board_server import Board


def test_frontend_contract():
    assert features(np.zeros(16000)).shape == (768,)
    assert features(np.zeros(16000)).dtype == np.uint8
    assert not features(np.zeros(16000)).any()
    assert np.allclose(mel_bank().sum(1), 1)
    tone = np.sin(2*np.pi*1000*np.arange(16000)/16000).astype(np.float32)
    assert features(tone).max() > 100
    assert np.array_equal(features(tone), features(np.r_[tone, tone]))
    with pytest.raises(ValueError): features([np.nan])


def _keyword_in_fresh_interpreter(env_value=None):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != 'KWS_KEYWORD'}
    if env_value is not None:
        env['KWS_KEYWORD'] = env_value
    code = ("import keyword_config as K; print(K.KEYWORD, K.MULTI.name);"
            "import numpy as np; from robust_eval import WindowDetector;"
            "q = dict(np.load('deploy/model.npz'))\n"
            "try:\n    WindowDetector(q); print('window model accepted')\n"
            "except ValueError as e: print('window model refused:', e)")
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()


def test_the_project_keyword_is_sheila_and_yes_models_are_guarded():
    default = _keyword_in_fresh_interpreter()
    assert default[0] == 'sheila multi_sheila'
    assert default[1].startswith('window model refused: window model was trained for \'yes\'')
    assert 'KWS_KEYWORD=yes' in default[1]
    # The earlier keyword stays available, with its own data folder and its release model.
    assert _keyword_in_fresh_interpreter('yes') == ['yes multi', 'window model accepted']


def test_signed_division_and_threshold_ties():
    assert trunc_div(np.array([-9,-8,-7,0,7,8,9]),8).tolist() == [-1,-1,0,0,0,1,1]
    scores = np.array([2,2,1,0]); y=np.array([1,0,1,0])
    t=choose_threshold(y,scores)
    assert t == 1
    assert metrics(y,scores>=t)['f1'] == pytest.approx(.8)


def test_hand_computed_lif_and_reset():
    # One neuron, exactly threshold drive every step -> one spike per step.
    q=dict(w1=np.zeros((1,768),np.int16), b1=np.array([1024]),
           w2=np.array([[-3],[5]],np.int16), b2=np.array([2,-1]), steps=3, encoding=0)
    scores, spikes = integer_forward(np.zeros((2,768),np.uint8),q)
    assert scores.tolist() == [[-3,12],[-3,12]]
    assert spikes.tolist() == [3,3]
    # Repeated calls must start with a fresh membrane and phase.
    assert np.array_equal(integer_forward(np.zeros((1,768),np.uint8),q)[0],scores[:1])


class FakeBoard:
    def __init__(self): self.opcodes=[]
    def infer(self,payload,opcode=1):
        assert len(payload)==768
        self.opcodes.append(opcode)
        return (0,-3,12,12345,3,1 if opcode==1 else 3)


def test_tcp_request_reply_and_multiple_requests():
    a,b=socket.socketpair()
    thread=threading.Thread(target=handle,args=(b,FakeBoard()),daemon=True)
    thread.start()
    with a:
        for seq in [1,2,0xffffffff,0]:
            result=request(a,seq,bytes(768))
            assert result['score0']==-3 and result['cycles']==12345 and result['detected']
    thread.join(timeout=2); b.close()
    assert not thread.is_alive()


def test_tcp_rejects_unbounded_length():
    a,b=socket.socketpair()
    thread=threading.Thread(target=handle,args=(b,FakeBoard()),daemon=True)
    thread.start()
    with a:
        # Fragmented headers exercise recv_exact as well as bounded allocation.
        packet=HEADER.pack(MAGIC,7,0xffff,0)
        for byte in packet: a.sendall(bytes([byte]))
        reply=RESPONSE.unpack(recv_exact(a,RESPONSE.size))
        assert reply[1:3]==(7,1)
    thread.join(timeout=2); b.close()
    assert not thread.is_alive()


def test_stream_mode_and_legacy_header():
    import struct
    a,b=socket.socketpair(); board=FakeBoard()
    thread=threading.Thread(target=handle,args=(b,board),daemon=True)
    thread.start()
    with a:
        r=request(a,1,bytes(768),MODE_STREAM)
        assert r['detected'] and r['window']
        # A client using the original '<4sII' header is served as a single window.
        a.sendall(struct.pack('<4sII',MAGIC,2,768)+bytes(768))
        reply=RESPONSE.unpack(recv_exact(a,RESPONSE.size))
        assert reply[1:3]==(2,0) and reply[-1]==1
        # Unknown modes are rejected.
        a.sendall(HEADER.pack(MAGIC,3,768,7))
        assert RESPONSE.unpack(recv_exact(a,RESPONSE.size))[1:3]==(3,1)
    thread.join(timeout=2); b.close()
    assert not thread.is_alive() and board.opcodes==[3,1]


def test_simulated_two_of_three_confirmation(monkeypatch):
    import board_server
    backend=SimulatedBoard.__new__(SimulatedBoard)
    backend.np=np; backend.led_until=0.; backend.history=[(False,0.),(False,0.)]
    backend.q={'decision_threshold':np.array(0)}
    margins=iter([])
    backend.forward=lambda x,q:(np.array([[0,next(margins)]]),np.array([1]))
    now=[0.]
    monkeypatch.setattr(board_server.time,'monotonic',lambda:now[0])
    def run(pattern,step=.25):
        nonlocal margins
        margins=iter([1 if p else -1 for p in pattern]); out=[]
        for _ in pattern:
            out.append(backend.infer(bytes(768),3)[-1]); now[0]+=step
        return out
    # +,+,-,+ : unconfirmed, confirmed, none, confirmed (window bit is bit1).
    assert run([1,1,0,1])==[2,3,0,3]
    # Single windows (command 1) are unaffected by the stream history.
    margins=iter([1]); assert backend.infer(bytes(768),1)[-1]==1
    # Positives more than 750 ms apart never confirm each other.
    now[0]+=5; assert run([1,1],step=.8)==[2,2]


def test_exported_model_silence():
    path=ROOT/'deploy/model.npz'
    if not path.exists(): pytest.skip('Run training/export first')
    q=dict(np.load(path))
    s,_=integer_forward(np.zeros((1,N_INPUT),np.uint8),q)
    assert s[0,1]-s[0,0] < int(q['decision_threshold'])


def test_real_model_over_tcp_and_wav_frontend():
    path=ROOT/'deploy/model.npz'
    if not path.exists(): pytest.skip('Run training/export first')
    vectors=np.load(ROOT/'results/verification_vectors.npz')
    backend=SimulatedBoard(path)
    with socket.create_server(('127.0.0.1',0)) as server:
        def serve():
            conn,_=server.accept()
            with conn: handle(conn,backend)
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        with socket.create_connection(server.getsockname(),timeout=5) as client:
            for i in range(3):
                r=request(client,i,vectors['x'][i])
                assert [r['score0'],r['score1']]==vectors['scores'][i].tolist()
                assert r['spikes']==int(vectors['spikes'][i])
        thread.join(timeout=3)
        assert not thread.is_alive()
    sources=ROOT/'results/vector_sources.json'
    if sources.exists():
        import json
        from features import read_wav
        wav=ROOT/'data/speech_commands_v0.02'/json.loads(sources.read_text())[0]
        if wav.exists():
            assert np.array_equal(features(read_wav(wav)),vectors['x'][0])


def test_board_mailbox_layout_and_sequence_wrap():
    import struct
    board=Board.__new__(Board)
    board.ram=bytearray(0x40000);board.reg=bytearray(4096)
    struct.pack_into('<I',board.ram,0x10400,0xffffffff)
    struct.pack_into('<I',board.ram,0x10404,0xffffffff)
    original_write=board.write
    payload=bytes([123])*768
    def firmware_response(offset,value):
        original_write(offset,value)
        if offset==0x10400:
            assert board.ram[0x10800:0x10b00]==payload
            assert board.read(0x10408)==1 and board.read(0x1040c)==768
            struct.pack_into('<IiiIII',board.ram,0x10418,0,-17,32,9876,5,1)
            original_write(0x10404,value)
    board.write=firmware_response
    assert board.infer(payload)==(0,-17,32,9876,5,1)
    assert board.read(0x10400)==0


def test_board_timeout_stops_core(monkeypatch):
    import board_server
    board=Board.__new__(Board)
    board.ram=bytearray(0x40000);board.reg=bytearray(4096)
    clock=iter([0,6])
    monkeypatch.setattr(board_server.time,'monotonic',lambda:next(clock))
    with pytest.raises(RuntimeError,match='timed out'): board.infer(bytes(768))
    assert board.read_reg(0)==1
    with pytest.raises(RuntimeError,match='stopped'): board.infer(bytes(768))


# ---------------------------------------------------------------- streaming SNN, ABI v3

def small_stream_model():
    """A randomly initialised StreamSNN (16 + 16 neurons), exported to integers."""
    import torch
    from snn_stream import StreamSNN
    from model import quantize_stream
    torch.manual_seed(0)
    m = StreamSNN(n1=16, n2=16, classes=3, seed=0)
    m.hard_delays = True
    with torch.no_grad():
        m.fc1.weight.mul_(8)   # enough drive that neurons spike on random frames
    q = quantize_stream(m, threshold=0.)
    q['yes_class'] = np.array(2)
    return q


def test_live_framing_equals_offline_frames():
    from features import frame_features
    from pc_keyword_demo import FrameStream
    audio = np.random.default_rng(0).standard_normal(3 * 16000 + 123).astype(np.float32) * .05
    fs = FrameStream()
    live = np.concatenate([fs.push(audio[i:i + 4000]) for i in range(0, len(audio), 4000)])
    assert np.array_equal(live, frame_features(audio))


def test_v3_frames_over_tcp_equal_oracle_and_state_persists():
    from model import integer_forward_stream
    from protocol import info, request_frames, MODE_RESET
    q = small_stream_model()
    backend = SimulatedBoard.__new__(SimulatedBoard)
    backend.np, backend.q, backend.abi, backend.led_until = np, q, 3, 0.
    backend.state, backend.frames_done, backend.last_event = None, 0, None
    backend.window, backend.recent = 1, np.zeros(0, np.int64)
    frames = np.random.default_rng(1).integers(0, 256, (1, 60, 24), dtype=np.uint8)
    expected, _, _ = integer_forward_stream(frames, q)
    a, b = socket.socketpair()
    thread = threading.Thread(target=handle, args=(b, backend), daemon=True); thread.start()
    with a:
        assert info(a, 1)['abi'] == 3
        request_frames(a, 2, b'', MODE_RESET)
        first = request_frames(a, 3, frames[0, :25].tobytes())
        second = request_frames(a, 4, frames[0, 25:].tobytes())   # state carried over
        assert (first['best'], first['last']) == (expected[0, :25].max(), expected[0, 24])
        assert (second['best'], second['last']) == (expected[0, 25:].max(), expected[0, -1])
        request_frames(a, 5, b'', MODE_RESET)
        again = request_frames(a, 6, frames[0, :25].tobytes())    # reset restores the start
        assert again['best'] == first['best'] and again['spikes'] == first['spikes']
    b.close(); thread.join(2)


def test_moving_sum_decision_crosses_requests_and_resets():
    """decision_window > 1: the running sum continues across requests (13-frame hops,
    window 7) and a reset clears it; detections and their frames equal the oracle."""
    from model import decision_scores, integer_forward_stream
    from protocol import request_frames, MODE_RESET
    q = small_stream_model()
    frames = np.random.default_rng(2).integers(0, 256, (1, 260, 24), dtype=np.uint8)
    raw, _, _ = integer_forward_stream(frames, q)
    d = decision_scores(raw[0], 7)
    q['decision_window'] = np.array(7)
    q['stream_threshold'] = np.array(int(np.quantile(d, .9)))   # a few crossings
    th, expected, last, at = int(q['stream_threshold']), [], None, []
    for f in range(len(d)):   # firmware rule: 100-frame hold-off
        if d[f] >= th and (last is None or f - last >= 100):
            last = f; at.append(f)
    backend = SimulatedBoard.__new__(SimulatedBoard)
    backend.np, backend.q, backend.abi, backend.led_until = np, q, 3, 0.
    backend.state, backend.frames_done, backend.last_event = None, 0, None
    backend.window, backend.recent = 7, np.zeros(0, np.int64)
    a, b = socket.socketpair()
    thread = threading.Thread(target=handle, args=(b, backend), daemon=True); thread.start()
    with a:
        seq = iter(range(1, 1000))
        for _ in range(2):   # the second round checks that the reset cleared the sum
            request_frames(a, next(seq), b'', MODE_RESET)
            got = []
            for f0 in range(0, 260, 13):
                r = request_frames(a, next(seq), frames[0, f0:f0 + 13].tobytes())
                if r['detected']:
                    got.append(f0 + r['at_frame'])
            assert got == at and at
    b.close(); thread.join(2)


def test_v2_request_to_v3_backend_is_refused():
    backend = SimulatedBoard.__new__(SimulatedBoard)
    backend.np, backend.q, backend.abi = np, small_stream_model(), 3
    a, b = socket.socketpair()
    thread = threading.Thread(target=handle, args=(b, backend), daemon=True); thread.start()
    with a:
        with pytest.raises(RuntimeError, match='error 2'):
            request(a, 1, bytes(N_INPUT))
    b.close(); thread.join(2)
