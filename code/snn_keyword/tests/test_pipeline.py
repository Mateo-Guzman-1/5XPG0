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
from protocol import request, recv_exact, HEADER, RESPONSE, MAGIC
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
    def infer(self,payload):
        assert len(payload)==768
        return (0,-3,12,12345,3,1)


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
        packet=HEADER.pack(MAGIC,7,0xffffffff)
        for byte in packet: a.sendall(bytes([byte]))
        reply=RESPONSE.unpack(recv_exact(a,RESPONSE.size))
        assert reply[1:3]==(7,1)
    thread.join(timeout=2); b.close()
    assert not thread.is_alive()


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
