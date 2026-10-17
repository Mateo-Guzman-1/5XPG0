import socket
import sys
import threading
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import features, mel_bank, N_INPUT
from model import integer_forward, trunc_div, metrics, choose_threshold, keyword_of
from protocol import request, recv_exact, HEADER, RESPONSE, MAGIC, MODE_STREAM
from board_server import handle
from board_server import SimulatedBoard
from board_server import Board


def release_evidence():
    """Folder with the deployed model's verification vectors (deploy/manifest.json "evidence")."""
    import json
    manifest=json.loads((ROOT/'deploy/manifest.json').read_text())
    return ROOT/manifest.get('evidence','results')


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
    vectors=np.load(release_evidence()/'verification_vectors.npz')
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
    sources=release_evidence()/'vector_sources.json'
    if sources.exists():
        import json
        from features import read_wav
        wav=ROOT/'data/speech_commands_v0.02'/json.loads(sources.read_text())[0]
        if wav.exists():
            assert np.array_equal(features(read_wav(wav)),vectors['x'][0])


def test_deploy_matches_manifest():
    import hashlib,json
    manifest=json.loads((ROOT/'deploy/manifest.json').read_text())
    for name,info in manifest['files'].items():
        data=(ROOT/'deploy'/name).read_bytes()
        assert (len(data),hashlib.sha256(data).hexdigest())==(info['bytes'],info['sha256']),name
    assert keyword_of(np.load(ROOT/'deploy/model.npz'))==manifest['keyword']


def test_board_mailbox_layout_and_sequence_wrap():
    import struct
    board=Board.__new__(Board)
    board.ram=bytearray(0x40000);board.reg=bytearray(4096)
    # Board.__init__ makes these 32-bit views over the mapped memory.
    board.ram32=memoryview(board.ram).cast('I');board.reg32=memoryview(board.reg).cast('I')
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
    board.ram32=memoryview(board.ram).cast('I');board.reg32=memoryview(board.reg).cast('I')
    clock=iter([0,6])
    monkeypatch.setattr(board_server.time,'monotonic',lambda:next(clock))
    with pytest.raises(RuntimeError,match='timed out'): board.infer(bytes(768))
    assert board.read_reg(0)==1
    with pytest.raises(RuntimeError,match='stopped'): board.infer(bytes(768))


# ---- Research part 2 (research/PLAN_SYNAPSE.md): synapse precision, fan-in, LIF table ----

def lanes_c_mirror(words,bits):
    """numpy copy of the firmware's lane extraction: (int32)(w << (32-b(k+1))) >> (32-b)."""
    lanes=32//bits; w=words.astype(np.uint64)[:,None]
    k=np.arange(lanes,dtype=np.uint64)[None,:]
    shifted=((w<<(np.uint64(32)-np.uint64(bits)*(k+np.uint64(1))))&np.uint64(0xFFFFFFFF)).astype(np.int64)
    shifted=np.where(shifted>=2**31,shifted-2**32,shifted)
    return (shifted>>(32-bits)).reshape(-1)


def test_default_quantize_is_unchanged():
    import torch
    from model import SpikeMLP, quantize
    torch.manual_seed(0)
    q=quantize(SpikeMLP(4))
    assert sorted(q)==['b1','b2','encoding','q','steps','w1','w2']
    assert q['w1'].dtype==np.int16


@pytest.mark.parametrize('bits',[8,6,4,3,2])
def test_pow2_quantizer_grid_matches_export(bits):
    import torch
    from model import SpikeMLP, quantize, fake_quant, pow2_shift, Q
    torch.manual_seed(bits)
    m=SpikeMLP(4,w1_bits=bits)
    m.fix_w1_shift()
    s=m.w1_shift
    wmax=float(m.fc1.weight.detach().abs().max())*Q
    assert s==pow2_shift(wmax,bits) and wmax/2**s<=2**(bits-1)-1
    q=quantize(m)
    grid=(fake_quant(m.fc1.weight,s,bits).detach().numpy()*Q)
    assert np.array_equal(np.rint(grid).astype(np.int64),q['w1'].astype(np.int64))
    c=q['w1'].astype(np.int64)>>s
    assert np.array_equal(c<<s,q['w1']) and c.min()>=-2**(bits-1) and c.max()<=2**(bits-1)-1
    assert q['w1_bits']==bits and q['w1_shift']==s and q['w1_clipped']==0


@pytest.mark.parametrize('bits',[16,8,4,2])
def test_packed_export_and_drive(bits):
    from export_model import pack_w1
    rng=np.random.default_rng(bits)
    shift=0 if bits==16 else 3
    c=rng.integers(-2**(bits-1),2**(bits-1),size=(3,768))
    if bits==16: c=rng.integers(-500,500,size=(3,768))
    q=dict(w1=(c<<shift).astype(np.int16),w1_bits=bits,w1_shift=shift)
    b,s,lanes,words=pack_w1(q)
    assert (b,s,lanes)==(bits,shift,32//bits) and words.dtype==np.uint32
    assert np.array_equal(lanes_c_mirror(words,bits).reshape(3,768),c)
    # Firmware: sum(c*x) * 2^s / 255 (truncating) == oracle: trunc_div(x @ w1.T, 255), also for negative sums.
    x=rng.integers(0,256,size=(5,768))
    firmware=trunc_div((x@c.T)*2**s,255)
    assert np.array_equal(firmware,trunc_div(x@q['w1'].astype(np.int64).T,255))
    assert (x@c.T<0).any()


def test_fan_in_pruning_and_sparse_export():
    import torch
    from model import SpikeMLP, quantize
    from export_model import sparse_w1
    torch.manual_seed(1)
    m=SpikeMLP(5,fan_in=32)
    m.prune_to(100); m.prune_to(32)
    assert (m.w1_mask.sum(1)==32).all() and (m.fc1.weight[m.w1_mask==0]==0).all()
    q=quantize(m)
    assert q['fan_in']==32 and (q['w1_mask'].sum(1)==32).all() and (q['w1'][q['w1_mask']==0]==0).all()
    k,words=sparse_w1(q)
    assert k==32 and len(words)==5*32
    w=(words.astype(np.int64)>>16); w=np.where(w>=2**15,w-2**16,w); idx=words&0x3ff
    dense=np.zeros((5,768),np.int64); dense[np.repeat(np.arange(5),32),idx]=w
    assert np.array_equal(dense,q['w1'])


def test_lif_threshold_table_is_exact():
    from model import lif_thresholds, lif_count, Q
    t=lif_thresholds(12)
    assert t==[161,233,310,388,489,547,660,728,818,906,988,1024]
    d=np.arange(-2**20,2**20,dtype=np.int64)
    v=np.zeros_like(d); n=np.zeros_like(d)
    for _ in range(12):   # the oracle's recurrence, for every drive at once
        v=trunc_div(v*7,8)+d; spk=v>=Q; v-=spk*Q; n+=spk
    assert np.array_equal(n,(d[:,None]>=np.array(t)[None,:]).sum(1))
    assert [lif_count(x,12) for x in (160,161,1023,1024)]==[0,1,11,12]
