"""R2 step B0 (research/PLAN_R2_GRU.md): does PyTorch train on the AMD GPU? Every GPU result is compared with the CPU.

Checks: torch.cuda.is_available, the device name, a matmul, torch.fft.rfft, F.ctc_loss (forward and backward), and a
GRU forward and backward pass shaped like Mateo's verifier (48 -> 64 -> 64, 75 steps, batch 256); then the time
of 20 GRU training steps on the GPU and on the CPU.

Usage: .venv-rocm/Scripts/python.exe research/tools/r2_gpu_smoke.py   ->   research/results/R2/B/gpu_smoke.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[1] / 'results/R2/B/gpu_smoke.json'


def close(a, b, tol=1e-3):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return float((a - b).abs().max() / (b.abs().max() + 1e-12)) < tol


def gru_net(dev):
    torch.manual_seed(0)
    return torch.nn.ModuleDict(dict(g1=torch.nn.GRU(48, 64, batch_first=True), g2=torch.nn.GRU(64, 64, batch_first=True),
                                    out=torch.nn.Linear(64, 40))).to(dev)


def gru_loss(net, x, tgt, tl):
    h, _ = net['g1'](x)
    h, _ = net['g2'](h)
    lp = net['out'](h).log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(lp, tgt, torch.full((x.shape[0],), x.shape[1], dtype=torch.long), tl, zero_infinity=True)


def main():
    import os
    # MIOpen's run-time kernel build fails on Windows (missing <type_traits> in hipRTC); R2_NO_MIOPEN=1 uses
    # PyTorch's own GRU kernels instead.
    if os.environ.get('R2_NO_MIOPEN') == '1':
        torch.backends.cudnn.enabled = False
    r = {'torch': torch.__version__, 'hip': getattr(torch.version, 'hip', None), 'cuda_available': torch.cuda.is_available(),
         'miopen_enabled': torch.backends.cudnn.enabled}
    print(r, flush=True)
    if not r['cuda_available']:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(r, indent=1))
        return
    dev = torch.device('cuda')
    r['device'] = torch.cuda.get_device_name(0)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(512, 768, generator=g), torch.randn(768, 256, generator=g)
    r['matmul_ok'] = close((a.to(dev) @ b.to(dev)), a @ b)
    s = torch.randn(64, 400, generator=g)
    r['rfft_ok'] = close(torch.fft.rfft(s.to(dev), n=512).abs(), torch.fft.rfft(s, n=512).abs())
    x = torch.randn(256, 75, 48, generator=g)
    tgt = torch.randint(1, 40, (256, 12), generator=g)
    tl = torch.full((256,), 12, dtype=torch.long)
    res = {}
    for d in ('cpu', 'cuda'):
        net = gru_net(d)
        loss = gru_loss(net, x.to(d), tgt.to(d), tl.to(d))
        loss.backward()
        res[d] = (loss, net['g1'].weight_ih_l0.grad)
    r['gru_ctc_loss_ok'] = close(res['cuda'][0], res['cpu'][0])
    r['gru_ctc_grad_ok'] = close(res['cuda'][1], res['cpu'][1], 1e-2)
    r['loss_cpu_gpu'] = [float(res['cpu'][0]), float(res['cuda'][0])]
    for d in ('cuda', 'cpu'):
        net = gru_net(d)
        opt = torch.optim.AdamW(net.parameters(), 1e-3)
        xs, ts, ls = x.to(d), tgt.to(d), tl.to(d)
        for i in range(25):
            if i == 5:
                if d == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            opt.zero_grad()
            gru_loss(net, xs, ts, ls).backward()
            opt.step()
        if d == 'cuda':
            torch.cuda.synchronize()
        r[f'step_s_{d}'] = round((time.perf_counter() - t0) / 20, 4)
    r['ok'] = all(r[k] for k in ('matmul_ok', 'rfft_ok', 'gru_ctc_loss_ok', 'gru_ctc_grad_ok'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
