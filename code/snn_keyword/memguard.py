"""Self-limiting jobs on the shared machine: exit cleanly (code 3) when over budget.

check() is called every 50 training steps and in every cache loop. Limits:
private (commit) memory 6 GB, reserved GPU memory 2.3 GB.
"""
import sys

import psutil

MAX_PRIVATE = 6 * 2 ** 30
MAX_GPU = int(2.3 * 2 ** 30)


def private_bytes():
    m = psutil.Process().memory_info()
    return getattr(m, 'private', m.rss)


def check(where=''):
    p = private_bytes()
    if p > MAX_PRIVATE:
        print(f'memguard: private memory {p / 2 ** 30:.2f} GB > 6 GB at {where}; exiting', flush=True)
        sys.exit(3)
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            g = torch.cuda.memory_reserved()
            if g > MAX_GPU:
                print(f'memguard: reserved GPU memory {g / 2 ** 30:.2f} GB > 2.3 GB at {where}; exiting', flush=True)
                sys.exit(3)


def free_ok(min_ram_gb=10., min_gpu_gb=2.5):
    """Start condition for GPU jobs: free system RAM and free GPU memory."""
    ram = psutil.virtual_memory().available / 2 ** 30
    gpu = None
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        gpu = free / 2 ** 30
    ok = ram >= min_ram_gb and (gpu is None or gpu >= min_gpu_gb)
    return ok, {'free_ram_gb': round(ram, 2), 'free_gpu_gb': None if gpu is None else round(gpu, 2)}


def count_instances(pids):
    """Distinct jobs among `pids`: a venv launcher and its interpreter child count once."""
    s, n = set(pids), 0
    for p in pids:
        try:
            n += psutil.Process(p).ppid() not in s
        except psutil.Error:
            pass
    return n


def other_instances(script):
    """PIDs of other processes running `script`, excluding this process, its parents and children
    (the venv launcher on Windows starts the real interpreter as a child with the same command line)."""
    me = psutil.Process()
    mine = {me.pid} | {q.pid for q in me.parents()} | {q.pid for q in me.children(recursive=True)}
    out = []
    for q in psutil.process_iter(['pid', 'cmdline', 'name']):
        try:
            cmd = ' '.join(q.info['cmdline'] or [])
        except Exception:
            continue
        if q.info['pid'] not in mine and 'python' in (q.info['name'] or '').lower() and script in cmd:
            out.append(q.info['pid'])
    return out
