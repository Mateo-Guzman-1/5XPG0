"""Check seed-zero models for all encodings, restoring the running board demo."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pedro_encoding_eval as study
import benchmark_extended_sweep_board as board
from export_extended_candidate import export_candidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', default='192.168.2.99')
    p.add_argument('--bash', type=Path, default=Path('C:/msys64/usr/bin/bash.exe'))
    p.add_argument('--out', type=Path, default=study.HERE / 'results/pedro_encoding')
    a = p.parse_args()
    flow = study.KWS.parent / 'pynqz2_riscv_flow'
    firmware, host = flow / 'firmware', flow / 'host'
    remote = '/home/xilinx/snn_keyword'
    login = f'xilinx@{a.board}'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', login]
    def remote_run(command):
        return subprocess.check_output([*ssh, command], text=True).strip()
    backup = remote_run('mktemp -d /home/xilinx/encoding-eval-backup.XXXXXX')
    remote_run(f'cp -a {remote} {backup}/')
    original = {name: (firmware / name).read_bytes() for name in ('keyword_model.h', 'keyword.bin')}
    active = remote_run('systemctl is-active snn-keyword.service || true')
    results = {}
    try:
        remote_run('sudo -n systemctl stop snn-keyword.service')
        remote_run(f'cd {remote} && ./keyword_service.sh stop || true')
        subprocess.run(['scp', '-q', str(flow / 'vivado/spike_top.bit'),
                        *[str(host / f) for f in ('spike_pynq.py', 'keyword_bridge.py', 'keyword_service.sh')],
                        f'{login}:{remote}/'], check=True)
        remote_run(f"sudo -n bash -lc 'source /etc/profile.d/xrt_setup.sh; cd {remote}; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-bit spike_top.bit'")
        configs, _ = study.catalog()
        for i, c in enumerate(configs):
            ck = a.out / 'models' / f'{c.key}_seed0.pt'
            while not ck.exists():
                time.sleep(10)
            candidate = a.out / 'board_exports' / c.key
            ex = export_candidate(ck, study.KWS / 'data/speech_commands_v2_sheila_subset', 'sheila',
                                  study.KWS / 'data/extended_encoding_cache_sheila_v2' / f'{c.key}.npy', candidate)
            shutil.copyfile(candidate / 'keyword_model.h', firmware / 'keyword_model.h')
            sizes = board.build_firmware(firmware, a.bash)
            shutil.copyfile(firmware / 'keyword.bin', candidate / 'keyword.bin')
            subprocess.run(['scp', '-q', str(candidate / 'keyword.bin'), str(candidate / 'audio_features.py'),
                            f'{login}:{remote}/'], check=True)
            remote_run(f"cd {remote} && ./keyword_service.sh stop >/dev/null 2>&1 || true; sudo -n bash -lc 'source /etc/profile.d/xrt_setup.sh; cd {remote}; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py stop; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-elf keyword.bin; /usr/local/share/pynq-venv/bin/python3 spike_pynq.py start'; cd {remote} && ./keyword_service.sh start")
            subprocess.run([sys.executable, str(study.HERE / 'benchmark_candidate_board.py'), a.board,
                            str(candidate / 'board_vectors.npz'), '--bands', str(c.board_rows),
                            '--time-bins', str(c.time_bins), '--label', c.key,
                            '--output', str(candidate / 'board.json')], check=True)
            measured = json.loads((candidate / 'board.json').read_text())
            if measured['score_mismatches']:
                raise RuntimeError(f'FPGA mismatch: {c.key}')
            results[c.key] = dict(config=asdict(c), **sizes, **measured)
            study.dump(a.out / 'board.json', dict(board=a.board, seed=0, candidates=results))
            print('BOARD COMPLETE', i + 1, '/', len(configs), flush=True)
    finally:
        for name, contents in original.items():
            (firmware / name).write_bytes(contents)
        remote_run(f'cd {remote} && ./keyword_service.sh stop >/dev/null 2>&1 || true; sudo -n cp -a {backup}/snn_keyword/. {remote}/')
        if active == 'active':
            remote_run('sudo -n systemctl restart snn-keyword.service')
        study.dump(a.out / 'board_restoration.json', dict(backup=backup, original_service=active,
                   current_service=remote_run('systemctl is-active snn-keyword.service || true')))


if __name__ == '__main__':
    main()
