"""Collect measured implementation evidence and hash the deployable artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vivado-project',type=Path,default=ROOT/'build/vivado_final')
    a=p.parse_args()
    result=ROOT/'results'; deploy=ROOT/'deploy'
    verification=json.loads((result/'verification.json').read_text())
    if sha(deploy/'model.npz')!=verification['model_sha256'] or sha(deploy/'keyword.bin')!=verification['firmware_sha256']:
        raise ValueError('Deployment artifacts differ from verified artifacts')
    if verification['native_c']['score_and_spike_mismatches'] or verification['rtl']['score_and_spike_mismatches']:
        raise ValueError('Inference verification failed')
    timing=(a.vivado_project/'timing_summary.rpt').read_text()
    util=(a.vivado_project/'utilization.rpt').read_text()
    drc=(a.vivado_project/'drc.rpt').read_text()
    log=(a.vivado_project/'rv.runs/impl_1/runme.log').read_text()
    if 'All user specified timing constraints are met.' not in timing or 'Bitgen Completed Successfully' not in log:
        raise ValueError('Implementation did not meet release criteria')
    if re.search(r'\|\s*(Error|Critical Warning)\s*\|',drc):
        raise ValueError('Unresolved implementation DRC error')
    built=a.vivado_project/'rv.runs/impl_1/spike_top.bit'
    if sha(built)!=sha(deploy/'keyword.bit'):
        raise ValueError('Released bitstream does not match the measured project')
    setup=re.search(r'Setup\s*:.*?Worst Slack\s+(-?[\d.]+)ns',timing).group(1)
    hold=re.search(r'Hold\s*:.*?Worst Slack\s+(-?[\d.]+)ns',timing).group(1)
    resources={key:int(re.search(r'^\|\s*'+re.escape(label)+r'\s*\|\s*(\d+)',util,re.M).group(1))
               for key,label in [('luts','Slice LUTs'),('registers','Slice Registers'),('bram36','Block RAM Tile'),('dsps','DSPs')]}
    hardware={'tool':'Vivado 2025.2','part':'xc7z020clg400-1','clock_hz':100000000,
              'setup_slack_ns':float(setup),'hold_slack_ns':float(hold), 'resources':resources,
              'drc_errors':0,'drc_advisories':'4 DPOP-1 and 6 DPOP-2 DSP pipeline warnings',
              'unconstrained_outputs':'10 asynchronous human-visible LED pins have no external output delay',
              'board_tested':False,'bitstream_sha256':sha(built)}
    (result/'hardware.json').write_text(json.dumps(hardware,indent=2))
    for name in ['timing_summary.rpt','utilization.rpt','drc.rpt']:
        shutil.copyfile(a.vivado_project/name,result/name)
    (result/'models').mkdir(exist_ok=True)
    for model in (ROOT/'runs').glob('*/model.npz'):
        shutil.copyfile(model,result/'models'/f'{model.parent.name}.npz')
    manifest={'hardware_abi':'0x00020000','keyword':'yes','board_tested':False,
              'files':{name:{'sha256':sha(deploy/name),'bytes':(deploy/name).stat().st_size}
                       for name in ['keyword.bin','keyword.bit','model.npz']},
              'sources_sha256_lf_normalized':{p.relative_to(ROOT.parent.parent).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                for p in sorted((ROOT.parent/'pynqz2_riscv_flow/rtl').glob('*.v'))}}
    (deploy/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(hardware,indent=2))


if __name__=='__main__': main()
