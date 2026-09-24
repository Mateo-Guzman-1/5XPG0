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


# Loaded over JTAG (jtag/bringup.tcl): SD boot failed on the available board, so
# the PYNQ Linux / Ethernet path (board_server.py) has not run on hardware.
BOARD_TESTED={'method':'JTAG (jtag/bringup.tcl, jtag/board_test.tcl)','date':'2026-09-24',
              'result':'40/40 verification vectors bit-exact, LED0 pulse about 1 s',
              'not_tested':'PYNQ Linux, Ethernet board_server.py, PYNQ Clocks setup'}


def implementation(project,released):
    """Timing, resources, and bitstream identity of one Vivado project."""
    timing=(project/'timing_summary.rpt').read_text()
    util=(project/'utilization.rpt').read_text()
    drc=(project/'drc.rpt').read_text()
    log=(project/'rv.runs/impl_1/runme.log').read_text()
    if 'All user specified timing constraints are met.' not in timing or 'Bitgen Completed Successfully' not in log:
        raise ValueError(f'{project}: implementation did not meet release criteria')
    if re.search(r'\|\s*(Error|Critical Warning)\s*\|',drc):
        raise ValueError(f'{project}: unresolved implementation DRC error')
    built=project/'rv.runs/impl_1/spike_top.bit'
    if sha(built)!=sha(released):
        raise ValueError(f'{released.name} does not match the measured project')
    setup=re.search(r'Setup\s*:.*?Worst Slack\s+(-?[\d.]+)ns',timing).group(1)
    hold=re.search(r'Hold\s*:.*?Worst Slack\s+(-?[\d.]+)ns',timing).group(1)
    resources={key:int(re.search(r'^\|\s*'+re.escape(label)+r'\s*\|\s*(\d+)',util,re.M).group(1))
               for key,label in [('luts','Slice LUTs'),('registers','Slice Registers'),('bram36','Block RAM Tile'),('dsps','DSPs')]}
    return {'tool':'Vivado 2025.2','part':'xc7z020clg400-1','clock_hz':100000000,
            'setup_slack_ns':float(setup),'hold_slack_ns':float(hold),'resources':resources,
            'drc_errors':0,'bitstream_sha256':sha(built)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vivado-project',type=Path,default=ROOT/'build/vivado_final')
    p.add_argument('--kdot-project',type=Path,default=ROOT/'build/vivado_kdot',
                   help='Vivado project of deploy/keyword_kdot.bit (skipped if absent)')
    a=p.parse_args()
    result=ROOT/'results'; deploy=ROOT/'deploy'
    verification=json.loads((result/'verification.json').read_text())
    if sha(deploy/'model.npz')!=verification['model_sha256'] or sha(deploy/'keyword.bin')!=verification['firmware_sha256']:
        raise ValueError('Deployment artifacts differ from verified artifacts')
    if verification['native_c']['score_and_spike_mismatches'] or verification['rtl']['score_and_spike_mismatches']:
        raise ValueError('Inference verification failed')
    hardware=implementation(a.vivado_project,deploy/'keyword.bit')
    hardware.update(drc_advisories='4 DPOP-1 and 6 DPOP-2 DSP pipeline warnings',
                    unconstrained_outputs='10 asynchronous human-visible LED pins have no external output delay',
                    board_tested=BOARD_TESTED)
    (result/'hardware.json').write_text(json.dumps(hardware,indent=2))
    for name in ['timing_summary.rpt','utilization.rpt','drc.rpt']:
        shutil.copyfile(a.vivado_project/name,result/name)
    files=['keyword.bin','keyword.bit','model.npz']
    if (a.kdot_project/'timing_summary.rpt').exists():
        kdot=implementation(a.kdot_project,deploy/'keyword_kdot.bit')
        kdot.update(hardware_abi='0x00020001',board_tested=BOARD_TESTED)
        (result/'hardware_kdot.json').write_text(json.dumps(kdot,indent=2))
        for name in ['timing_summary.rpt','utilization.rpt','drc.rpt']:
            shutil.copyfile(a.kdot_project/name,result/f'kdot_{name}')
        files+=['keyword_kdot.bin','keyword_kdot.bit','ps7_init.tcl']
    (result/'models').mkdir(exist_ok=True)
    for model in (ROOT/'runs').glob('*/model.npz'):
        shutil.copyfile(model,result/'models'/f'{model.parent.name}.npz')
    manifest={'hardware_abi':{'keyword.bit':'0x00020000','keyword_kdot.bit':'0x00020001'},'keyword':'yes',
              'board_tested':BOARD_TESTED,
              'files':{name:{'sha256':sha(deploy/name),'bytes':(deploy/name).stat().st_size} for name in files},
              'sources_sha256_lf_normalized':{p.relative_to(ROOT.parent.parent).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                for p in sorted((ROOT.parent/'pynqz2_riscv_flow/rtl').glob('*.v'))}}
    (deploy/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(hardware,indent=2))


if __name__=='__main__': main()
