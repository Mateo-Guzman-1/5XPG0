"""Energy per inference and per 250 ms window, from Vivado's power estimate and board cycle counts.

Inputs:
- research/results/S0/power.json: Vivado `report_power` of the two fabrics (parse_power.py output;
  post-route, vectorless, 25 C, typical process). Fabric "base" = RV32IM release design (commit
  714db5c), fabric "kdot" = release design with the kdot coprocessor (HEAD).
- cycles: measured on the board, bit-exact runs only (mean for energy, max for latency).

Power split (all from the raw report, no other assumptions):
- P_PL,dyn  = dynamic power - PS7 block power (programmable logic only; the ARM side is the host)
- P_PL,stat = static power of the PL supply rails (V x static current of Vccint, Vccaux, Vccbram,
              Vccadc and the Vcco I/O banks); the PS rails (Vccp*, DDR, MIO) are left out
- P_core,dyn = dynamic power of the u_soc instance (PicoRV32 + kdot + 256 KiB BRAM)
- P_infra,dyn = P_PL,dyn - P_core,dyn (AXI interface to the ARM, clock buffers, top level)

Energy (f = 100 MHz, T = 0.25 s):
- E_inf       = P_PL,dyn x C_mean / f                     dynamic energy of one inference
- E_inf,all   = (P_PL,dyn + P_PL,stat) x C_mean / f       incl. the static energy during it
- (a) busy-poll   E_win = (P_PL,dyn + P_PL,stat) x T      today's firmware: never idle
- (b) clock-gated E_win = (P_PL,stat + P_infra,dyn) x T + P_core,dyn x C_mean / f
- (c) scaled      E_win = P_PL,stat x T + P_PL,dyn x 1.1 C_max / f
                  (FCLK0 lowered to f_min = 1.1 C_max / T; dynamic power proportional to f)
- average power = E_win / T

Vectorless power does not depend on the workload, so within one fabric the energy of two models
differs only through their cycle counts. See research/ENERGY.md.

    python energy_model.py --fabric kdot --cycles-mean 393210 --cycles-max 395945 --macs 73150
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # research/
POWER_JSON = ROOT / 'results/S0/power.json'
F_CLK = 100e6        # FCLK0 (Hz)
T_HOP = 0.25         # time between windows (s)
MARGIN = 1.1         # clock-scaling scenario keeps 10 % slack on the worst-case cycles
FABRICS = ('base', 'kdot')
PS_RAILS = ('vccpint', 'vccpaux', 'vccpll', 'vcco_ddr', 'vcco_mio0', 'vcco_mio1')
CORE = 'u_soc'


def _component(report, name):
    for row in report['on_chip']:
        if row['On-Chip'].lower() == name.lower():
            return float(row['Power (W)'])
    raise KeyError(f'no on-chip row {name!r}')


def _instance(report, name):
    for row in report['hierarchy']:
        if row['Name'] == name:
            return float(row['Power (W)'])
    raise KeyError(f'no hierarchy row {name!r}')


def split(report):
    """Power terms (W) of one parsed report_power report."""
    dynamic = float(report['summary']['Dynamic (W)'])
    ps7 = _component(report, 'PS7')
    pl_static = ps_static = 0.0
    rails = {}
    for s in report['supplies']:
        name, volts, amps = s['Source'], float(s['Voltage (V)']), float(s['Static (A)'])
        rails[name] = volts * amps
        if name.lower() in PS_RAILS:
            ps_static += volts * amps
        else:
            pl_static += volts * amps
    pl_dyn = dynamic - ps7
    core = _instance(report, CORE)
    return dict(dynamic=dynamic, ps7=ps7, pl_dyn=pl_dyn, core_dyn=core, infra_dyn=pl_dyn - core,
                pl_static=pl_static, ps_static=ps_static,
                device_static=float(report['summary']['Device Static (W)']),
                total=float(report['summary']['Total On-Chip Power (W)']), static_by_rail=rails)


EXTRA_POWER = [ROOT / 'results/S6/power.json']   # later fabrics (S6: 'ksdot'), each in its own file


def load_power(path=POWER_JSON, report='power'):
    """{fabric: power split} for the given report name (power, power_toggle6.25, power_toggle25).
    The S0 file gives 'base' and 'kdot'; fabrics measured later are added from their own files."""
    data = json.loads(Path(path).read_text())
    out = {fabric: split(data[fabric][report]) for fabric in FABRICS}
    if Path(path) == POWER_JSON:
        for extra in EXTRA_POWER:
            if extra.exists():
                out.update({fabric: split(reports[report])
                            for fabric, reports in json.loads(extra.read_text()).items()})
    return out


def energy(cycles_mean, cycles_max, p, macs=None):
    """Energy and power figures of one model on one fabric (SI units: J, W, s, Hz)."""
    pl = p['pl_dyn'] + p['pl_static']
    e = dict(t_mean=cycles_mean / F_CLK, t_max=cycles_max / F_CLK,
             e_cycle=p['pl_dyn'] / F_CLK,
             e_inf=p['pl_dyn'] * cycles_mean / F_CLK,
             e_inf_all=pl * cycles_mean / F_CLK,
             e_win_busy=pl * T_HOP,
             e_win_gated=(p['pl_static'] + p['infra_dyn']) * T_HOP + p['core_dyn'] * cycles_mean / F_CLK,
             f_min=MARGIN * cycles_max / T_HOP,
             e_win_scaled=p['pl_static'] * T_HOP + p['pl_dyn'] * MARGIN * cycles_max / F_CLK)
    for k in ('busy', 'gated', 'scaled'):
        e[f'p_avg_{k}'] = e[f'e_win_{k}'] / T_HOP
    if macs:
        e['e_per_mac'] = e['e_inf'] / macs
        e['e_all_per_mac'] = e['e_inf_all'] / macs
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fabric', choices=FABRICS, required=True)
    p.add_argument('--cycles-mean', type=float, required=True)
    p.add_argument('--cycles-max', type=float, required=True)
    p.add_argument('--macs', type=float, help='executed synaptic multiply-adds per inference')
    p.add_argument('--report', default='power', help='power, power_toggle6.25 or power_toggle25')
    a = p.parse_args()
    power = load_power(report=a.report)[a.fabric]
    print('power terms (W):', {k: round(v, 4) for k, v in power.items() if k != 'static_by_rail'})
    for k, v in energy(a.cycles_mean, a.cycles_max, power, a.macs).items():
        print(f'{k:>14} = {v:.6g}')


if __name__ == '__main__':
    main()
