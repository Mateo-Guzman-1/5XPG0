"""Parse Vivado `report_power` text reports into JSON, keeping every raw number.

    python parse_power.py research/results/S0/vivado/base research/results/S0/vivado/kdot \
        --out research/results/S0/power.json

Every folder is one fabric (its name is the key); every power*.rpt in it becomes one JSON object
(key = file stem) with the report header, the summary table, the on-chip components, the
power-supply table, the environment settings and the hierarchy (with depth). Nothing is derived
here; research/tools/energy_model.py turns these values into energy.
"""
import argparse
import json
import re
from pathlib import Path


def _number(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return text


def _cells(line):
    """Cells of one '| a | b |' row; the first cell keeps its leading spaces (hierarchy depth)."""
    parts = line.rstrip().rstrip('|').split('|')[1:]
    return [parts[0].rstrip()] + [p.strip() for p in parts[1:]]


def tables(text):
    """(section title, rows) for every ASCII table; rows are lists of cells, header row included."""
    lines = text.splitlines()
    section, i, out = None, 0, []
    while i < len(lines):
        line = lines[i]
        heading = re.match(r'^\d+(?:\.\d+)*\.?\s+(\S.*)$', line.strip())   # "1. Summary", "1.1 On-Chip ..."
        if heading and i + 1 < len(lines) and lines[i + 1].strip() and set(lines[i + 1].strip()) == {'-'}:
            section = heading.group(1).strip()
        if line.startswith('+') and set(line.strip()) <= set('+-'):
            rows = []
            while i < len(lines) and lines[i].startswith(('+', '|')):
                if lines[i].startswith('|'):
                    rows.append(_cells(lines[i]))
                i += 1
            out.append((section, rows))
            continue
        i += 1
    return out


def parse(path):
    text = Path(path).read_text(errors='replace')
    header = {}
    for key in ('Tool Version', 'Date', 'Design', 'Device', 'Design State', 'Command'):
        m = re.search(rf'^\|\s*{re.escape(key)}\s*:\s*(.*?)\s*$', text, re.M)
        if m:
            header[key] = m.group(1)
    report = {'file': str(path), 'header': header}
    for section, rows in tables(text):
        if not rows:
            continue
        name = (section or '').lower()
        if name == 'summary':
            report['summary'] = {r[0].strip(): _number(r[1]) for r in rows if len(r) >= 2}
        elif name == 'on-chip components':
            head = [c.strip() for c in rows[0]]
            report['on_chip'] = [dict(zip(head, [r[0].strip()] + [_number(c) for c in r[1:]]),
                                      depth=(len(r[0]) - len(r[0].lstrip()) - 1) // 2)
                                 for r in rows[1:]]
        elif name == 'power supply summary':
            head = [c.strip() for c in rows[0]]
            report['supplies'] = [dict(zip(head, [r[0].strip()] + [_number(c) for c in r[1:]]))
                                  for r in rows[1:]]
        elif name == 'environment':
            report['environment'] = {r[0].strip(): _number(r[1]) for r in rows if len(r) >= 2}
        elif name == 'by hierarchy':
            head = [c.strip() for c in rows[0]]
            report['hierarchy'] = [dict(zip(head, [r[0].strip()] + [_number(c) for c in r[1:]]),
                                        depth=(len(r[0]) - len(r[0].lstrip()) - 1) // 2)
                                   for r in rows[1:]]
    missing = [k for k in ('summary', 'on_chip', 'supplies', 'hierarchy') if k not in report]
    if missing:
        raise SystemExit(f'{path}: tables not found: {missing}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+', type=Path, help='one folder of power*.rpt per fabric')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = {d.name: {r.stem: parse(r) for r in sorted(d.glob('power*.rpt'))} for d in a.folders}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=2))
    for fabric, reports in data.items():
        for stem, r in reports.items():
            s = r['summary']
            print(f"{fabric}/{stem}: total {s.get('Total On-Chip Power (W)')} W, dynamic {s.get('Dynamic (W)')} W, "
                  f"static {s.get('Device Static (W)')} W, confidence {s.get('Confidence Level')}")


if __name__ == '__main__':
    main()
