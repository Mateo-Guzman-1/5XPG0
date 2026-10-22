"""Export one native PGFPlots chart per swept variable for the report."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent / 'results/pedro_encoding'
    agg = json.loads((root / 'aggregate.json').read_text())
    source = json.loads((root / 'summary.json').read_text())
    rows = agg['rows']
    labels = {'bits': ('bits', 'Feature precision (bits)'), 'bands': ('bands', 'Frequency bands'),
              'time_bins': ('time_bins', 'Time bins'), 'duration': ('duration_ms', 'Represented duration (ms)'),
              'band_mapping': ('band_mapping', 'Band mapping'), 'quantizer': ('quantizer', 'Four-bit quantisation'),
              'feature_mode': ('feature_mode', 'Features')}
    dest = a.report / 'figures/encoding'
    dest.mkdir(parents=True, exist_ok=True)
    plot_data = {}
    for group, keys in source['groups'].items():
        if group in ('feature_mode', 'quantizer'):
            continue  # covered briefly in text/table, avoiding redundant charts
        numeric = group in ('bits', 'bands', 'time_bins', 'duration')
        points, ticks = [], []
        for i, key in enumerate(keys):
            r = rows[key]
            c = r['config']
            if group in labels:
                field, xlabel = labels[group]
                value = c[field]
                text = {'absolute': 'Magnitude', 'absolute_delta': 'Magnitude+delta'}.get(str(value), str(value))
            elif group == 'range':
                value, xlabel = i + 1, 'Passband (kHz)'
                text = f"{c['low_hz']/1000:g}--{c['high_hz']/1000:g}"
            elif group == 'activity':
                value, xlabel = i + 1, 'RMS gate / full scale'
                text = f"{c['rms_gate_u6']/1e6:g}/{c['rms_full_scale_u6']/1e6:g}"
            else:
                value, xlabel = i + 1, 'FFT window / hop (ms)'
                text = f"{c['fft_window_ms']}/{c['fft_hop_ms']}"
            x = value if numeric else i + 1
            mean, sd = r['live_recall']['mean'] * 100, r['live_recall']['sd'] * 100
            points.append((x, mean, sd))
            ticks.append(text)
        plot_data[group] = dict(points=points, labels=ticks)
        options = [r'width=.46\textwidth', 'height=4.2cm', f'xlabel={{{xlabel}}}',
                   r'ylabel={Live recall (\%)}', 'ymin=0', 'ymax=100', 'ytick={0,25,50,75,100}',
                   'grid=major', r'tick label style={font=\scriptsize}', r'label style={font=\small}']
        if numeric:
            tick_values = [x for x, _, _ in points]
            if group == 'duration':
                tick_values = [400, 800, 1200, 1600]
            options.append('xtick={' + ','.join(map(str, tick_values)) + '}')
        else:
            options += ['xtick={' + ','.join(str(i + 1) for i in range(len(keys))) + '}',
                        'xticklabels={' + ','.join(ticks) + '}']
            if len(keys) > 3:
                options.append(r'x tick label style={rotate=55,anchor=east,font=\tiny}')
        coords = '\n'.join(f'({x},{m:.5f}) +- (0,{s:.5f})' for x, m, s in points)
        tex = ('% Generated from the completed three-seed streaming evaluation.\n'
               '\\begin{tikzpicture}\n\\begin{axis}[\n' + ',\n'.join(options) + ']\n'
               '\\addplot[blue,thick,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {\n'
               + coords + '\n};\n\\end{axis}\n\\end{tikzpicture}\n')
        (dest / f'streaming_{group}.tex').write_text(tex, encoding='utf8')
    (dest / 'streaming_plot_data.json').write_text(json.dumps(plot_data, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
