#!/usr/bin/env python3
"""Plot test accuracy vs. r_test from uea/droprate.py's results.csv (paper Figure 6).

Lines are the mean over training seeds.  Colours and markers match the Fading
Flash OOD figure, so a model looks the same in both; the Mamba-1/2/3 colours
were chosen to stay distinguishable from the other seven (OKLab ΔE >= 18 normal
vision, >= 11.7 under simulated protan/deutan), and every series also has its
own marker.  Mamba-3 uses a stroke-only x so a point it shares with RFormer
still shows RFormer's marker underneath.

Usage:
    python uea/plot_droprate.py results/droprate/results.csv [--out fig_droprate.pdf]
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_TRAIN = 0.5
STYLE = {  # results.csv name: (label, colour, marker)
    'S5':           (r'S5',                   '#0072B2', '^'),
    'TIDES_Lambda': (r'TIDES$_\Lambda$',      '#17BECF', '*'),
    'Mamba_S':      (r'Mamba$_{\mathrm{S}}$', '#CC0000', 's'),
    'TIDES_BC':     (r'TIDES$_\mathrm{BC}$',  '#D55E00', 'v'),
    'TIDES':        (r'TIDES',                '#009E73', 'o'),
    'TIDES_full':   (r'TIDES$_\mathrm{full}$', '#E69F00', 'D'),
    'RFormer':      (r'RFormer',              '#CC79A7', 'P'),
    'Mamba':        (r'Mamba',                '#1100EE', 'p'),
    'Mamba2':       (r'Mamba-2',              '#773366', 'h'),
    'Mamba3':       (r'Mamba-3',              '#8866FF', 'x'),
}

_FS = 9
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.25,
    'font.size': _FS, 'axes.labelsize': _FS, 'axes.titlesize': _FS,
    'legend.fontsize': _FS - 1, 'xtick.labelsize': _FS - 1, 'ytick.labelsize': _FS - 1,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('results_csv', nargs='+', help='results.csv file(s) from uea/droprate.py')
    p.add_argument('--out', default='fig_droprate.pdf', help='.pdf or .png')
    p.add_argument('--ylim', nargs=2, type=float, default=[0.38, 0.80])
    args = p.parse_args()

    accs = defaultdict(list)
    for path in args.results_csv:
        with open(path) as f:
            for row in csv.DictReader(f):
                accs[(row['config'], float(row['r_test']))].append(float(row['test_acc']))
    r_tests = sorted({r for _, r in accs})

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.axvline(R_TRAIN, color='gray', lw=1.0, ls='--', zorder=1, label=r'$r_\mathrm{train}$')
    for name, (label, colour, marker) in STYLE.items():
        if not any((name, r) in accs for r in r_tests):
            continue
        means = [np.mean(accs[(name, r)]) if (name, r) in accs else np.nan for r in r_tests]
        ax.plot(r_tests, means, marker + '-', color=colour, lw=1.2,
                ms=7 if marker == '*' else 5, label=label, zorder=3)
        print(f'{name:13s} ' + '  '.join(f'{m:.3f}' for m in means) +
              f'   (n={len(accs[(name, r_tests[0])])} seeds)')

    ax.set_xlabel(r'Test drop rate $r_\mathrm{test}$')
    ax.set_ylabel('Test accuracy')
    ax.set_xticks(r_tests)
    ax.set_xlim(min(r_tests) - 0.05, max(r_tests) + 0.05)
    ax.set_ylim(*args.ylim)
    ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), framealpha=0.9,
              handlelength=1.8, borderaxespad=0)
    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(right=0.72)
    fig.savefig(args.out, bbox_inches='tight', dpi=200)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
