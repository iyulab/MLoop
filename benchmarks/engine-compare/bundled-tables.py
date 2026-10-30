#!/usr/bin/env python3
"""Runs `ensemble-probe` on the small tables scikit-learn ships with (no download), so the ensemble question can be read without OpenML.

    python bundled-tables.py <work-dir> [--budget 20] [--seeds 3] [--tables breast_cancer,wine,...]

For every table and split seed it writes train.csv / test.csv under <work-dir> (a 75 / 25 split, stratified for the classification
tasks), runs the probe, and prints one table: the best single model against the equal-weight averages of the best 3 and 5 trials and
the forward-selected blend, as mean ± standard error over the split seeds, and a verdict per pair (a pair is `equal` when the mean
paired difference is within its standard error). Needs scikit-learn and pandas; the probe is built beforehand:
    dotnet build benchmarks/engine-compare/ensemble-probe -c Release
The tables are tiny (150 - 1,800 rows), so this reads the shape of the answer, not the 16-table result `run.py` gives.
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [('single', lambda p: p['best_single']), ('k3', lambda p: p['average'].get('k3')), ('k5', lambda p: p['average'].get('k5')),
        ('oofpick', lambda p: (p.get('forward') or {}).get('oof_pick_test')), ('forward', lambda p: (p.get('forward') or {}).get('test'))]
# metric -> direction (+1: higher is better), as run.py's PRIMARY
DIRECTION = {'auc': +1, 'log_loss': -1, 'rmse': -1}


def tables():
    from sklearn import datasets
    out = {}
    cancer = datasets.load_breast_cancer(as_frame=True)
    frame = cancer.frame.copy()
    frame['target'] = frame['target'].map(lambda i: cancer.target_names[i])
    out['breast_cancer'] = ('binary', 'target', frame)
    for name, loader in (('wine', datasets.load_wine), ('digits', datasets.load_digits), ('iris', datasets.load_iris)):
        bunch = loader(as_frame=True)
        frame = bunch.frame.copy()
        frame['target'] = frame['target'].map(lambda i, names=bunch.target_names: 'class_' + str(names[i]))
        out[name] = ('multiclass', 'target', frame)
    out['diabetes'] = ('regression', 'target', datasets.load_diabetes(as_frame=True).frame)
    return out


def mean_se(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)


def verdict(a, b, direction):
    """a against b: `better`, `worse` or `equal` (paired over the split seeds)."""
    diffs = [(x - y) * direction for x, y in zip(a, b)]
    mean, se = mean_se(diffs)
    return 'equal' if abs(mean) <= se else ('better' if mean > 0 else 'worse')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work')
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--tables', default='')
    parser.add_argument('--probe', default='dotnet ' + os.path.join(HERE, 'ensemble-probe', 'bin', 'Release', 'net10.0', 'EnsembleProbe.dll'))
    args = parser.parse_args()
    from sklearn.model_selection import train_test_split
    probe = shlex.split(args.probe)
    chosen = [t for t in args.tables.split(',') if t]
    print('| table | task | rows | metric | ' + ' | '.join(name for name, _ in ARMS)
          + ' | k3 vs single | k5 vs single | oofpick vs single | forward vs single | forward vs oofpick | forward vs k5 | forward members used | forward top-member share |')
    print('|---|---|---|---|' + '---|' * (len(ARMS) + 8))
    summary = {key: {'better': 0, 'equal': 0, 'worse': 0} for key in ('k3', 'k5', 'oofpick', 'forward', 'forward vs oofpick', 'forward vs k5')}
    for name, (task, target, frame) in tables().items():
        if chosen and name not in chosen:
            continue
        runs = []
        for seed in range(1, args.seeds + 1):
            directory = os.path.join(args.work, name, str(seed))
            os.makedirs(directory, exist_ok=True)
            train, test = train_test_split(frame, test_size=0.25, random_state=seed, stratify=frame[target] if task != 'regression' else None)
            train.to_csv(os.path.join(directory, 'train.csv'), index=False)
            test.to_csv(os.path.join(directory, 'test.csv'), index=False)
            done = subprocess.run(probe + [os.path.join(directory, 'train.csv'), os.path.join(directory, 'test.csv'), target, task,
                                           str(args.budget), str(seed), '5'], capture_output=True, text=True)
            line = next((l for l in reversed(done.stdout.splitlines()) if l.startswith('{')), None)
            if line is None:
                print('probe failed on %s seed %d: %s' % (name, seed, (done.stdout + done.stderr)[-300:]), file=sys.stderr)
                continue
            result = json.loads(line)
            if result.get('forward_error'):
                print('forward arm failed on %s seed %d: %s' % (name, seed, result['forward_error']), file=sys.stderr)
            runs.append(result)
        # A seed whose search left fewer than five distinct members has no k5 (or no forward arm): it is dropped from every arm, and said so.
        complete = [r for r in runs if all(fn(r) is not None for _, fn in ARMS)]
        if len(complete) < len(runs):
            print('%s: %d of %d seeds dropped (fewer than five distinct members)' % (name, len(runs) - len(complete), len(runs)), file=sys.stderr)
        runs = complete
        if not runs:
            print(f'| {name} | {task} | {len(frame)} | — | no complete run |')
            continue
        metric = runs[0]['metric']
        direction = DIRECTION[metric]
        by = {arm: [fn(r) for r in runs] for arm, fn in ARMS}
        cells = ['%.4f±%.4f' % mean_se(by[arm]) for arm, _ in ARMS]
        verdicts = {'k3': verdict(by['k3'], by['single'], direction), 'k5': verdict(by['k5'], by['single'], direction),
                    'oofpick': verdict(by['oofpick'], by['single'], direction), 'forward': verdict(by['forward'], by['single'], direction),
                    'forward vs oofpick': verdict(by['forward'], by['oofpick'], direction), 'forward vs k5': verdict(by['forward'], by['k5'], direction)}
        for key, v in verdicts.items():
            summary[key][v] += 1
        # What forward selection chose: how many of the five members got any weight, and the largest member's share of the steps.
        chosen_counts = [r['forward']['counts'] for r in runs]
        used = sum(sum(1 for c in counts if c > 0) for counts in chosen_counts) / len(chosen_counts)
        top_share = sum(max(counts) / sum(counts) for counts in chosen_counts) / len(chosen_counts)
        print(f'| {name} | {task} | {len(frame)} | {metric} | ' + ' | '.join(cells) + ' | ' + ' | '.join(verdicts[k] for k in ('k3', 'k5', 'oofpick', 'forward', 'forward vs oofpick', 'forward vs k5'))
              + f' | {used:.1f} | {top_share:.2f} |  (n={len(runs)})')
    print()
    for key, tally in summary.items():
        print(f'{key}{"" if " vs " in key else " vs single"}: {tally}')


if __name__ == '__main__':
    main()
