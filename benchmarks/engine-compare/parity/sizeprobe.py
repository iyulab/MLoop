#!/usr/bin/env python3
"""What a large LightGBM model costs to keep, in ML.NET: file size, time to save, load and score, by model size
(trees x leaves) — the quantity MLoop's search used to cap when it picked the smallest model within a standard error.

  python sizeprobe.py --mlnet "dotnet bin/MlnetLgbm.dll"
"""
import argparse, json, os, shlex, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run as R

# (trees, leaves): ~0.1 M, ~1 M, ~3 M, ~7 M trees x leaves
SHAPES = [(300, 330), (1000, 1000), (1500, 2000), (2500, 2800)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mlnet', required=True)
    parser.add_argument('--table', default='houses')
    args = parser.parse_args()
    entry = {e['name']: e for e in json.load(open(os.path.join(os.path.dirname(HERE), 'datasets.json')))}[args.table]
    work = os.path.join(tempfile.gettempdir(), 'sizeprobe')
    train, test = R.split(entry, R.load_table(entry, os.path.join(tempfile.gettempdir(), 'engine-compare-cache')), 0, work)
    print(f'table {args.table}: {len(train)} train rows')
    print('| trees x leaves | size | model file | save | load | score test rows | test rmse |')
    print('|---|---|---|---|---|---|---|')
    for trees, leaves in SHAPES:
        config = json.dumps({'leaves': leaves, 'trees': trees, 'minChild': 5, 'lr': 0.05, 'bins': 255, 'colsample': 1.0, 'l1': 0.0, 'l2': 1.0})
        done = subprocess.run(shlex.split(args.mlnet) + [os.path.join(work, 'train.csv'), os.path.join(work, 'test.csv'), entry['target'], config],
                              capture_output=True, text=True)
        try:
            r = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(f'| {trees} x {leaves} | failed: {(done.stdout + done.stderr)[-200:]} |')
            continue
        print(f"| {trees} x {leaves} | {trees * leaves / 1e6:.2f} M | {r['modelBytes'] / 1e6:.1f} MB | {r['saveSeconds']:.1f} s | "
              f"{r['loadSeconds']:.1f} s | {r['scoreSeconds']:.2f} s ({r['scoredRows']}) | {r['rmse']:.4f} |", flush=True)


if __name__ == '__main__':
    main()
