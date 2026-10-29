#!/usr/bin/env python3
"""Cut public tables into the two regimes `lifecycle-check.py --tables` walks a model through.

Real data has no "later regime", so one is made from the table itself: the numeric column that tracks the label best
(by rank correlation) is the one that "moves". Rows at or below its 60th percentile are regime A (the training
regime); rows above it are regime B, a part of the population the first model has barely seen, and there the label
is rewritten (a class becomes the next class, a regression target is mirrored about regime A's mean) — the rule the
world followed in A no longer holds in B. The input shift is real; the concept shift is injected.

Each table becomes a directory: train_a.csv, new_a.csv (regime A, in-regime rows to predict), train_b.csv, test_b.csv,
new_b.csv (regime B), and manifest.json. Needs pandas and numpy.

  python lifecycle-tables.py --source <openml-benchmarks dir> --out <dir> [--datasets credit-g,houses] [--per-task 4]

--source is a folder holding MANIFEST.csv and data/<id>_<name>.csv (OpenML suites as downloaded by the benchmark
fetcher). Without --datasets, each task kind contributes the first --per-task tables (by name) of 1,500 to 50,000
rows, at most 100 columns and at least three numeric features.
"""
import argparse, csv, json, os, sys

import numpy as np
import pandas as pd

SEED = 20261030
REGIME_CAP = 6000   # rows per regime; the time a lifecycle takes should not turn on the size of the table
SHIFT_QUANTILE = 0.6


def load_manifest(source):
    with open(os.path.join(source, 'MANIFEST.csv'), newline='') as f:
        return list(csv.DictReader(f))


def pick(rows, per_task):
    chosen, seen, names = [], {}, set()
    for r in sorted(rows, key=lambda r: (r['name'], int(r['did']))):
        kind = r['task_type']
        try:
            n, width = float(r['rows']), float(r['cols'])
        except ValueError:
            continue
        # one table per name (a table sits in several suites), of a size a lifecycle can retrain in a minute and narrow enough to read like a business table
        if kind not in ('binary', 'multiclass', 'regression') or not 1500 <= n <= 50000 or width > 100 or r.get('deferred') or r['name'] in names:
            continue
        names.add(r['name'])
        if seen.get(kind, 0) < per_task:
            seen[kind] = seen.get(kind, 0) + 1
            chosen.append(r)
    return chosen


def shift_columns(df, label, task):
    """The numeric columns that could move, the one tracking the label best first."""
    y = df[label]
    codes = pd.Series(pd.factorize(y)[0], index=df.index) if task != 'regression' else pd.to_numeric(y, errors='coerce')
    scored = []
    for c in df.columns:
        if c == label or not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() < 10:
            continue
        # a counter or an identifier (whole numbers, nearly all different) says nothing about when a row was recorded
        if (df[c].dropna() % 1 == 0).all() and df[c].nunique() > 0.3 * len(df):
            continue
        score = abs(df[c].rank().corr(codes.rank()))  # rank correlation, without scipy
        if score == score:
            scored.append((score, c))
    return [c for _, c in sorted(scored, reverse=True)]


def cut_at(df, c):
    """The 60th percentile of column c, or None when regime A would pile up on its own maximum (a capped column: a shift past
    the cap lands in the same top bin as the rows sitting on it, so the rows that moved cannot be told from the ones that did not)."""
    at = df[c].quantile(SHIFT_QUANTILE)
    a = df[c][df[c] <= at]
    return None if len(a) == 0 or (a == a.max()).mean() > 0.2 else at


def cut(row, source, out):
    name, task, label = row['name'], row['task_type'], row['target']
    df = pd.read_csv(os.path.join(source, 'data', row['file']), low_memory=False)
    df = df.dropna(subset=[label]).reset_index(drop=True)
    numeric = [c for c in df.columns if c != label and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 3:
        return f'{name}: skipped (fewer than 3 numeric features)'
    col, at = next(((c, at) for c in shift_columns(df, label, task) if (at := cut_at(df, c)) is not None), (None, None))
    if col is None:
        return f'{name}: skipped (no numeric column to move)'
    in_b = df[col] > at
    a = df[~in_b].sample(frac=1.0, random_state=SEED).head(REGIME_CAP)
    b = df[in_b].sample(frac=1.0, random_state=SEED).head(REGIME_CAP)
    if len(b) < 400 or len(a) < 700:
        return f'{name}: skipped (regimes of {len(a)} and {len(b)} rows)'

    if task == 'regression':
        centre = pd.to_numeric(a[label], errors='coerce').mean()
        b = b.assign(**{label: 2 * centre - pd.to_numeric(b[label], errors='coerce')})
    else:
        classes = sorted(df[label].astype(str).unique())
        nxt = {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
        b = b.assign(**{label: b[label].astype(str).map(nxt)})
        a = a.assign(**{label: a[label].astype(str)})

    # the drift check needs 100 logged predictions, so the in-regime rows to predict are at least that many
    n_new_a, n_new_b = min(300, len(a) // 5), min(300, len(b) // 5)
    n_test_b = max(60, len(b) // 5)
    parts = {'new_a': a.iloc[:n_new_a], 'train_a': a.iloc[n_new_a:],
             'new_b': b.iloc[:n_new_b], 'test_b': b.iloc[n_new_b:n_new_b + n_test_b], 'train_b': b.iloc[n_new_b + n_test_b:]}
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for key, part in parts.items():
        part.to_csv(os.path.join(d, key + '.csv'), index=False)
    k = df[label].nunique()
    learned = {'binary': ['auc', 0.6], 'multiclass': ['macro_accuracy', round(1 / k + 0.05, 3)], 'regression': ['r_squared', 0.1]}[task]
    manifest = {'name': name, 'task': task, 'label': label, 'shift_column': col, 'numeric_columns': numeric,
                'n_new_a': n_new_a, 'n_new_b': n_new_b, 'learned': learned, 'regime_cut': float(at),
                'rows': {key: len(part) for key, part in parts.items()}}
    with open(os.path.join(d, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return f'{name}: {task} label={label} moves={col} rows={manifest["rows"]}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', default='')
    ap.add_argument('--per-task', type=int, default=4)
    args = ap.parse_args()
    rows = load_manifest(args.source)
    if args.datasets:
        wanted = set(args.datasets.split(','))
        chosen = [r for r in rows if r['name'] in wanted]
        missing = wanted - {r['name'] for r in chosen}
        if missing:
            sys.exit(f'not in the manifest: {sorted(missing)}')
    else:
        chosen = pick(rows, args.per_task)
    os.makedirs(args.out, exist_ok=True)
    for r in chosen:
        print(cut(r, args.source, args.out), flush=True)


if __name__ == '__main__':
    main()
