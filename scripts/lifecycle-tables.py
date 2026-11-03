#!/usr/bin/env python3
"""Cut public tables into the two regimes `lifecycle-check.py --tables` walks a model through.

Real data has no "later regime", so one is made from the table itself: the numeric column that tracks the label best
(by rank correlation) is the one that "moves". Rows at or below its 60th percentile are regime A (the training
regime); rows above it are regime B, a part of the population the first model has barely seen, and there the label
is rewritten (a class becomes the next class, a regression target is mirrored about regime A's mean) — the rule the
world followed in A no longer holds in B. The input shift is real; the concept shift is injected.

Each table becomes a directory: train_a.csv, new_a.csv (regime A, in-regime rows to predict), train_b.csv, test_b.csv,
new_b.csv (regime B), and manifest.json. Needs pandas and numpy.

  python lifecycle-tables.py --source <openml-benchmarks dir> --out <dir> [--datasets credit-g,houses] [--per-task 4] [--shape korean-export]
  python lifecycle-tables.py --synthetic wide,many-classes,spelling,duplicate-name --out <dir>

--synthetic needs no source: it writes tables of the shapes real exports have and a small public table does not — a hundred and
fifty columns, a label with a hundred classes, one class spelled two ways (3 and 3.00), two columns of one name. The last is
refused by training, and the manifest says so (expect: refused): the walk asserts the refusal and stops there.

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


def hangul(text):
    """The same text spelled in Hangul syllables, one syllable for each printable ASCII character (injective, so two names never meet)."""
    return ''.join(chr(0xAC00 + (ord(c) - 32) * 37) if 32 < ord(c) < 127 and c not in '_-.' else c for c in str(text))


def export_shape(part, label, task, shift_col, numeric):
    """A part of a table as the usual Korean spreadsheet export writes it: Hangul column names and words (a table that is read as
    ASCII everywhere hides what a name outside ASCII does), one header cell broken over two lines. Returns the part and the
    names the commands will read (a broken cell reads as one name with a space). The file is written in CP949 with the dataframe's
    index in front by write_part."""
    renamed = {c: hangul(c) for c in part.columns}
    breakable = [c for c in numeric if c not in (shift_col, label) and len(renamed[c]) >= 2]
    broken = breakable[0] if breakable else None
    out = part.copy()
    for c in out.columns:
        if c != label or task != 'regression':
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda v: v if v != v else hangul(v))
    cells = {c: (renamed[c][:len(renamed[c]) // 2] + '\n' + renamed[c][len(renamed[c]) // 2:] if c == broken else renamed[c]) for c in part.columns}
    read = {c: renamed[c] if c != broken else renamed[c][:len(renamed[c]) // 2] + ' ' + renamed[c][len(renamed[c]) // 2:] for c in part.columns}
    return out.rename(columns=cells), read


def write_part(part, path, shape):
    if shape == 'korean-export':
        part.to_csv(path, index=True, encoding='cp949')   # the index has no name: the header starts with an empty cell
    else:
        part.to_csv(path, index=False)


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


def cut(row, source, out, shape='plain'):
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
    if len(b) < 400 or len(a) < 1500:
        return f'{name}: skipped (regimes of {len(a)} and {len(b)} rows)'

    if task == 'regression':
        centre = pd.to_numeric(a[label], errors='coerce').mean()
        b = b.assign(**{label: 2 * centre - pd.to_numeric(b[label], errors='coerce')})
    else:
        classes = sorted(df[label].astype(str).unique())
        nxt = {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
        b = b.assign(**{label: b[label].astype(str).map(nxt)})
        a = a.assign(**{label: a[label].astype(str)})

    # the drift check needs 300 logged predictions (fewer read chance as drift), so the in-regime rows to predict are that many
    n_new_a, n_new_b = 300, min(300, len(b) // 5)
    n_test_b = max(60, len(b) // 5)
    parts = {'new_a': a.iloc[:n_new_a], 'train_a': a.iloc[n_new_a:],
             'new_b': b.iloc[:n_new_b], 'test_b': b.iloc[n_new_b:n_new_b + n_test_b], 'train_b': b.iloc[n_new_b + n_test_b:]}
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    read = {c: c for c in df.columns}
    if shape == 'korean-export':
        shaped = {}
        for key, part in parts.items():
            shaped[key], read = export_shape(part, label, task, col, numeric)
        parts = shaped
    for key, part in parts.items():
        write_part(part, os.path.join(d, key + '.csv'), shape)
    k = df[label].nunique()
    learned = {'binary': ['auc', 0.6], 'multiclass': ['macro_accuracy', round(1 / k + 0.05, 3)], 'regression': ['r_squared', 0.1]}[task]
    manifest = {'name': name, 'task': task, 'label': read[label], 'shift_column': read[col], 'numeric_columns': [read[c] for c in numeric],
                'n_new_a': n_new_a, 'n_new_b': n_new_b, 'learned': learned, 'regime_cut': float(at),
                'rows': {key: len(part) for key, part in parts.items()}}
    if shape == 'korean-export':
        manifest['encoding'] = 'cp949'   # lifecycle-check reads the files in this encoding, drops the unnamed index column, and reads a broken header cell with a space
    with open(os.path.join(d, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return f'{name}: {task} label={label} moves={col} rows={manifest["rows"]}'


SYNTHETIC = ('wide', 'many-classes', 'spelling', 'duplicate-name')


def synthetic_table(kind, regime_b, n, rng):
    """n rows of a synthetic table: a label that follows the moving column f0 (and a few more) as it sits in regime A. In regime B f0
    is shifted by 3, which reads as another answer to a model that learned A."""
    width = {'wide': 150, 'many-classes': 12, 'spelling': 8, 'duplicate-name': 6}[kind]
    x = rng.normal(0, 1, size=(n, width))
    if regime_b:
        x[:, 0] += 3.0
    z = 1.5 * (x[:, 0] - (3.0 if regime_b else 0.0)) - x[:, 1] + 0.5 * x[:, 2] + rng.normal(0, 0.6, size=n)
    if kind == 'many-classes':
        # 100 classes a tree can learn: the tenth of f0 (as it sits in regime A) times the tenth of f1 — a label cut finer from one noisy
        # score is not a signal, and a table the first model learns nothing from is refused by the promotion gate and ends the walk
        edges = np.quantile(np.random.default_rng(1).normal(size=200000), np.linspace(0, 1, 11)[1:-1])
        y = pd.Series([f'class_{10 * i + j:03d}' for i, j in zip(np.searchsorted(edges, x[:, 0] - (3.0 if regime_b else 0.0)), np.searchsorted(edges, x[:, 1]))])
    elif kind == 'spelling':
        k = np.digitize(z, [-1.0, 1.0])
        # the same three numbers, spelled two ways: 0 / 0.00, 1 / 1.00, 2 / 2.00 — one class each
        y = pd.Series([f'{v}.00' if rng.random() < 0.4 else str(v) for v in k])
    else:
        y = pd.Series((z > 0).astype(int))
    cols = [f'f{i}' for i in range(width)]
    if kind == 'duplicate-name':
        cols[-1] = cols[-2]
    df = pd.DataFrame(x.round(4), columns=cols)
    df['y'] = y.values
    return df


def synthetic(kind, out):
    rng = np.random.default_rng(SEED)
    sizes = {'wide': (3000, 1000, 300), 'many-classes': (12000, 2400, 300), 'spelling': (3000, 1000, 300), 'duplicate-name': (1500, 400, 300)}[kind]
    n_a, n_b, n_new = sizes
    n_test = 400 if kind == 'many-classes' else 300   # the paired comparison of the walk needs rows enough to tell two models apart
    task = 'binary' if kind in ('wide', 'duplicate-name') else 'multiclass'
    a, b = synthetic_table(kind, False, n_a + n_new, rng), synthetic_table(kind, True, n_b + n_new + n_test, rng)
    classes = sorted(a['y'].astype(str).unique())
    if task == 'multiclass':
        # concept shift in B, as the public tables get: a class becomes the next class
        nxt = {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
        b['y'] = b['y'].astype(str).map(lambda c: nxt.get(c, c))
    else:
        b['y'] = 1 - b['y']
    parts = {'new_a': a.iloc[:n_new], 'train_a': a.iloc[n_new:], 'new_b': b.iloc[:n_new], 'test_b': b.iloc[n_new:n_new + n_test], 'train_b': b.iloc[n_new + n_test:]}
    d = os.path.join(out, 'synthetic-' + kind)
    os.makedirs(d, exist_ok=True)
    for key, part in parts.items():
        write_part(part, os.path.join(d, key + '.csv'), 'plain')
    k = len({str(float(c)) for c in classes}) if kind == 'spelling' else len(classes)
    learned = {'binary': ['auc', 0.6], 'multiclass': ['macro_accuracy', round(1 / k + 0.05, 3)]}[task]
    manifest = {'name': 'synthetic-' + kind, 'task': task, 'label': 'y', 'shift_column': 'f0',
                'numeric_columns': [c for c in a.columns if c != 'y' and c != a.columns[-2]] if kind == 'duplicate-name' else [c for c in a.columns if c != 'y'],
                'n_new_a': n_new, 'n_new_b': n_new, 'learned': learned, 'classes': k, 'rows': {key: len(part) for key, part in parts.items()}}
    if kind == 'duplicate-name':
        manifest['expect'] = 'refused'
        manifest['refusal_mentions'] = a.columns[-1]
    with open(os.path.join(d, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return f'synthetic-{kind}: {task} rows={manifest["rows"]} classes={k}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='')
    ap.add_argument('--synthetic', default='', help=f'comma-separated shapes made without a source: {", ".join(SYNTHETIC)}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', default='')
    ap.add_argument('--per-task', type=int, default=4)
    ap.add_argument('--shape', choices=['plain', 'korean-export'], default='plain',
                    help='korean-export writes the files as a Korean spreadsheet export does: CP949, Hangul names and words, the dataframe index in front, one header cell broken over two lines')
    args = ap.parse_args()
    if args.synthetic:
        os.makedirs(args.out, exist_ok=True)
        for kind in args.synthetic.split(','):
            if kind not in SYNTHETIC:
                sys.exit(f'unknown shape {kind!r}: {", ".join(SYNTHETIC)}')
            print(synthetic(kind, args.out), flush=True)
        return
    if not args.source:
        sys.exit('--source or --synthetic is required')
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
        print(cut(r, args.source, args.out, args.shape), flush=True)


if __name__ == '__main__':
    main()
