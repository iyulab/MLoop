#!/usr/bin/env python3
"""Can ML.NET's LightGBM reproduce what another search found, and what costs it?

Replays recorded LightGBM settings through ML.NET's LightGBM (MlnetLgbm) and the LightGBM library on the same split files the
engine-compare harness scored them on:
  - `flaml-lgbm-configs.json`: the settings FLAML's LightGBM chose per (table, split seed), with the held-out RMSE FLAML reported;
  - `mloop-lgbm-best.json`: the settings MLoop's own search ranked best among its LightGBM trials, with the held-out RMSE of its run.
Variants of each: as recorded; the L2 penalty cut to MLoop's search-space cap (1); and, for MLoop's settings, bagging on
(the subsample fraction applied every iteration) and the subsample fraction set to 1 — ML.NET's default bagging frequency
is 0, which leaves a subsample fraction without effect.

  python replay.py --mlnet "dotnet bin/Release/net10.0/MlnetLgbm.dll" --out replay.jsonl
"""
import argparse, json, math, os, shlex, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run as R


def numeric_files(entry, df, seed, work):
    train, test = R.split(entry, df, seed, work)
    target = entry['target']
    for column in [c for c in train.columns if c != target and not _is_number(train[c])]:
        cats = sorted(set(train[column].astype(str)) | set(test[column].astype(str)))
        train[column] = train[column].astype(str).map({c: i for i, c in enumerate(cats)})
        test[column] = test[column].astype(str).map({c: i for i, c in enumerate(cats)})
    tr, te = os.path.join(work, 'numeric-train.csv'), os.path.join(work, 'numeric-test.csv')
    train.to_csv(tr, index=False)
    test.to_csv(te, index=False)
    return train, test, tr, te


def _is_number(series):
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def from_flaml(c):
    return {'leaves': c['num_leaves'], 'trees': c['n_estimators'], 'minChild': c['min_child_samples'], 'lr': c['learning_rate'],
            'bins': (1 << int(c['log_max_bin'])) - 1, 'colsample': c['colsample_bytree'], 'l1': c['reg_alpha'], 'l2': c['reg_lambda']}


def from_mloop(p):
    return {'leaves': int(p['NumberOfLeaves']), 'trees': int(p['NumberOfTrees']), 'minChild': int(p['MinimumExampleCountPerLeaf']),
            'lr': float(p['LearningRate']), 'bins': int(p['MaximumBinCountPerFeature']), 'colsample': float(p['FeatureFraction']),
            'l1': float(p['L1Regularization']), 'l2': float(p['L2Regularization']), 'subsample': float(p['SubsampleFraction'])}


def mlnet_rmse(mlnet, cfg, train_csv, test_csv, label):
    done = subprocess.run(shlex.split(mlnet) + [train_csv, test_csv, label, json.dumps(cfg)], capture_output=True, text=True)
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])['rmse']
    except (ValueError, IndexError, KeyError):
        sys.stderr.write(done.stderr[-300:] + '\n')
        return None


def library_rmse(cfg, xtr, ytr, xte, yte):
    import numpy as np
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(num_leaves=cfg['leaves'], n_estimators=cfg['trees'], min_child_samples=cfg['minChild'], learning_rate=cfg['lr'],
                      max_bin=cfg['bins'], colsample_bytree=cfg['colsample'], reg_alpha=cfg['l1'], reg_lambda=cfg['l2'],
                      random_state=0, deterministic=True, verbose=-1)
    m.fit(xtr, ytr)
    return float(np.sqrt(np.mean((m.predict(xte) - yte) ** 2)))


def variants(kind, cfg):
    yield 'as-recorded', cfg
    yield 'l2-capped-at-1', {**cfg, 'l2': min(cfg['l2'], 1.0)}
    if kind == 'mloop':
        yield 'bagging-on', {**cfg, 'subsampleFreq': 1}
        yield 'subsample-1', {**cfg, 'subsample': 1.0}
        yield 'seed-1', {**cfg, 'seed': 1}
        # the product fits its final model on 95 % of the train file (5 % is held back for its own test) and seeds with 1
        yield 'train-95-seed-1', {**cfg, 'seed': 1, '_train_fraction': 0.95}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mlnet', required=True)
    parser.add_argument('--out', default='replay.jsonl')
    args = parser.parse_args()
    entries = {e['name']: e for e in json.load(open(os.path.join(os.path.dirname(HERE), 'datasets.json')))}
    inputs = os.path.join(HERE, 'replay-input')
    work_items = ([('flaml', r['dataset'], r['seed'], from_flaml(r['config']), r['test_rmse']) for r in json.load(open(os.path.join(inputs, 'flaml-lgbm-configs.json')))]
                  + [('mloop', r['dataset'], r['seed'], from_mloop(r['params']), r['test_rmse']) for r in json.load(open(os.path.join(inputs, 'mloop-lgbm-best.json')))])
    cache = os.path.join(tempfile.gettempdir(), 'engine-compare-cache')
    tables = {}
    rows = []
    with open(args.out, 'w') as out:
        for kind, dataset, seed, cfg, reported in work_items:
            entry = entries[dataset]
            if dataset not in tables:
                tables[dataset] = R.load_table(entry, cache)
            work = os.path.join(tempfile.gettempdir(), 'replay', dataset, str(seed))
            train, test, tr, te = numeric_files(entry, tables[dataset], seed, work)
            target = entry['target']
            xtr, xte = train.drop(columns=[target]), test.drop(columns=[target])
            for name, variant in variants(kind, cfg):
                variant = dict(variant)
                fraction = variant.pop('_train_fraction', None)
                run_tr = tr
                if fraction:
                    subset = train.sample(frac=fraction, random_state=seed)
                    run_tr = os.path.join(work, 'numeric-train-subset.csv')
                    subset.to_csv(run_tr, index=False)
                record = {'source': kind, 'dataset': dataset, 'seed': seed, 'variant': name, 'reported_rmse': reported,
                          'mlnet': mlnet_rmse(args.mlnet, variant, run_tr, te, target),
                          'library': library_rmse(variant, xtr, train[target].values, xte, test[target].values) if name in ('as-recorded', 'l2-capped-at-1') else None}
                rows.append(record)
                out.write(json.dumps(record) + '\n')
                out.flush()
                print(kind, dataset, seed, name, record['reported_rmse'], record['mlnet'], record['library'], flush=True)
    print('\n| source | dataset | variant | reported | ML.NET | library | seeds |\n|---|---|---|---|---|---|---|')
    keyed = {}
    for r in rows:
        keyed.setdefault((r['source'], r['dataset'], r['variant']), []).append(r)
    for (kind, dataset, name), rs in sorted(keyed.items()):
        mean = lambda key: statistics.fmean(r[key] for r in rs if r[key] is not None) if any(r[key] is not None for r in rs) else float('nan')
        print(f"| {kind} | {dataset} | {name} | {mean('reported_rmse'):.4f} | {mean('mlnet'):.4f} | {mean('library'):.4f} | {len(rs)} |")


if __name__ == '__main__':
    main()
