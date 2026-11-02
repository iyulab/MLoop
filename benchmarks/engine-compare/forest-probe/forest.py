#!/usr/bin/env python3
"""Is a gap to a random forest a matter of the forest or of the settings MLoop's search can reach?

On numeric tables where FLAML's winner is a random forest or extra trees, each split seed is scored with:
  - scikit-learn RandomForest and ExtraTrees (300 trees, library defaults), and
  - ML.NET FastForest at several fixed configurations, in two groups: `reach-*` use only what MLoop's search can set
    (trees, leaves, the per-tree feature fraction; the rest stays at ML.NET's defaults), `open-*` also set what the
    search space does not expose (minimum examples per leaf, the bagged share of the rows, the per-split feature
    fraction, the bin count).
If the `open-*` rows reach the forest and the `reach-*` rows do not, the missing piece is in the search space; if neither
does, it is the learner.

  python forest.py --mlnet "dotnet forest-probe/bin/Release/net10.0/MlnetForest.dll" --seeds 0,1,2,3,4 --out forest.jsonl
"""
import argparse, json, math, os, shlex, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run as R

TABLES = ['qsar-biodeg', 'wine_quality']
# name -> trees, leaves, minLeaf, bagFraction, treeFeatures, splitFeatures, bins
CONFIGS = {
    'reach-default': (100, 20, 10, 0.7, 0.7, 0.7, 255),
    'reach-300x1000': (300, 1000, 10, 0.7, 1.0, 0.7, 255),
    'reach-300x1000-ff5': (300, 1000, 10, 0.7, 0.5, 0.7, 255),
    'open-leaf2': (300, 1000, 2, 0.7, 1.0, 0.7, 255),
    'open-leaf1-bag1': (300, 1000, 1, 1.0, 1.0, 0.7, 255),
    'open-leaf1-bag1-split33': (300, 1000, 1, 1.0, 1.0, 0.33, 255),
    'open-leaf1-bag1-split20': (300, 1000, 1, 1.0, 1.0, 0.2, 255),
    'open-leaf1-bag1-split15': (300, 1000, 1, 1.0, 1.0, 0.15, 255),
    'open-leaf2-bag1-split15': (300, 1000, 2, 1.0, 1.0, 0.15, 255),
    'open-leaf1-bag07-split15': (300, 1000, 1, 0.7, 1.0, 0.15, 255),
    'open-leaf1-bag1-split50-bins1024': (300, 1000, 1, 1.0, 1.0, 0.5, 1024),
}


def mlnet_score(mlnet, task, cfg, train_csv, test_csv, label):
    trees, leaves, min_leaf, bag, tree_ff, split_ff, bins = cfg
    config = json.dumps({'trees': trees, 'leaves': leaves, 'minLeaf': min_leaf, 'bagFraction': bag,
                         'treeFeatures': tree_ff, 'splitFeatures': split_ff, 'bins': bins})
    done = subprocess.run(shlex.split(mlnet) + [train_csv, test_csv, label, task, config], capture_output=True, text=True)
    try:
        out = json.loads(done.stdout.strip().splitlines()[-1])
        return out['auc'] if task == 'binary' else out['rmse']
    except (ValueError, IndexError, KeyError):
        sys.stderr.write(done.stderr[-400:] + '\n')
        return None


def _is_number(series):
    import pandas as pd
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def sklearn_scores(task, xtr, ytr, xte, yte):
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, binary_cls, regress_cls in (('sk-rf', RandomForestClassifier, RandomForestRegressor), ('sk-et', ExtraTreesClassifier, ExtraTreesRegressor)):
        if task == 'binary':
            model = binary_cls(n_estimators=300, random_state=0, n_jobs=-1).fit(xtr, ytr)
            out[name] = float(roc_auc_score(yte, model.predict_proba(xte)[:, 1]))
        else:
            model = regress_cls(n_estimators=300, random_state=0, n_jobs=-1).fit(xtr, ytr)
            out[name] = float(np.sqrt(np.mean((model.predict(xte) - yte) ** 2)))
    return out


def summarize(path):
    rows = [json.loads(line) for line in open(path)]
    by = {}
    for r in rows:
        by.setdefault((r['dataset'], r['system']), {})[r['seed']] = r['score']
    for dataset in sorted({d for d, _ in by}):
        task = 'binary' if dataset == 'qsar-biodeg' else 'regression'
        higher = task == 'binary'
        print(f'\n{dataset} ({"AUC, higher is better" if higher else "RMSE, lower is better"})')
        print('| system | mean | se | vs sk-rf (paired, ± se) |')
        print('|---|---|---|---|')
        base = by.get((dataset, 'sk-rf'), {})
        for (d, system), values in sorted(by.items()):
            if d != dataset:
                continue
            xs = [v for v in values.values() if v is not None]
            if not xs:
                continue
            se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float('nan')
            diffs = [values[s] - base[s] for s in values if s in base and values[s] is not None]
            paired = (f'{statistics.fmean(diffs):+.4f} ± {statistics.stdev(diffs) / math.sqrt(len(diffs)):.4f}' if len(diffs) > 1 else '')
            print(f'| {system} | {statistics.fmean(xs):.4f} | {se:.4f} | {paired} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mlnet', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--out', default='forest.jsonl')
    parser.add_argument('--tables', default=','.join(TABLES))
    args = parser.parse_args()
    entries = {e['name']: e for e in json.load(open(os.path.join(os.path.dirname(HERE), 'datasets.json')))}
    cache = os.path.join(tempfile.gettempdir(), 'engine-compare-cache')
    work_root = os.path.join(tempfile.gettempdir(), 'forest-probe')
    with open(args.out, 'w') as out:
        for name in args.tables.split(','):
            entry = entries[name]
            df = R.load_table(entry, cache)
            for seed in [int(s) for s in args.seeds.split(',')]:
                work = os.path.join(work_root, name, str(seed))
                train, test = R.split(entry, df, seed, work)
                target, task = entry['target'], entry['task']
                # every column numeric; a binary label as 0/1 (the first class in sorted order is 0, as the harness encodes it)
                if task == 'binary':
                    ytr, yte = R.encode_classes(train, test, target)
                    train, test = train.assign(**{target: ytr.values}), test.assign(**{target: yte.values})
                else:
                    ytr, yte = train[target], test[target]
                for column in [c for c in train.columns if c != target and not _is_number(train[c])]:
                    # a text column becomes its category number (the same numbering on both files)
                    cats = sorted(set(train[column].astype(str)) | set(test[column].astype(str)))
                    train[column] = train[column].astype(str).map({c: i for i, c in enumerate(cats)})
                    test[column] = test[column].astype(str).map({c: i for i, c in enumerate(cats)})
                tr_csv, te_csv = os.path.join(work, 'numeric-train.csv'), os.path.join(work, 'numeric-test.csv')
                train.to_csv(tr_csv, index=False)
                test.to_csv(te_csv, index=False)
                xtr, xte = train.drop(columns=[target]), test.drop(columns=[target])
                scores = sklearn_scores(task, xtr, ytr.values, xte, yte.values)
                for cname, cfg in CONFIGS.items():
                    scores[cname] = mlnet_score(args.mlnet, task, cfg, tr_csv, te_csv, target)
                for system, score in scores.items():
                    out.write(json.dumps({'dataset': name, 'seed': seed, 'system': system, 'score': score}) + '\n')
                    out.flush()
                print(name, seed, {k: (round(v, 4) if v is not None else None) for k, v in scores.items()}, flush=True)
    summarize(args.out)


if __name__ == '__main__':
    main()
