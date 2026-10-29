#!/usr/bin/env python3
"""Does ML.NET's LightGBM give what the LightGBM library gives for the same settings?

For each numeric regression table and split seed, one fixed set of hyperparameters is trained twice — with the LightGBM
library (scikit-learn API) and with ML.NET's LightGbm regressor — and both are scored on the same held-out rows. If the
two agree within the split-to-split spread, the learner is the same and a gap between systems is a matter of search; if
ML.NET is consistently worse at identical settings, the gap is in the learner (a setting the two read differently).

  python parity.py --mlnet "dotnet bin/MlnetLgbm.dll" --seeds 0,1,2,3,4 --out parity.jsonl
"""
import argparse, json, os, shlex, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run as R

TABLES = ['airfoil_self_noise', 'concrete_compressive_strength', 'wine_quality', 'houses']
# name -> (leaves, trees, min child samples, learning rate, max bin, colsample, L1, L2)
CONFIGS = {
    'defaults': (31, 100, 20, 0.1, 255, 1.0, 0.0, 0.0),
    'small-slow': (15, 600, 20, 0.03, 255, 1.0, 0.0, 0.0),
    'big-leaves': (206, 321, 10, 0.1213, 127, 0.9695, 0.000977, 1.0054),
    'wide-leaves': (699, 188, 19, 0.1085, 127, 0.8994, 0.000977, 5.563),
    'many-trees': (63, 1117, 40, 0.1433, 511, 0.6845, 0.00264, 901.28),
}


def python_rmse(cfg, xtr, ytr, xte, yte):
    import numpy as np
    from lightgbm import LGBMRegressor
    leaves, trees, child, lr, bins, col, l1, l2 = cfg
    m = LGBMRegressor(num_leaves=leaves, n_estimators=trees, min_child_samples=child, learning_rate=lr, max_bin=bins,
                      colsample_bytree=col, reg_alpha=l1, reg_lambda=l2, random_state=0, deterministic=True, verbose=-1)
    m.fit(xtr, ytr)
    return float(np.sqrt(np.mean((m.predict(xte) - yte) ** 2)))


def mlnet_rmse(mlnet, cfg, train_csv, test_csv, label):
    leaves, trees, child, lr, bins, col, l1, l2 = cfg
    config = json.dumps({'leaves': leaves, 'trees': trees, 'minChild': child, 'lr': lr, 'bins': bins, 'colsample': col, 'l1': l1, 'l2': l2})
    done = subprocess.run(shlex.split(mlnet) + [train_csv, test_csv, label, config], capture_output=True, text=True)
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])['rmse']
    except (ValueError, IndexError, KeyError):
        return None


def summarize(path):
    import collections, math, statistics
    rows = [json.loads(line) for line in open(path)]
    groups = collections.defaultdict(list)
    for r in rows:
        if r['mlnet'] is not None and r['lightgbm'] is not None:
            groups[(r['dataset'], r['config'])].append((r['lightgbm'], r['mlnet']))
    print('| dataset | config | LightGBM rmse | ML.NET rmse | ML.NET relative to LightGBM (paired, ± se) | seeds |')
    print('|---|---|---|---|---|---|')
    for (dataset, config), pairs in sorted(groups.items()):
        rel = [(m - l) / l for l, m in pairs]
        se = statistics.stdev(rel) / math.sqrt(len(rel)) if len(rel) > 1 else float('nan')
        print(f'| {dataset} | {config} | {statistics.fmean(l for l, _ in pairs):.4f} | {statistics.fmean(m for _, m in pairs):.4f} '
              f'| {statistics.fmean(rel) * 100:+.1f} % ± {se * 100:.1f} | {len(pairs)} |')
    failed = sum(1 for r in rows if r['mlnet'] is None)
    print(f'\nML.NET fits that returned nothing: {failed} of {len(rows)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mlnet', required=True)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--out', default='parity.jsonl')
    args = parser.parse_args()
    entries = {e['name']: e for e in json.load(open(os.path.join(os.path.dirname(HERE), 'datasets.json')))}
    cache = os.path.join(tempfile.gettempdir(), 'engine-compare-cache')
    work_root = os.path.join(tempfile.gettempdir(), 'parity')
    with open(args.out, 'w') as out:
        for name in TABLES:
            entry = entries[name]
            df = R.load_table(entry, cache)
            for seed in [int(s) for s in args.seeds.split(',')]:
                work = os.path.join(work_root, name, str(seed))
                train, test = R.split(entry, df, seed, work)
                target = entry['target']
                xtr, xte = train.drop(columns=[target]), test.drop(columns=[target])
                for cname, cfg in CONFIGS.items():
                    record = {'dataset': name, 'seed': seed, 'config': cname,
                              'lightgbm': python_rmse(cfg, xtr, train[target].values, xte, test[target].values),
                              'mlnet': mlnet_rmse(args.mlnet, cfg, os.path.join(work, 'train.csv'), os.path.join(work, 'test.csv'), target)}
                    out.write(json.dumps(record) + '\n')
                    out.flush()
                    print(name, seed, cname, record['lightgbm'], record['mlnet'], flush=True)
    summarize(args.out)


if __name__ == '__main__':
    main()
