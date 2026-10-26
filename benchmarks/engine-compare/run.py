#!/usr/bin/env python3
"""Compare MLoop's trainer search with FLAML and stand-alone gradient-boosting libraries on public OpenML tables.

Each (dataset, split seed) gets one stratified 80/20 split; every system receives the same train and test files.

  mloop         `mloop train --time B --metric <reported metric> --test-split 0.05` on the train file, then
                `mloop evaluate` on the held-out file (the search optimizes what the table reports and trains on
                nearly all of the train file)
  mloop-defaults  the same without `--metric` / `--test-split`: the shipped defaults (an accuracy-family metric,
                20% of the train file held back for the model's own test)
  flaml         FLAML AutoML, time_budget=B, its default learners
  flaml-nogbm   FLAML restricted to rf / extra_tree / a linear model (lrl1, lassolars for regression); no gradient boosting
  lightgbm      LightGBM, library defaults
  xgboost       XGBoost, library defaults

Metrics: binary ROC AUC, multiclass log-loss, regression RMSE. Datasets are fetched from OpenML by id
(see datasets.json) and cached; no data is stored in this repository.

  python run.py run --mloop "dotnet ../../tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" --seeds 0,1,2 --budget 60
  python run.py summarize results-*.jsonl
"""
import argparse, collections, json, math, os, shlex, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_FLAG = {'binary': 'binary-classification', 'multiclass': 'multiclass-classification', 'regression': 'regression'}
# metric -> direction (+1: higher is better)
PRIMARY = {'binary': ('auc', +1), 'multiclass': ('log_loss', -1), 'regression': ('rmse', -1)}
SYSTEMS = ['mloop', 'mloop-defaults', 'flaml', 'flaml-nogbm', 'lightgbm', 'xgboost']
# what `mloop train --metric` is given so the search optimizes the metric the table reports
OPTIMIZE = {'binary': 'auc', 'multiclass': 'log_loss', 'regression': 'rmse'}


def load_table(entry, cache):
    from sklearn.datasets import fetch_openml
    # The default (pandas) ARFF parser fails on tables whose categories are numbers; liac-arff reads all 16.
    return fetch_openml(data_id=entry['id'], as_frame=True, parser='liac-arff', data_home=cache).frame


def split(entry, df, seed, work):
    from sklearn.model_selection import train_test_split
    y = df[entry['target']]
    strat = y if entry['task'] != 'regression' and y.value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)
    os.makedirs(work, exist_ok=True)
    train.to_csv(os.path.join(work, 'train.csv'), index=False)
    test.to_csv(os.path.join(work, 'test.csv'), index=False)
    return train, test


def feature_frames(train, test, target):
    import pandas as pd
    xtr, xte = train.drop(columns=[target]).copy(), test.drop(columns=[target]).copy()
    for c in xtr.columns:
        if not pd.api.types.is_numeric_dtype(xtr[c]) or pd.api.types.is_bool_dtype(xtr[c]):
            cats = pd.Categorical(xtr[c].astype(str)).categories
            xtr[c] = pd.Categorical(xtr[c].astype(str), categories=cats)
            xte[c] = pd.Categorical(xte[c].astype(str), categories=cats)
    return xtr, xte


def score(task, y_true, proba=None, pred=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
    if task == 'binary':
        return {'auc': float(roc_auc_score(y_true, proba[:, 1]))}
    if task == 'multiclass':
        return {'log_loss': float(log_loss(y_true, np.clip(proba, 1e-15, 1), labels=list(range(proba.shape[1]))))}
    return {'rmse': float(mean_squared_error(y_true, pred) ** 0.5)}


def encode_classes(train, test, target):
    classes = sorted(train[target].astype(str).unique())
    index = {c: i for i, c in enumerate(classes)}
    return train[target].astype(str).map(index), test[target].astype(str).map(index)


def run_mloop(mloop, entry, work, budget, aligned):
    project = os.path.join(work, 'mloop-project')
    shutil.rmtree(project, ignore_errors=True)
    init = subprocess.run(mloop + ['init', 'mloop-project', '--task', TASK_FLAG[entry['task']]],
                          cwd=work, capture_output=True, text=True)
    if not os.path.isdir(project):
        return {'error': 'init failed: ' + (init.stdout + init.stderr)[-300:]}
    started = time.time()
    train = subprocess.run(mloop + ['train', os.path.join(work, 'train.csv'), '--label', entry['target'],
                                    '--task', TASK_FLAG[entry['task']], '--time', str(budget), '--no-promote', '--json']
                           + (['--metric', OPTIMIZE[entry['task']], '--test-split', '0.05'] if aligned else []),
                           cwd=project, capture_output=True, text=True)
    wall = time.time() - started
    result, trials = None, 0
    for line in train.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('event') == 'result':
            result = event
        elif event.get('event') == 'trial':
            trials += 1
    if result is None:
        return {'error': 'train failed: ' + (train.stdout + train.stderr)[-400:]}
    evaluated = subprocess.run(mloop + ['evaluate', result['experimentId'], os.path.join(work, 'test.csv'), '--json'],
                               cwd=project, capture_output=True, text=True)
    try:
        test_metrics = json.loads(evaluated.stdout)['testMetrics']
    except (ValueError, KeyError):
        return {'error': 'evaluate failed: ' + (evaluated.stdout + evaluated.stderr)[-300:]}
    return {'learner': result.get('bestTrainer'), 'trials': trials, 'wall_s': round(wall, 1), 'test': test_metrics}


def run_flaml(entry, train, test, budget, seed, estimators=None):
    from flaml import AutoML
    target, task = entry['target'], entry['task']
    xtr, xte = feature_frames(train, test, target)
    options = dict(time_budget=budget, verbose=0, seed=seed, n_jobs=-1)
    if estimators:
        options['estimator_list'] = estimators
    automl = AutoML()
    started = time.time()
    if task == 'regression':
        automl.fit(xtr, train[target], task='regression', metric='rmse', **options)
        test_metrics = score(task, test[target].values, pred=automl.predict(xte))
    else:
        ytr, yte = encode_classes(train, test, target)
        automl.fit(xtr, ytr, task='classification', metric='roc_auc' if task == 'binary' else 'log_loss', **options)
        test_metrics = score(task, yte.values, proba=automl.predict_proba(xte))
    return {'learner': automl.best_estimator, 'wall_s': round(time.time() - started, 1), 'test': test_metrics}


def run_library(kind, entry, train, test):
    target, task = entry['target'], entry['task']
    xtr, xte = feature_frames(train, test, target)
    started = time.time()
    if kind == 'lightgbm':
        import lightgbm
        model = lightgbm.LGBMRegressor(verbose=-1) if task == 'regression' else lightgbm.LGBMClassifier(verbose=-1)
    else:
        import xgboost
        params = dict(enable_categorical=True, tree_method='hist')
        model = xgboost.XGBRegressor(**params) if task == 'regression' else xgboost.XGBClassifier(**params)
    if task == 'regression':
        model.fit(xtr, train[target])
        test_metrics = score(task, test[target].values, pred=model.predict(xte))
    else:
        ytr, yte = encode_classes(train, test, target)
        model.fit(xtr, ytr)
        test_metrics = score(task, yte.values, proba=model.predict_proba(xte))
    return {'learner': kind, 'wall_s': round(time.time() - started, 1), 'test': test_metrics}


def run_all(args):
    mloop = shlex.split(args.mloop)
    entries = json.load(open(os.path.join(HERE, 'datasets.json')))
    if args.datasets:
        wanted = set(args.datasets.split(','))
        entries = [e for e in entries if e['name'] in wanted]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r['dataset'], r['seed'], r['system']))
    cache = os.path.join(args.work, 'openml-cache')
    for seed in [int(s) for s in args.seeds.split(',')]:
        for entry in entries:
            work = os.path.abspath(os.path.join(args.work, f"{entry['name']}-seed{seed}"))
            df = None
            for system in args.systems.split(','):
                if (entry['name'], seed, system) in done:
                    continue
                if df is None:
                    df = load_table(entry, cache)
                    train, test = split(entry, df, seed, work)
                try:
                    if system in ('mloop', 'mloop-defaults'):
                        result = run_mloop(mloop, entry, work, args.budget, aligned=system == 'mloop')
                    elif system == 'flaml':
                        result = run_flaml(entry, train, test, args.budget, seed)
                    elif system == 'flaml-nogbm':
                        linear = 'lassolars' if entry['task'] == 'regression' else 'lrl1'
                        result = run_flaml(entry, train, test, args.budget, seed, ['rf', 'extra_tree', linear])
                    else:
                        result = run_library(system, entry, train, test)
                except Exception as error:  # a failed run is a row, not the end of the sweep
                    result = {'error': repr(error)[:300]}
                record = {'dataset': entry['name'], 'openml_id': entry['id'], 'task': entry['task'], 'rows': len(df),
                          'seed': seed, 'system': system, 'budget': args.budget, **result}
                with open(args.out, 'a') as out:
                    out.write(json.dumps(record) + '\n')
                print(entry['name'], seed, system, json.dumps(result.get('test') or result.get('error'))[:100], flush=True)
            shutil.rmtree(os.path.join(work, 'mloop-project'), ignore_errors=True)


def summarize(paths):
    import numpy as np
    by = collections.defaultdict(dict)
    meta = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            metric, _ = PRIMARY[r['task']]
            value = (r.get('test') or {}).get(metric)
            if value is None:
                continue
            by[r['dataset']].setdefault(r['system'], {})[r['seed']] = value
            meta[r['dataset']] = (r['task'], r['rows'])

    def mean_se(values):
        a = np.array(list(values.values()))
        return a.mean(), (a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float('nan'))

    def verdict(a, b, direction):
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 2:
            return 'n/a'
        diffs = np.array([(a[s] - b[s]) * direction for s in seeds])  # > 0: the first is better
        se = diffs.std(ddof=1) / math.sqrt(len(diffs))
        return 'equal' if abs(diffs.mean()) <= se else ('ahead' if diffs.mean() > 0 else 'behind')

    print('| dataset | task | rows | metric | ' + ' | '.join(SYSTEMS) + ' | mloop vs flaml | mloop vs flaml-nogbm |')
    print('|---|---|---|---|' + '---|' * (len(SYSTEMS) + 2))
    tallies = {'flaml': collections.Counter(), 'flaml-nogbm': collections.Counter()}
    for name in sorted(by, key=lambda n: (meta[n][0], meta[n][1])):
        task, rows = meta[name]
        metric, direction = PRIMARY[task]
        cells = ['%.4f±%.4f' % mean_se(by[name][s]) if s in by[name] else '—' for s in SYSTEMS]
        verdicts = [verdict(by[name].get('mloop', {}), by[name].get(other, {}), direction) for other in tallies]
        for other, v in zip(tallies, verdicts):
            tallies[other][v] += 1
        print(f'| {name} | {task} | {rows} | {metric} | ' + ' | '.join(cells) + ' | ' + ' | '.join(verdicts) + ' |')
    print()
    for other, tally in tallies.items():
        print(f'mloop vs {other}: {dict(tally)}')
    print('\nA pair counts as equal when the mean paired difference over split seeds is within its standard error.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--mloop', required=True, help='command that starts the mloop CLI, e.g. "dotnet path/to/mloop.dll"')
    r.add_argument('--seeds', default='0,1,2', help='split seeds')
    r.add_argument('--budget', type=int, default=60, help='time budget in seconds for the searching systems')
    r.add_argument('--datasets', default='', help='comma-separated dataset names from datasets.json (default: all)')
    r.add_argument('--systems', default=','.join(SYSTEMS))
    r.add_argument('--out', default='results.jsonl')
    r.add_argument('--work', default=os.path.join(tempfile.gettempdir(), 'engine-compare'))
    s = sub.add_parser('summarize')
    s.add_argument('results', nargs='+')
    args = parser.parse_args()
    if args.cmd == 'run':
        run_all(args)
    else:
        summarize(args.results)


if __name__ == '__main__':
    main()
