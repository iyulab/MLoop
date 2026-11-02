#!/usr/bin/env python3
"""Compare MLoop's trainer search with FLAML and stand-alone gradient-boosting libraries on public OpenML tables.

Each (dataset, split seed) gets one stratified 80/20 split; every system receives the same train and test files.

  mloop         `mloop train --time B --metric <reported metric> --test-split 0.05` on the train file, then
                `mloop evaluate` on the held-out file (the search optimizes what the table reports and trains on
                nearly all of the train file)
  mloop-defaults  the same without `--metric` / `--test-split`: the shipped defaults (an accuracy-family metric,
                20% of the train file held back for the model's own test)
  mloop-ensemble  `mloop` with `--ensemble` (binary and regression only): the product path that answers with a blend of the best
                trials when one scores better out of fold. Not in the default `--systems`; name it to run it. Its row records
                whether the run answered with a blend or kept the single model (and why), and `summarize` pairs it with `mloop`
  flaml         FLAML AutoML, time_budget=B, its default learners
  flaml-nogbm   FLAML restricted to rf / extra_tree / a linear model (lrl1, lassolars for regression); no gradient boosting
  flaml-lgbm    FLAML restricted to its LightGBM learner (not in the default `--systems`)
  flaml-forest  FLAML restricted to rf / extra_tree (not in the default `--systems`)
  autogluon     AutoGluon-Tabular, `time_limit=B`, its default preset (not in the default `--systems`; the workflow installs the package
                only when it is named). Its split seed is the table's; AutoGluon takes none of its own
  lightgbm      LightGBM, library defaults
  xgboost       XGBoost, library defaults
  automl-legacy ML.NET's own AutoML (Microsoft.ML.AutoML 0.23.1, the package MLoop's search replaced), time budget B,
                optimizing the reported metric; the runner is automl-legacy/ (build it first)

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
SYSTEMS = ['mloop', 'mloop-defaults', 'flaml', 'flaml-nogbm', 'lightgbm', 'xgboost', 'automl-legacy']
# run only when named in --systems: the blend is for binary classification and regression, and costs an extra search's worth of time;
# the single-family FLAML arms attribute a gap to a learner family
OPTIONAL_SYSTEMS = ['mloop-ensemble', 'flaml-lgbm', 'flaml-forest', 'autogluon']
# the optional systems that stand for another searcher (or a restriction of one) and are read against `mloop` in their own table
OTHER_SYSTEMS = ['flaml-lgbm', 'flaml-forest', 'autogluon']
# `ensemble` is a probe, not a system: one search yields four rows - its best single model, the equal-weight average of its best 3 and 5 trials,
# the member the train rows' out-of-fold answers pick (what forward selection collapses to on one member), and a forward-selected blend of its
# best 5 (weights chosen on those same out-of-fold answers)
ENSEMBLE_ROWS = ['ens-single', 'ens-k3', 'ens-k5', 'ens-oofpick', 'ens-fwd']
# what `mloop train --metric` is given so the search optimizes the metric the table reports
OPTIMIZE = {'binary': 'auc', 'multiclass': 'log_loss', 'regression': 'rmse'}


def load_table(entry, cache, attempts=8):
    from sklearn.datasets import fetch_openml
    # The default (pandas) ARFF parser fails on tables whose categories are numbers; liac-arff reads all 16.
    # OpenML answers "database connection error" under load (and says a table is not found) — a sweep of many jobs
    # starts together, so a failed fetch is retried with a growing wait rather than ending the job.
    for attempt in range(attempts):
        try:
            return fetch_openml(data_id=entry['id'], as_frame=True, parser='liac-arff', data_home=cache).frame
        except Exception as error:
            if attempt == attempts - 1:
                raise
            wait = min(30 * 2 ** attempt, 300)
            print(f"OpenML fetch of {entry['name']} failed ({type(error).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)


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


def read_trials(project, experiment_id):
    """The trials the search kept, as the experiment recorded them (trainer, settings, metrics, runtime, selected)."""
    path = os.path.join(project, 'models', 'default', 'staging', experiment_id, 'trials.ndjson')
    rows = []
    try:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rows.append({key: row[key] for key in ('trialNumber', 'trainer', 'metrics', 'runtimeSeconds', 'selected', 'error') if key in row})
    except OSError:
        pass
    return rows


def read_ensemble_outcome(project, experiment_id):
    """What a run asked for a blend came to: `result.ensemble` in the experiment's metadata.json."""
    for root, _dirs, files in os.walk(os.path.join(project, 'models')):
        if os.path.basename(root) == experiment_id and 'metadata.json' in files:
            try:
                outcome = (json.load(open(os.path.join(root, 'metadata.json'))).get('result') or {}).get('ensemble') or {}
            except (OSError, ValueError):
                return None
            return {'blend': bool(outcome.get('members')), 'members': outcome.get('members'), 'skipped': outcome.get('skippedBecause'),
                    'oof': outcome.get('outOfFoldScore'), 'oof_single': outcome.get('bestSingleOutOfFoldScore')}
    return None


def run_mloop(mloop, entry, work, budget, aligned, ensemble=False):
    project = os.path.join(work, 'mloop-project')
    shutil.rmtree(project, ignore_errors=True)
    init = subprocess.run(mloop + ['init', 'mloop-project', '--task', TASK_FLAG[entry['task']]],
                          cwd=work, capture_output=True, text=True)
    if not os.path.isdir(project):
        return {'error': 'init failed: ' + (init.stdout + init.stderr)[-300:]}
    started = time.time()
    train = subprocess.run(mloop + ['train', os.path.join(work, 'train.csv'), '--label', entry['target'],
                                    '--task', TASK_FLAG[entry['task']], '--time', str(budget), '--no-promote', '--json']
                           + (['--metric', OPTIMIZE[entry['task']], '--test-split', '0.05'] if aligned else [])
                           + (['--ensemble'] if ensemble else []),
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
    row = {'learner': result.get('bestTrainer'), 'trials': trials, 'wall_s': round(wall, 1), 'test': test_metrics,
           'trial_log': read_trials(project, result['experimentId'])}
    if ensemble:
        row['ensemble'] = read_ensemble_outcome(project, result['experimentId'])
    return row


def run_automl_legacy(command, entry, work, budget):
    started = time.time()
    done = subprocess.run(command + [os.path.join(work, 'train.csv'), os.path.join(work, 'test.csv'), entry['target'], entry['task'], str(budget)],
                          capture_output=True, text=True)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith('{'):
            result = json.loads(line)
            result['wall_s'] = round(time.time() - started, 1)  # the process, load and scoring included, as for the other systems
            return result
    return {'error': 'automl-legacy failed: ' + (done.stdout + done.stderr)[-400:]}


def run_ensemble(command, entry, work, budget, seed):
    started = time.time()
    done = subprocess.run(command + [os.path.join(work, 'train.csv'), os.path.join(work, 'test.csv'), entry['target'], entry['task'], str(budget), str(seed), '5'],
                          capture_output=True, text=True)
    for line in reversed(done.stdout.splitlines()):
        if line.startswith('{'):
            probe = json.loads(line)
            wall = round(time.time() - started, 1)
            metric = probe['metric']
            values = {'ens-single': probe['best_single'], 'ens-k3': probe['average'].get('k3'), 'ens-k5': probe['average'].get('k5'),
                      'ens-oofpick': (probe.get('forward') or {}).get('oof_pick_test'), 'ens-fwd': (probe.get('forward') or {}).get('test')}
            return {name: {'test': {metric: value}, 'wall_s': wall, 'trials': probe['trials']} for name, value in values.items() if value is not None}
    return {'ens-single': {'error': 'ensemble-probe failed: ' + (done.stdout + done.stderr)[-400:]}}


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
    # what the search settled on, so a gap to another system can be read against the winning settings and not only the learner's name
    return {'learner': automl.best_estimator, 'wall_s': round(time.time() - started, 1), 'test': test_metrics,
            'best_config': _jsonable(automl.best_config), 'best_loss': _jsonable(automl.best_loss),
            'best_config_train_time': _jsonable(automl.best_config_train_time),
            'best_loss_per_learner': _jsonable(automl.best_loss_per_estimator),
            'best_config_per_learner': _jsonable(automl.best_config_per_estimator)}


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def run_autogluon(entry, train, test, budget, work):
    from autogluon.tabular import TabularPredictor
    import numpy as np
    target, task = entry['target'], entry['task']
    metric = {'binary': 'roc_auc', 'multiclass': 'log_loss', 'regression': 'root_mean_squared_error'}[task]
    path = os.path.join(work, 'autogluon')
    shutil.rmtree(path, ignore_errors=True)
    started = time.time()
    try:
        if task == 'regression':
            predictor = TabularPredictor(label=target, eval_metric=metric, path=path, verbosity=0).fit(train, time_limit=budget)
            test_metrics = score(task, test[target].values, pred=predictor.predict(test.drop(columns=[target])).values)
        else:
            # classes are named by their position in sorted order, as `encode_classes` does for the other systems
            classes = sorted(train[target].astype(str).unique())
            fit_frame = train.assign(**{target: train[target].astype(str)})
            predictor = TabularPredictor(label=target, eval_metric=metric, path=path, verbosity=0).fit(fit_frame, time_limit=budget)
            proba = predictor.predict_proba(test.drop(columns=[target]))
            proba.columns = [str(c) for c in proba.columns]
            proba = proba.reindex(columns=classes, fill_value=0.0)
            _, yte = encode_classes(train, test, target)
            test_metrics = score(task, yte.values, proba=proba.values.astype(np.float64))
        return {'learner': predictor.model_best, 'wall_s': round(time.time() - started, 1), 'test': test_metrics}
    finally:
        shutil.rmtree(path, ignore_errors=True)


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
    ensemble_command = shlex.split(args.ensemble_probe)
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
                if (entry['name'], seed, 'ens-single' if system == 'ensemble' else system) in done:
                    continue
                if df is None:
                    df = load_table(entry, cache)
                    train, test = split(entry, df, seed, work)
                if system == 'ensemble':
                    try:
                        rows = run_ensemble(ensemble_command, entry, work, args.budget, seed)
                    except Exception as error:
                        rows = {'ens-single': {'error': repr(error)[:300]}}
                    for name, result in rows.items():
                        record = {'dataset': entry['name'], 'openml_id': entry['id'], 'task': entry['task'], 'rows': len(df),
                                  'seed': seed, 'system': name, 'budget': args.budget, **result}
                        with open(args.out, 'a') as out:
                            out.write(json.dumps(record) + '\n')
                        print(entry['name'], seed, name, json.dumps(result.get('test') or result.get('error'))[:100], flush=True)
                    continue
                try:
                    if system == 'mloop-ensemble' and entry['task'] not in ('binary', 'regression'):
                        continue   # a blend is made for binary classification and regression
                    if system in ('mloop', 'mloop-defaults', 'mloop-ensemble'):
                        result = run_mloop(mloop, entry, work, args.budget, aligned=system != 'mloop-defaults', ensemble=system == 'mloop-ensemble')
                    elif system == 'automl-legacy':
                        result = run_automl_legacy(shlex.split(args.automl_legacy), entry, work, args.budget)
                    elif system == 'flaml':
                        result = run_flaml(entry, train, test, args.budget, seed)
                    elif system == 'autogluon':
                        result = run_autogluon(entry, train, test, args.budget, work)
                    elif system == 'flaml-lgbm':
                        result = run_flaml(entry, train, test, args.budget, seed, ['lgbm'])
                    elif system == 'flaml-forest':
                        result = run_flaml(entry, train, test, args.budget, seed, ['rf', 'extra_tree'])
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
    wall = collections.defaultdict(dict)    # dataset -> system -> seed -> seconds the `mloop train` process took
    blend = collections.defaultdict(dict)   # dataset -> seed -> what the blend run came to (system mloop-ensemble)
    budget = {}                             # dataset -> the time budget the rows were run with
    ece = collections.defaultdict(dict)     # dataset -> system -> seed -> expected calibration error on the held-out file (MLoop rows only)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            metric, _ = PRIMARY[r['task']]
            value = (r.get('test') or {}).get(metric)
            if value is None:
                continue
            by[r['dataset']].setdefault(r['system'], {})[r['seed']] = value
            meta[r['dataset']] = (r['task'], r['rows'])
            if r.get('wall_s') is not None:
                wall[r['dataset']].setdefault(r['system'], {})[r['seed']] = r['wall_s']
            budget[r['dataset']] = r.get('budget')
            if (r.get('test') or {}).get('ece') is not None:
                ece[r['dataset']].setdefault(r['system'], {})[r['seed']] = r['test']['ece']
            if r['system'] == 'mloop-ensemble':
                blend[r['dataset']][r['seed']] = r.get('ensemble') or {}

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

    tallies = {'flaml': collections.Counter(), 'flaml-nogbm': collections.Counter()}
    if any('automl-legacy' in by[name] for name in by):
        tallies['automl-legacy'] = collections.Counter()
    print('| dataset | task | rows | metric | ' + ' | '.join(SYSTEMS) + ' | ' + ' | '.join(f'mloop vs {other}' for other in tallies) + ' |')
    print('|---|---|---|---|' + '---|' * (len(SYSTEMS) + len(tallies)))
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
    if any('mloop-ensemble' in by[name] for name in by):
        # The product with a blend asked for against the product without: only `--ensemble` differs (same split, same budget, same seeds).
        print('\n| dataset | task | rows | metric | mloop | mloop-ensemble | on vs off | blend chosen | train s off | train s on |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        onoff, chosen_total, run_total = collections.Counter(), 0, 0
        for name in sorted(by, key=lambda n: (meta[n][0], meta[n][1])):
            if 'mloop-ensemble' not in by[name]:
                continue
            task, rows = meta[name]
            metric, direction = PRIMARY[task]
            v = verdict(by[name]['mloop-ensemble'], by[name].get('mloop', {}), direction)
            onoff[v] += 1
            chosen = sum(1 for o in blend[name].values() if o.get('blend'))
            chosen_total, run_total = chosen_total + chosen, run_total + len(blend[name])
            seconds = lambda system: (sum(wall[name].get(system, {}).values()) / len(wall[name][system])) if wall[name].get(system) else float('nan')
            print(f"| {name} | {task} | {rows} | {metric} | {'%.4f±%.4f' % mean_se(by[name]['mloop']) if 'mloop' in by[name] else '—'} | "
                  f"{'%.4f±%.4f' % mean_se(by[name]['mloop-ensemble'])} | {v} | {chosen}/{len(blend[name])} | {seconds('mloop'):.0f} | {seconds('mloop-ensemble'):.0f} |")
        print(f'\nmloop-ensemble vs mloop: {dict(onoff)}; the blend was chosen in {chosen_total} of {run_total} runs')
        reasons = collections.Counter((o.get('skipped') or '')[:70] for per in blend.values() for o in per.values() if not o.get('blend'))
        for reason, n in reasons.most_common():
            print(f'  single model kept ({n}): {reason or "(no reason recorded)"}')
    others = [o for o in OTHER_SYSTEMS if any(o in by[name] for name in by)]
    if others:
        # The restricted FLAML arms and AutoGluon against `mloop` and `flaml` on the tables they ran on.
        print('\n| dataset | task | rows | metric | mloop | flaml | ' + ' | '.join(others) + ' | ' + ' | '.join(f'mloop vs {o}' for o in others) + ' |')
        print('|---|---|---|---|---|---|' + '---|' * (2 * len(others)))
        counts = {o: collections.Counter() for o in others}
        for name in sorted(by, key=lambda n: (meta[n][0], meta[n][1])):
            if not any(o in by[name] for o in others):
                continue
            task, rows = meta[name]
            metric, direction = PRIMARY[task]
            cells = ['%.4f±%.4f' % mean_se(by[name][s]) if s in by[name] else '—' for s in ['mloop', 'flaml', *others]]
            verdicts = [verdict(by[name].get('mloop', {}), by[name].get(o, {}), direction) for o in others]
            for o, v in zip(others, verdicts):
                counts[o][v] += 1
            print(f'| {name} | {task} | {rows} | {metric} | ' + ' | '.join(cells) + ' | ' + ' | '.join(verdicts) + ' |')
        print()
        for o, tally in counts.items():
            print(f'mloop vs {o}: {dict(tally)}')
    if any('ens-single' in by[name] for name in by):
        # The average of the search's best trials against that same search's best single model: only the average differs.
        counts = {k: collections.Counter() for k in ('ens-k3', 'ens-k5', 'ens-oofpick', 'ens-fwd')}
        print('\n| dataset | task | rows | metric | ens-single | ens-k3 | ens-k5 | ens-oofpick | ens-fwd | k3 vs single | k5 vs single | oofpick vs single | fwd vs single |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for name in sorted(by, key=lambda n: (meta[n][0], meta[n][1])):
            if 'ens-single' not in by[name]:
                continue
            task, rows = meta[name]
            metric, direction = PRIMARY[task]
            cells = ['%.4f±%.4f' % mean_se(by[name][s]) if s in by[name] else '—' for s in ENSEMBLE_ROWS]
            verdicts = []
            for k in counts:
                v = verdict(by[name].get(k, {}), by[name]['ens-single'], direction)
                counts[k][v] += 1
                verdicts.append(v)
            print(f'| {name} | {task} | {rows} | {metric} | ' + ' | '.join(cells) + ' | ' + ' | '.join(verdicts) + ' |')
        print()
        for k, tally in counts.items():
            print(f'{k} vs ens-single: {dict(tally)}')

    if 'mloop' in {s for name in by for s in by[name]} and any(s in by[name] for name in by for s in ('flaml', 'mloop-ensemble')):
        seed_view(by, meta, wall, budget, ece, verdict)


def signs(a, b, direction):
    """One character per split seed both systems have, in seed order: + the first is better, - worse, = the same value."""
    return ' '.join('=' if a[s] == b[s] else ('+' if (a[s] - b[s]) * direction > 0 else '-') for s in sorted(set(a) & set(b))) or '—'


def over_budget(seconds, budget):
    """A `mloop train` that took longer than its budget plus the room a trial in flight needs (the larger of 2 s and 20 %)."""
    return budget is not None and seconds > budget + max(2.0, 0.2 * budget)


def seed_view(by, meta, wall, budget, ece, verdict):
    """What the means hide: the sign of each split seed, runs that overran their budget, and the calibration of the held-out answers."""
    # the product as asked for: a blend where `--ensemble` was run (binary, regression), the single model where it was not (multiclass)
    shipped = lambda name: by[name].get('mloop-ensemble') or by[name].get('mloop') or {}
    pairs = [(label, a, b) for label, a, b in (('mloop vs flaml', lambda n: by[n].get('mloop', {}), lambda n: by[n].get('flaml', {})),
                                               ('ensemble vs flaml', lambda n: by[n].get('mloop-ensemble', {}), lambda n: by[n].get('flaml', {})),
                                               ('ensemble where run vs flaml', shipped, lambda n: by[n].get('flaml', {})),
                                               ('ensemble vs mloop', lambda n: by[n].get('mloop-ensemble', {}), lambda n: by[n].get('mloop', {})))
             if any(a(name) and b(name) for name in by)]
    print('\nPer split seed (+ the first system is better, - worse, = the same value; seeds in order):')
    print('| dataset | task | ' + ' | '.join(f'{label} (verdict)' for label, _, _ in pairs) + ' | over budget mloop | over budget ensemble | test ECE mloop | test ECE ensemble |')
    print('|---|---|' + '---|' * (len(pairs) + 4))
    tallies = {label: collections.Counter() for label, _, _ in pairs}
    overruns = collections.Counter()
    ran = collections.Counter()
    for name in sorted(by, key=lambda n: (meta[n][0], meta[n][1])):
        task, _ = meta[name]
        direction = PRIMARY[task][1]
        cells = []
        for label, a, b in pairs:
            if a(name) and b(name):
                v = verdict(a(name), b(name), direction)
                tallies[label][v] += 1
                cells.append(f'{signs(a(name), b(name), direction)} ({v})')
            else:
                cells.append('—')
        late = {}
        for system in ('mloop', 'mloop-ensemble'):
            seconds = wall[name].get(system, {})
            late[system] = sum(1 for t in seconds.values() if over_budget(t, budget.get(name)))
            overruns[system] += late[system]
            ran[system] += len(seconds)
        mean = lambda system: ('%.3f' % (sum(ece[name][system].values()) / len(ece[name][system]))) if ece[name].get(system) else '—'
        over = lambda system: f'{late[system]}/{len(wall[name][system])}' if wall[name].get(system) else '—'
        print(f'| {name} | {task} | ' + ' | '.join(cells) + f" | {over('mloop')} | {over('mloop-ensemble')} | {mean('mloop')} | {mean('mloop-ensemble')} |")
    print()
    for label, tally in tallies.items():
        print(f'{label}: {dict(tally)}')
    for system in ('mloop', 'mloop-ensemble'):
        if ran[system]:
            print(f'{system} runs over budget + max(2 s, 20 %): {overruns[system]} of {ran[system]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--mloop', required=True, help='command that starts the mloop CLI, e.g. "dotnet path/to/mloop.dll"')
    r.add_argument('--seeds', default='0,1,2', help='split seeds')
    r.add_argument('--budget', type=int, default=60, help='time budget in seconds for the searching systems')
    r.add_argument('--datasets', default='', help='comma-separated dataset names from datasets.json (default: all)')
    r.add_argument('--systems', default=','.join(SYSTEMS), help='comma-separated; ' + ', '.join(OPTIONAL_SYSTEMS) + ' runs only when named')
    r.add_argument('--automl-legacy', default='dotnet ' + os.path.join(HERE, 'automl-legacy', 'bin', 'Release', 'net10.0', 'AutomlLegacy.dll'),
                   help='command that starts the legacy AutoML runner (system automl-legacy)')
    r.add_argument('--ensemble-probe', default='dotnet ' + os.path.join(HERE, 'ensemble-probe', 'bin', 'Release', 'net10.0', 'EnsembleProbe.dll'),
                   help='command that starts the ensemble probe (used by the `ensemble` pseudo-system)')
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
