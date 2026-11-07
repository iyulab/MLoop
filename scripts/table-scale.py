#!/usr/bin/env python3
"""Walks the tabular tasks (binary and multiclass classification, regression) over tables of growing size - rows, classes, columns - and records, per
step, the wall time and the peak resident memory of the CLI process, and whether the training kept to its time budget.

The lifecycle walk (lifecycle-check.py) uses tables of a few hundred rows. This script asks the other question: does a table of a million rows, or of
a hundred classes, still train inside the time it was given, with what memory, and does the model it kept predict and evaluate. The table is
synthetic and learnable (the label follows a projection of the numeric columns, cut into the classes, plus noise), so a failure is the tool's, not
the data's. On Linux the trainer search includes LightGbm, which the macOS build does not have.

  python table-scale.py --mloop "dotnet path/to/mloop.dll" [--rows 10000,100000,1000000] [--classes 2,10,100] [--features 20] [--tasks binary,multiclass,regression]

Linux and macOS (peak memory comes from getrusage of a wrapper's child: kilobytes on Linux, bytes on macOS).
"""
import argparse, csv, json, math, os, platform, random, shlex, subprocess, sys, tempfile, time

WRAPPER = ("import json,resource,subprocess,sys;p=subprocess.run(sys.argv[1:],capture_output=True,text=True);"
           "print(json.dumps({'rc':p.returncode,'maxrss':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'out':p.stdout[-4000000:],'err':p.stderr[-2000:]}))")
STEP_TIMEOUT = 3600
HEAD = 500
PREDICT_ROWS = 10_000

TASKS = {'binary': 'binary-classification', 'multiclass': 'multiclass-classification', 'regression': 'regression'}


def write_table(path, n, features, classes, task, seed, signal='linear'):
    """Writes n rows: `features` numeric columns f0.. and a label y. The label follows s = sum(a_i f_i) + 0.5 * f0 * f1 + noise, standardized: a
    regression's y is s itself; a classification's y is the class its quantile falls in (cut points at the normal quantiles, so the classes are
    balanced). A third of the columns carry no signal.

    `signal='interactions'` replaces the linear projection with a sum of products of column pairs, s = sum(a_k f_2k f_2k+1): no column predicts the
    label alone and a linear model learns nothing, so the score keeps rising with the rows a tree-based model sees - a table on which fitting the
    model on a sample of the rows costs something, which the linear signal (saturated by tens of thousands of rows) cannot show."""
    rng = random.Random(seed)
    useful = max(2, features * 2 // 3)
    weights = [rng.uniform(-1, 1) for _ in range(useful)]
    pairs = max(1, useful // 2)
    if signal == 'interactions':
        spread = math.sqrt(sum(w * w for w in weights[:pairs]) + 0.3 ** 2)
    else:
        spread = math.sqrt(sum(w * w for w in weights) + 0.25 + 0.3 ** 2)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([f'f{i}' for i in range(features)] + ['y'])
        for _ in range(n):
            x = [rng.gauss(0, 1) for _ in range(features)]
            if signal == 'interactions':
                s = sum(weights[k] * x[2 * k] * x[2 * k + 1] for k in range(pairs)) + rng.gauss(0, 0.3)
            else:
                s = sum(weights[i] * x[i] for i in range(useful)) + 0.5 * x[0] * x[1] + rng.gauss(0, 0.3)
            z = s / spread
            if task == 'regression':
                y = round(z, 5)
            else:
                q = 0.5 * (1 + math.erf(z / math.sqrt(2)))   # standard normal cdf, in (0, 1)
                y = min(classes - 1, int(q * classes))
            w.writerow([round(v, 5) for v in x] + [y])


def step(mloop, name, args, cwd, log, timeout=None):
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, '-c', WRAPPER, *shlex.split(mloop), *args], cwd=cwd, capture_output=True, text=True, timeout=timeout or STEP_TIMEOUT)
        wall = time.monotonic() - t0
        info = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {'rc': -1, 'maxrss': 0, 'out': '', 'err': p.stderr[-500:]}
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        info = {'rc': 'timeout', 'maxrss': 0, 'out': '', 'err': f'timeout after {timeout or STEP_TIMEOUT}s'}
    kb = info['maxrss'] / (1 if platform.system() == 'Linux' else 1024)
    rec = {'step': name, 'rc': info['rc'], 'wall_s': round(wall, 1), 'peak_mb': round(kb / 1024, 0), 'tail': (info['err'] or info['out'])[-300:].replace('\n', ' '),
           'head': (info['err'] or info['out'])[:HEAD].replace('\n', ' '), 'out': info['out'] if name in ('train', 'evaluate') else ''}
    log.append(rec)
    print(f"   {name:<9} rc={rec['rc']} wall={rec['wall_s']}s peak={rec['peak_mb']:.0f}MB" + ('' if info['rc'] == 0 else f"  HEAD: {rec['head']}  TAIL: {rec['tail']}"), flush=True)
    return info


def events(out):
    return [json.loads(l) for l in out.splitlines() if l.startswith('{')]


def result_of(out):
    """The training result the --json output ends with: its trainer, trial count and metrics (None where it does not parse)."""
    try:
        ev = events(out)
        r = next((e for e in reversed(ev) if e.get('event') == 'result'), None) or {}
        return {k: r.get(k) for k in ('bestTrainer', 'trainer', 'trials', 'trialCount', 'metrics', 'promoted', 'rows') if k in r}
    except Exception as e:
        return {'result_error': repr(e)[:120]}


def walk(mloop, task, n, classes, features, time_budget, seed, timeout=None, signal='linear'):
    base = tempfile.mkdtemp(prefix=f'table-scale-{task}-')
    log = []
    print(f'== {task} rows={n:,} classes={classes if task != "regression" else "-"} features={features} time={time_budget}s seed={seed}  ({base})', flush=True)
    t0 = time.monotonic()
    write_table(os.path.join(base, 'train.csv'), n, features, classes, task, seed * 2 - 1, signal)
    write_table(os.path.join(base, 'test.csv'), min(n, PREDICT_ROWS), features, classes, task, seed * 2, signal)
    gen_s = round(time.monotonic() - t0, 1)
    step(mloop, 'init', ['init', 'proj', '--task', TASKS[task], '--label', 'y', '--json'], base, log)
    proj = os.path.join(base, 'proj')
    result = {}
    tr = step(mloop, 'train', ['train', os.path.join(base, 'train.csv'), '--label', 'y', '--time', str(time_budget), '--json'], proj, log, timeout)
    if tr['rc'] == 0:
        result = result_of(tr.get('out', ''))
        step(mloop, 'predict', ['predict', os.path.join(base, 'test.csv'), '--json'], proj, log)
        step(mloop, 'evaluate', ['evaluate', 'exp-001', os.path.join(base, 'test.csv'), '--json'], proj, log)
    train_wall = next((s['wall_s'] for s in log if s['step'] == 'train'), None)
    return {'task': task, 'signal': signal, 'rows': n, 'classes': classes if task != 'regression' else None, 'features': features, 'time_budget': time_budget, 'seed': seed,
            'generate_s': gen_s, 'train_over_budget_s': round(train_wall - time_budget, 1) if train_wall is not None else None, **result, 'steps': log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True)
    ap.add_argument('--rows', default='10000,100000,1000000')
    ap.add_argument('--classes', default='2,10,100', help='classification: comma-separated class counts (2 is binary, the others multiclass); regression ignores it')
    ap.add_argument('--features', default='20', help='comma-separated numeric column counts')
    ap.add_argument('--tasks', default='binary,multiclass,regression')
    ap.add_argument('--time', type=int, default=60)
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--signal', default='linear', choices=['linear', 'interactions'], help='the label function: a linear projection (default) or a sum of products of column pairs')
    ap.add_argument('--step-timeout', type=int, default=0, help='seconds per step (0 = 3600)')
    ap.add_argument('--out', default='table-scale-results.json')
    a = ap.parse_args()
    results = []
    for task in a.tasks.split(','):
        class_counts = [int(c) for c in a.classes.split(',')] if task != 'regression' else [0]
        for k in class_counts:
            if (task == 'binary') != (k == 2) and task != 'regression':
                continue   # binary walks 2 classes, multiclass the others
            for f in a.features.split(','):
                for n in a.rows.split(','):
                    for sd in a.seeds.split(','):
                        results.append(walk(a.mloop, task, int(n), k, int(f), a.time, int(sd), a.step_timeout or None, a.signal))
                        json.dump(results, open(a.out, 'w'), indent=1)   # a long walk keeps what it has measured
    print('\n| task | rows | classes | cols | step | rc | wall s | peak MB | over budget s | trainer |\n|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        for s in r['steps']:
            extra = f"{r['train_over_budget_s']}" if s['step'] == 'train' else ''
            trainer = r.get('bestTrainer') or r.get('trainer') or '' if s['step'] == 'train' else ''
            print(f"| {r['task']} | {r['rows']:,} | {r['classes'] or ''} | {r['features']} | {s['step']} | {s['rc']} | {s['wall_s']} | {s['peak_mb']:.0f} | {extra} | {trainer} |")
    return 0 if all(s['rc'] == 0 for r in results for s in r['steps']) else 1


if __name__ == '__main__':
    sys.exit(main())
