#!/usr/bin/env python3
"""Walks the series tasks (forecasting, time-series anomaly) over series of growing length and records, per step, the wall time and the peak
resident memory of the CLI process.

The lifecycle walk (lifecycle-check.py) uses series of a few hundred points. This script asks a different question: does a step's cost stay
close to linear in the length of the series, and does a long series still finish. The series is synthetic and learnable (a season of 12, a
trend and noise; the anomaly series gets a spike every 1,000 points), so a failure is the tool's, not the data's.

  python series-scale.py --mloop "dotnet path/to/mloop.dll" [--rows 10000,100000,1000000] [--tasks forecast,anomaly] [--out series-scale-results.json]

Linux and macOS (peak memory comes from getrusage of a wrapper's child: kilobytes on Linux, bytes on macOS).
"""
import argparse, csv, json, math, os, platform, random, shlex, subprocess, sys, tempfile, time

WRAPPER = ("import json,resource,subprocess,sys;p=subprocess.run(sys.argv[1:],capture_output=True,text=True);"
           "print(json.dumps({'rc':p.returncode,'maxrss':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'out':p.stdout[-4000:],'err':p.stderr[-2000:]}))")
STEP_TIMEOUT = 1800
HEAD = 500


WEEKDAYS = (0.0, 0.0, 0.0, 0.0, 0.0, -4.0, -6.0)
HORIZON = 10


def write_series(path, n, spikes, seed, shape='monthly'):
    """Writes one series and returns the mean absolute noise over its last HORIZON points: what a forecaster that knew the signal exactly would still miss.

    'monthly' is a season of 12; 'daily' is a daily series with a yearly season of 365 (two harmonics) and a weekly pattern, the shape on which a
    window shorter than the season is in question."""
    rng = random.Random(seed)
    tail = []
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['value'])
        for t in range(n):
            noise = rng.gauss(0, 1.0)
            if shape == 'daily':
                signal = (100.0 + 0.01 * t + 40.0 * math.sin(2 * math.pi * t / 365) + 10.0 * math.sin(4 * math.pi * t / 365 + 1.0) + WEEKDAYS[t % 7])
            else:
                signal = 100.0 + 0.00005 * t + 10.0 * math.sin(2 * math.pi * t / 12)
            v = signal + noise
            if spikes and t % 1000 == 500:
                v += 40.0
            if t >= n - HORIZON:
                tail.append(abs(noise))
            w.writerow([round(v, 4)])
    return round(sum(tail) / len(tail), 3)


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
    rec = {'step': name, 'rc': info['rc'], 'wall_s': round(wall, 1), 'peak_mb': round(kb / 1024, 0), 'tail': (info['err'] or info['out'])[-300:].replace('\n', ' '), 'head': (info['err'] or info['out'])[:HEAD].replace('\n', ' '), 'out': info['out'] if name in ('train', 'evaluate') else ''}
    log.append(rec)
    print(f"   {name:<9} rc={rec['rc']} wall={rec['wall_s']}s peak={rec['peak_mb']:.0f}MB" + ('' if info['rc'] == 0 else f"  HEAD: {rec['head']}  TAIL: {rec['tail']}"), flush=True)
    return info


def walk(mloop, task, n, time_budget, window=None, timeout=None, seed=1, shape='monthly'):
    base = tempfile.mkdtemp(prefix=f'series-scale-{task}-')
    log = []
    spikes = task == 'anomaly'
    oracle = write_series(os.path.join(base, 'train.csv'), n, spikes, seed * 2 - 1, shape)
    write_series(os.path.join(base, 'test.csv'), n, spikes, seed * 2, shape)
    flag = 'forecasting' if task == 'forecast' else 'time-series-anomaly'
    print(f'== {task} n={n:,} window={window or "default"} seed={seed}  ({base})', flush=True)
    step(mloop, 'init', ['init', 'proj', '--task', flag, '--label', 'value', '--json'], base, log)
    proj = os.path.join(base, 'proj')
    if window:
        yaml_path = os.path.join(proj, 'mloop.yaml')
        text = open(yaml_path).read().replace('    horizon: 10\n', f'    horizon: 10\n    window_size: {window}\n', 1)
        open(yaml_path, 'w').write(text)
    tr = step(mloop, 'train', ['train', os.path.join(base, 'train.csv'), '--label', 'value', '--time', str(time_budget), '--json'], proj, log, timeout)
    if tr['rc'] == 0:
        if task == 'forecast':
            step(mloop, 'predict', ['predict', '--json'], proj, log)
        else:
            step(mloop, 'predict', ['predict', os.path.join(base, 'test.csv'), '--json'], proj, log)
        step(mloop, 'evaluate', ['evaluate', 'exp-001', os.path.join(base, 'test.csv'), '--json'], proj, log)
    return {'task': task, 'rows': n, 'window': window, 'seed': seed, 'shape': shape, 'oracle_mae': oracle, 'steps': log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True)
    ap.add_argument('--rows', default='10000,100000,1000000')
    ap.add_argument('--tasks', default='forecast,anomaly')
    ap.add_argument('--time', type=int, default=60)
    ap.add_argument('--windows', default='', help='forecast only: comma-separated SSA window sizes to try at every length (window_size in mloop.yaml); empty = the default')
    ap.add_argument('--seeds', default='1', help='comma-separated series seeds (the noise differs)')
    ap.add_argument('--step-timeout', type=int, default=0, help='seconds per step (0 = 1800)')
    ap.add_argument('--shape', default='monthly', choices=['monthly', 'daily'], help='monthly = a season of 12; daily = a yearly season of 365 and a weekly pattern')
    ap.add_argument('--out', default='series-scale-results.json')
    a = ap.parse_args()
    windows = [int(w) for w in a.windows.split(',') if w] or [None]
    results = [walk(a.mloop, t, int(n), a.time, w if t == 'forecast' else None, a.step_timeout or None, int(sd), a.shape)
               for t in a.tasks.split(',') for n in a.rows.split(',') for w in (windows if t == 'forecast' else [None]) for sd in a.seeds.split(',')]
    json.dump(results, open(a.out, 'w'), indent=1)
    print('\n| task | rows | window | seed | step | rc | wall s | peak MB | mae | oracle mae |\n|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        for s in r['steps']:
            mae = ''
            if s['step'] in ('train', 'evaluate') and s.get('out'):
                try:
                    ev = [json.loads(l) for l in s['out'].splitlines() if l.startswith('{')]
                    m = next((e for e in ev if e.get('event') == 'result'), None) or next((e for e in ev if 'testMetrics' in e), {})
                    mae = ((m.get('metrics') or m.get('testMetrics')) or {}).get('mae', '')
                except Exception:
                    mae = '?'
            print(f"| {r['task']} | {r['rows']:,} | {r.get('window') or 'default'} | {r.get('seed', 1)} | {s['step']} | {s['rc']} | {s['wall_s']} | {s['peak_mb']:.0f} | {mae} | {r.get('oracle_mae', '')} |")
    return 0 if all(s['rc'] == 0 for r in results for s in r['steps']) else 1


if __name__ == '__main__':
    sys.exit(main())
