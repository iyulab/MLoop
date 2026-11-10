#!/usr/bin/env python3
"""Does MLoop's monitoring see drift that nobody injected? A time-ordered public table is cut into consecutive windows and walked in order.

The lifecycle walk (`lifecycle-check.py`) makes its drift: it moves a column and rewrites the label, so a pass says "the wiring flags
what we planted". Here a real series changes by itself. The first model learns the oldest part; later windows are predicted with
`--log`, `drift` is read after each, and the production model's accuracy on the window is measured with `evaluate`. Then a model
retrained on the oldest 60% is compared with production on the last window (`evaluate --against production`).

This REPORTS, it does not assert: whether drift flags line up with a fall in accuracy is the measurement.

  python natural_drift_walk.py --mloop "dotnet /abs/path/mloop.dll" --table electricity.csv --label class --out result.json

--table is a CSV whose rows are in time order (the oldest first); the OpenML table "electricity" (data id 151) is one.
"""
import argparse, csv, json, os, shutil, subprocess, sys, tempfile


def run(mloop, args, cwd, tolerate=False):
    p = subprocess.run(mloop.split() + args, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0 and not tolerate:
        raise SystemExit(f'FAILED: {" ".join(args)} rc={p.returncode}\n{(p.stdout + p.stderr)[-600:]}')
    return p


def jrun(mloop, args, cwd, tolerate=False):
    p = run(mloop, args, cwd, tolerate)
    out = p.stdout
    try:
        return p.returncode, json.loads(out[out.index('{'):out.rindex('}') + 1])
    except Exception:
        return p.returncode, None


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True); ap.add_argument('--table', required=True, help='CSV, rows in time order'); ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--windows', type=int, default=6)
    ap.add_argument('--time', type=int, default=60); ap.add_argument('--trials', type=int, default=10)
    a = ap.parse_args()
    src = a.table
    with open(src, newline='') as f:
        r = csv.reader(f); header = next(r); rows = list(r)
    n = len(rows); li = header.index(a.label)
    first = int(n * 0.2)                              # the oldest fifth is what the first model learns
    rest = rows[first:]; wsize = len(rest) // a.windows
    work = tempfile.mkdtemp(prefix='natural-drift-')
    proj = os.path.join(work, 'proj')
    run(a.mloop, ['init', 'proj', '--task', 'binary-classification', '--label', a.label, '--json'], work)
    d = os.path.join(proj, 'datasets')
    write(os.path.join(d, 'train_a.csv'), header, rows[:first])
    rc, tr = jrun(a.mloop, ['train', os.path.join(d, 'train_a.csv'), '--label', a.label, '--time', str(a.time), '--max-trials', str(a.trials), '--json'], proj)
    report = {'table': os.path.basename(a.table), 'rows': n, 'first_model_rows': first, 'window_rows': wsize, 'windows': []}
    for k in range(a.windows):
        w = rest[k * wsize:(k + 1) * wsize]
        wl = os.path.join(d, f'w{k}_labeled.csv'); wp = os.path.join(d, f'w{k}_predict.csv')
        write(wl, header, w)
        write(wp, [h for i, h in enumerate(header) if i != li], [[v for i, v in enumerate(row) if i != li] for row in w])
        run(a.mloop, ['predict', wp, '--log', '--json'], proj)
        rc, drift = jrun(a.mloop, ['drift', '--rows', str(len(w)), '--json'], proj)
        rc, ev = jrun(a.mloop, ['evaluate', 'exp-001', wl, '--json'], proj, tolerate=True)
        dr = (drift or {}).get('drift') or {}
        cols = sorted(((c.get('column'), round(c.get('psi') or 0, 3)) for c in (dr.get('features') or [])), key=lambda t: -(t[1] or 0))[:2]
        report['windows'].append({
            'window': k, 'rows': len(w), 'status': (drift or {}).get('status'), 'level': dr.get('level'), 'max_psi': dr.get('maxPsi'),
            'top_columns': cols, 'accuracy_exp001': ((ev or {}).get('testMetrics') or {}).get('accuracy')})
        print(report['windows'][-1], flush=True)
    # a model retrained on the oldest 60%, compared on the last window
    write(os.path.join(d, 'train_b.csv'), header, rows[:int(n * 0.6)])
    rc, tb = jrun(a.mloop, ['train', os.path.join(d, 'train_b.csv'), '--label', a.label, '--time', str(a.time), '--max-trials', str(a.trials), '--no-promote', '--json'], proj)
    rc, cmp_ = jrun(a.mloop, ['evaluate', 'exp-002', os.path.join(d, f'w{a.windows - 1}_labeled.csv'), '--against', 'production', '--json'], proj, tolerate=True)
    report['retrain_60pct_vs_production_on_last_window'] = {'exp002_test': (cmp_ or {}).get('testMetrics'), 'comparison': (cmp_ or {}).get('comparison')}
    json.dump(report, open(a.out, 'w'), indent=1, default=str)
    shutil.rmtree(work, ignore_errors=True)


main()
