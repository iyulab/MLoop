#!/usr/bin/env python3
"""Score `mloop detect` against the Numenta Anomaly Benchmark labels - a window-hit reading, NOT NAB's official score.

NAB labels a window around each labelled timestamp (together 10% of the series, split over the anomalies, centered on the label).
Per series: window recall (windows holding at least one flagged point), flagged points outside every window (false alarms, as a share of the
points outside the windows), and whether the command finished. The aggregate is the mean over series; a series with no label counts only in
the false-alarm share (artificialNoAnomaly).

  python detect_vs_nab.py --mloop "dotnet /abs/path/mloop.dll" --nab <NAB checkout: data/ and labels/> --out results.json [--limit N] [--only substr]
"""
import argparse, json, os, subprocess, sys, time
import pandas as pd


def windows(ts, labels):
    n = len(ts)
    if not labels:
        return []
    half = max(1, int(0.10 * n / len(labels) / 2))
    idx = {t: i for i, t in enumerate(ts)}
    out = []
    for lab in labels:
        if lab in idx:
            i = idx[lab]
            out.append((max(0, i - half), min(n - 1, i + half)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True); ap.add_argument('--nab', required=True, help='a checkout of numenta/NAB (MIT) holding data/ and labels/'); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    a = ap.parse_args()
    nab = a.nab
    labels = json.load(open(os.path.join(nab, 'labels', 'combined_labels.json')))
    rows = []
    for key in sorted(labels):
        if a.only and a.only not in key:
            continue
        path = os.path.join(nab, 'data', key)
        df = pd.read_csv(path)
        ts = df['timestamp'].astype(str).tolist()
        t0 = time.time()
        p = subprocess.run(a.mloop.split() + ['detect', path, '--column', 'value', '--json'], capture_output=True, text=True)
        secs = round(time.time() - t0, 1)
        if p.returncode != 0:
            rows.append({'series': key, 'rc': p.returncode, 'secs': secs, 'error': (p.stdout + p.stderr)[-300:]})
            continue
        try:
            res = json.loads(p.stdout)
        except Exception as e:
            rows.append({'series': key, 'rc': 0, 'secs': secs, 'error': 'unparsable json: ' + str(e)[:100]}); continue
        flagged = {pt['index'] for pt in res['points'] if pt['isAnomaly']}
        wins = windows(ts, labels[key])
        inside = set()
        for lo, hi in wins:
            inside.update(range(lo, hi + 1))
        hit = sum(1 for lo, hi in wins if any(lo <= i <= hi for i in flagged))
        outside_flagged = len([i for i in flagged if i not in inside])
        outside_total = len(ts) - len(inside)
        rows.append({'series': key, 'rc': 0, 'secs': secs, 'points': len(ts), 'labels': len(labels[key]), 'windows': len(wins), 'windows_hit': hit,
                     'flagged': len(flagged), 'flagged_outside': outside_flagged, 'false_alarm_share': round(outside_flagged / max(1, outside_total), 5)})
        if a.limit and len(rows) >= a.limit:
            break
    json.dump(rows, open(a.out, 'w'), indent=1)
    ok = [r for r in rows if r['rc'] == 0 and 'error' not in r]
    lab = [r for r in ok if r['windows'] > 0]
    print('series', len(rows), 'finished', len(ok), 'failed', len(rows) - len(ok))
    print('series with labels', len(lab), 'window recall (pooled)', sum(r['windows_hit'] for r in lab), '/', sum(r['windows'] for r in lab))
    print('mean false-alarm share (outside the windows)', round(sum(r['false_alarm_share'] for r in ok) / max(1, len(ok)), 5))
    print('no-label series flagged points', sum(r['flagged'] for r in ok if r['windows'] == 0), 'of', sum(r['points'] for r in ok if r['windows'] == 0))
    for r in rows:
        if 'error' in r:
            print('  FAIL', r['series'], r['error'][:160].replace('\n', ' '))


main()
