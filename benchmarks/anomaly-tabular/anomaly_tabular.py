#!/usr/bin/env python3
"""Tabular anomaly detection read against a rare class.

Each table is a public classification table whose rarer class stands for "the anomaly". `anomaly-detection` is trained on the
features only (the label is held back), the same rows are scored, and the score is read against the label: AUC by rank, the share
of the true anomalies among the k highest scores (k = their count), the number of rows `isAnomaly` calls anomalous, and the rows
that came back with no score. A table the command cannot train on is reported with its error.

This REPORTS. An unsupervised detector on a table that was never an anomaly problem is a stress of the wiring and a coarse reading
of the score, not a quality claim.

  python anomaly_tabular.py --mloop "dotnet /abs/path/mloop.dll" --out result.json \
      --table wilt.csv:class --table Satellite.csv:Target --table seismic-bumps.csv:HighEnergySeismicBump ...

--table is `<csv path>:<label column>`, repeatable. The OpenML tables wilt (40983), Satellite (40900), seismic-bumps (46956),
ozone-level-8hr (1487), sick (38) and taiwanese_bankruptcy_prediction (46962) are the set the first reading used.
"""
import argparse, json, os, shutil, subprocess, tempfile
import numpy as np
import pandas as pd


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def predictions_of(stdout):
    start = stdout.index('{"model"')
    return json.JSONDecoder().raw_decode(stdout[start:])[0]['predictions']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--table', action='append', required=True, help='<csv path>:<label column>')
    a = ap.parse_args()
    mloop = a.mloop.split()
    results = []
    for spec in a.table:
        path, label = spec.rsplit(':', 1)
        d = pd.read_csv(path)
        y = d[label]
        yy = (y == y.value_counts().idxmin()).astype(int).to_numpy()
        features = d.drop(columns=[label])
        work = tempfile.mkdtemp(prefix='anomaly-tabular-')
        project = os.path.join(work, 'p')
        subprocess.run(mloop + ['init', 'p', '--task', 'anomaly-detection', '--json'], cwd=work, capture_output=True)
        features.to_csv(os.path.join(project, 'datasets', 'train.csv'), index=False)
        train = subprocess.run(mloop + ['train', 'datasets/train.csv', '--json'], cwd=project, capture_output=True, text=True)
        predict = subprocess.run(mloop + ['predict', 'datasets/train.csv', '--json'], cwd=project, capture_output=True, text=True)
        row = {'table': os.path.basename(path), 'rows': len(d), 'features': features.shape[1], 'anomaly_share': round(float(yy.mean()), 4),
               'train_rc': train.returncode, 'predict_rc': predict.returncode}
        try:
            preds = predictions_of(predict.stdout)
            scored = [i for i, p in enumerate(preds) if 'anomalyScore' in p]
            score = np.array([preds[i]['anomalyScore'] for i in scored])
            truth = yy[scored]
            k = int(truth.sum())
            row.update({'auc': round(float(auc(score, truth)), 4), 'top_k_hit_share': round(float(truth[np.argsort(-score)[:k]].mean()), 4),
                        'flagged': int(sum(p['isAnomaly'] for p in preds)), 'unscored_rows': len(preds) - len(scored)})
        except Exception as e:
            row['error'] = ((train.stdout + train.stderr)[-250:] + ' | ' + str(e)[:80]).replace('\n', ' ')
        results.append(row)
        print(row, flush=True)
        shutil.rmtree(work, ignore_errors=True)
    json.dump(results, open(a.out, 'w'), indent=1)


main()
