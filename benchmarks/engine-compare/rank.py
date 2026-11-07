#!/usr/bin/env python3
"""Rank MLoop's ranking task against ML.NET's own AutoML ranking search on the same queries.

For each split seed `rank-tables.py` cuts MSLR-WEB10K (validation fold) into a train file and a test file, held out by query. Then:

  * `mloop`         - `mloop train --group-column qid` (a ranking run fits one FastTreeRanking pipeline: there is no search to give the
                      time budget to) and `mloop evaluate` on the test file;
  * `automl-legacy` - Microsoft.ML.AutoML 0.23.1's ranking experiment for the same time budget on the same train file, scored on the
                      same test file by ML.NET's ranking evaluator (build `automl-legacy/` first).

Both numbers are NDCG at 10 from ML.NET's own evaluator with its default gains (0, 3, 7, 15, 31), so they are the same quantity.

  python rank.py run --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" --source validation.parquet --seeds 0,1,2 --budget 60 --out rank.jsonl
  python rank.py summarize rank.jsonl

`summarize` reads a pair as equal when the mean paired difference over the seeds lies within its standard error (the rule of `run.py summarize`);
with three seeds that is coarse. Needs pandas, numpy and pyarrow.
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.join(HERE, 'automl-legacy', 'bin', 'Release', 'net10.0', 'AutomlLegacy.dll')


def sh(command, cwd=None):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'{" ".join(command)[:200]} failed ({done.returncode}):\n{(done.stdout + done.stderr)[-1500:]}')
    return done.stdout


def run_mloop(mloop, work, budget):
    project = os.path.join(work, 'proj')
    sh([*mloop, 'init', 'proj', '--task', 'ranking', '--label', 'label'], cwd=work)
    sh([*mloop, 'train', os.path.join(work, 'train.csv'), '--label', 'label', '--group-column', 'qid', '--time', str(budget), '--no-promote'], cwd=project)
    out = json.loads(sh([*mloop, 'evaluate', 'exp-001', os.path.join(work, 'test.csv'), '--json'], cwd=project))
    metadata = json.load(open(os.path.join(project, 'models', 'default', 'staging', 'exp-001', 'metadata.json')))
    trainer = ((metadata.get('result') or {}).get('trainer') or {}).get('name')
    return {'learner': trainer, 'ndcg_at_10': out['testMetrics']['ndcg_at_10']}


def run_legacy(work, budget):
    out = json.loads(sh(['dotnet', LEGACY, os.path.join(work, 'train.csv'), os.path.join(work, 'test.csv'), 'label', 'ranking', str(budget), 'qid']).strip().splitlines()[-1])
    return {'learner': out['learner'], 'trials': out['trials'], 'ndcg_at_10': out['test']['ndcg_at_10']}


def run(args):
    mloop = shlex.split(args.mloop)
    rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        with tempfile.TemporaryDirectory(prefix=f'rank-{seed}-') as work:
            sh([sys.executable, os.path.join(HERE, 'rank-tables.py'), '--source', args.source, '--out', work, '--seed', str(seed),
                '--train-queries', str(args.train_queries), '--test-queries', str(args.test_queries)])
            for system, measure in (('mloop', lambda: run_mloop(mloop, work, args.budget)), ('automl-legacy', lambda: run_legacy(work, args.budget))):
                row = {'seed': seed, 'system': system, 'budget_s': args.budget, **measure()}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.out, 'a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def verdict(base, other):
    """Paired reading of `base` against `other` (higher NDCG is better): ahead, equal (within one standard error of the mean difference) or behind."""
    seeds = sorted(set(base) & set(other))
    diffs = [base[s] - other[s] for s in seeds]
    mean = sum(diffs) / len(diffs)
    se = (math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) / math.sqrt(len(diffs))) if len(diffs) > 1 else float('inf')
    return mean, se, ('equal' if abs(mean) <= se else 'ahead' if mean > 0 else 'behind'), len(diffs)


def summarize(args):
    rows = [json.loads(line) for line in open(args.results, encoding='utf-8') if line.strip()]
    by = {}
    for row in rows:
        by.setdefault(row['system'], {})[row['seed']] = row['ndcg_at_10']
    for system, seeds in sorted(by.items()):
        values = list(seeds.values())
        print(f'{system:14s} NDCG@10 mean {sum(values) / len(values):.4f} over seeds {sorted(seeds)}: ' + ' '.join(f'{v:.4f}' for _, v in sorted(seeds.items())))
    if 'mloop' in by and 'automl-legacy' in by:
        mean, se, word, n = verdict(by['mloop'], by['automl-legacy'])
        print(f'mloop - automl-legacy: {mean:+.4f} +/- {se:.4f} over {n} seed(s) -> mloop is {word}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--mloop', required=True)
    r.add_argument('--source', required=True)
    r.add_argument('--seeds', default='0,1,2')
    r.add_argument('--budget', type=int, default=60)
    r.add_argument('--train-queries', type=int, default=600)
    r.add_argument('--test-queries', type=int, default=150)
    r.add_argument('--out', default='rank-results.jsonl')
    r.set_defaults(func=run)
    s = sub.add_parser('summarize')
    s.add_argument('results')
    s.set_defaults(func=summarize)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
