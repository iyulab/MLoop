#!/usr/bin/env python3
"""Cut a learning-to-rank set into the train and test files `rank.py` compares systems on.

The set is MSLR-WEB10K (the validation fold: 2,000 queries, one row per query holding the labels and the 136 feature vectors of
its documents), read from a Parquet file with the columns `query`, `labels`, `features`. For a split seed the queries are shuffled,
`--train-queries` of them go to the train file and the next `--test-queries` to the test file - held out by query, as a ranking
model is judged on queries it has not seen. Each file has one row per document: `qid`, `label` (0-4 relevance), `f0` .. `f135`.

  python rank-tables.py --source validation.parquet --out <dir> --seed 0 [--train-queries 600] [--test-queries 150]

Needs pandas, numpy and pyarrow.
"""
import argparse
import os

import numpy as np
import pandas as pd


def documents(rows):
    """One DataFrame row per document of the queries in `rows`."""
    frames = []
    for _, row in rows.iterrows():
        features = np.vstack(row['features'])
        frame = pd.DataFrame(features, columns=[f'f{i}' for i in range(features.shape[1])])
        frame.insert(0, 'label', np.asarray(row['labels'], dtype=float).astype(int))
        frame.insert(0, 'qid', int(row['query']))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-queries', type=int, default=600)
    ap.add_argument('--test-queries', type=int, default=150)
    args = ap.parse_args()

    queries = pd.read_parquet(args.source)
    order = np.random.default_rng(args.seed).permutation(len(queries))
    need = args.train_queries + args.test_queries
    if need > len(queries):
        raise SystemExit(f'{need} queries asked for, the set has {len(queries)}')
    train = queries.iloc[order[:args.train_queries]]
    test = queries.iloc[order[args.train_queries:need]]

    os.makedirs(args.out, exist_ok=True)
    for name, part in (('train', train), ('test', test)):
        frame = documents(part)
        frame.to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
        print(f'{name}: {len(part)} queries, {len(frame)} documents, {frame.shape[1] - 2} features, labels {sorted(frame.label.unique())}')


if __name__ == '__main__':
    main()
