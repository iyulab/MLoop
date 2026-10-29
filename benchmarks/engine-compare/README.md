# engine-compare

Measures MLoop's trainer search against FLAML and stand-alone LightGBM / XGBoost on public OpenML tables, so a change to
the search (or a claim about it) has numbers behind it.

- **Data**: 16 tables from the TabArena suite (OpenML suite 457; 8 binary, 3 multiclass, 5 regression; 898 to 53,940
  rows), fetched by OpenML data id — see `datasets.json`. Nothing is stored in this repository.
- **Protocol**: per (dataset, split seed) one stratified 80/20 split; every system gets the same train and test files.
  MLoop trains on the train file (`--time <budget> --no-promote`) and is scored with `mloop evaluate` on the held-out
  file; the other systems are scored on the same rows. Metrics: ROC AUC (binary), log-loss (multiclass), RMSE
  (regression). The split seed is separate from `mloop train --seed`, which does not change the split.
- **Systems**: `mloop` (the search is told to optimize the reported metric and holds back only 5% of the train file),
  `mloop-defaults` (the shipped defaults: an accuracy-family metric, 20% of the train file held back), `flaml` (default learners), `flaml-nogbm` (FLAML restricted to random forest, extra trees and
  a linear model — L1 logistic regression, or LassoLars for regression; the learner families MLoop has without a
  gradient-boosting library), `lightgbm`, `xgboost`
  (library defaults, no tuning).
- **Reading it**: a pair is "equal" when the mean paired difference over split seeds lies within its standard error.
  With three seeds that is coarse — trust large gaps only. Compare `mloop` with `flaml-nogbm` to see the search policy
  with the same learner families, and with `flaml` to see the effect of having gradient-boosting learners at all.
- **Trial log**: each `mloop` row carries `trial_log` — the trials the search kept (trainer, the settings it chose, metrics,
  runtime, which one was selected), read from the experiment's `trials.ndjson`. `summarize` ignores it; it is there to see
  where a search went (does the winner sit on a boundary, how often did it beat the trainer's own defaults).
- **Platform**: the LightGBM native library that ML.NET bundles exists for Linux x64 and Windows x64, not for macOS on
  ARM, where `mloop` searches without LightGBM. Run the sweep on Linux x64 for the full learner set (the `Engine compare`
  workflow does).

```bash
pip install flaml lightgbm xgboost scikit-learn pandas numpy
dotnet build tools/MLoop.CLI -c Release
python benchmarks/engine-compare/run.py run --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" --seeds 0,1,2 --budget 60
python benchmarks/engine-compare/run.py summarize results.jsonl
```

The `Engine compare` workflow (`workflow_dispatch`) runs one job per split seed and writes the summary table to the run.
