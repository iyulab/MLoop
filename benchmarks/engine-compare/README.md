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
  (library defaults, no tuning), `automl-legacy` (ML.NET's own AutoML, Microsoft.ML.AutoML 0.23.1 — the search MLoop's
  replaced it with — run for the same time budget on the same files; build `automl-legacy/` first).
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

## Learner parity and the cost of a large model

`parity/` answers two questions that a gap between systems raises. `parity.py` trains one fixed set of hyperparameters with the LightGBM
library and with ML.NET's LightGBM regressor (`MlnetLgbm`) on numeric regression tables and reports the held-out RMSE side by side — if they
agree, a gap is a matter of search, not of the learner. `sizeprobe.py` trains ML.NET LightGBM at growing sizes (trees × leaves) and reports the
model file, the time to save and load it and to score the held-out rows. Both run in the `LightGBM parity` workflow (`workflow_dispatch`).

## Averaging the best trials (`ensemble-probe/`)

`ensemble-probe` answers one question outside the product: how much does an equal-weight average of the search's best distinct trials gain over its best single model? It runs
MLoop's own search (`TrialRunner`) for a time budget on a train file, refits the best K distinct (trainer, settings) on the whole train file,
and scores the held-out file for the best single model and for the average of the first K members (probabilities for classification, predictions for regression), K = 2..maxK.
An average with equal weights is a floor for what a blend with fitted weights could gain, but it can lose to the best single model (few trials, multiclass) - so the probe also fits the weights: forward selection with replacement (Caruana et al., 2004) on the members' out-of-fold answers over the train rows (a plain five-fold split, one for every member), keeping the best prefix. Its weights never see the held-out rows, which only score it.

```bash
dotnet build benchmarks/engine-compare/ensemble-probe -c Release
dotnet benchmarks/engine-compare/ensemble-probe/bin/Release/net10.0/EnsembleProbe.dll train.csv test.csv <label> <binary|multiclass|regression> <budget-seconds> <seed> [maxK]
```

`run.py run --systems ensemble` runs it on the same split files as the other systems and writes five rows (`ens-single`, `ens-k3`, `ens-k5`, `ens-oofpick`, `ens-fwd`); `summarize` then adds a table that sets the average of the best 3 and 5 trials, the member the out-of-fold answers pick and the forward-selected blend against that same search's best single model, per table and over split seeds. `ens-oofpick` is forward selection with its weights collapsed onto one member: what `ens-fwd` gains beyond it is the blend, not the choice made on out-of-fold answers. The `Engine compare` workflow builds the probe and takes `ensemble` in its `systems` input. The forward arm refits each of the best five members five more times on the train rows, so a probe run costs more than its time budget.

Without OpenML, `bundled-tables.py <work-dir>` runs the probe on the five small tables scikit-learn ships (breast cancer, wine, digits, iris, diabetes; 150 - 1,800 rows) over split seeds and prints the same comparison - it reads the shape of the answer, not the 16-table result.

It prints one JSON object (`best_single`, each member's validation, test and `reference` metric — ML.NET's own evaluation of the same predictions, a cross-check of the
score-column order — and `average.k2..kK`). The metric is AUC for binary, log-loss for multiclass and RMSE for regression. The average costs K times the refit,
which the table does not charge; run it on Linux x64 so LightGBM is among the trainers, like the rest of this directory.
