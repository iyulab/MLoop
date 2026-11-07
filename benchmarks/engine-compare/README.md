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

The `Engine compare` workflow (`workflow_dispatch`) runs one job per split seed (the `seeds` input, `0,1,2` by default) and writes the summary table to the run.

`summarize` ends with a per-seed table when the rows hold `mloop` and `flaml` or `mloop-ensemble`: the sign of each split seed (`+` the first system is better, `-` worse, `=` the same value) next to the verdict, the number of runs that took longer than the budget plus the larger of 2 s and 20 %, and the held-out expected calibration error of the MLoop rows. `ensemble where run vs flaml` counts a blend where `--ensemble` was run (binary, regression) and the single model where it was not (multiclass), so its tally covers every table.

## Reading one execution against an earlier one (`regress`)

`run.py regress --base <results> --candidate <results>` answers whether the search has got worse between two executions of this harness. Two executions run on
different machines at different times, so an absolute metric would carry the runner. The rule therefore reads each execution against a reference system that
ran in the same execution (`--reference`, FLAML by default): per table the verdict of `mloop` against it, as `summarize` prints it. It reports `WATCH` for a
table that was ahead or equal in the baseline and is behind by at least two standard errors in the candidate (or when the number of tables where `mloop` is behind rose by two),
and `REGRESSION` (exit status 1) when that number rose by three or more - with five seeds a table whose true gap is nil is read "behind" about 19 % of the time, which
makes a rise of two a coin-toss-sized event across sixteen tables and a rise of three rare (about 1 % for identical code, simulated). The rule was fixed before the code was written; `test_regress.py` builds a flip, a machine that is slower for
both systems, and a new draw of noise around the same gaps, and checks the status of each. A table that sits on the boundary between equal and behind can flip on
noise alone - that is why one flip is only a watch.

The `Benchmarks scheduled` workflow runs the comparison (systems `mloop`, `flaml`, `lightgbm`; split seeds 0-4) and the lifecycle walk on a weekly schedule and reads
each comparison against the previous successful run.

## Ranking against the old AutoML (`rank.py`)

`rank.py` answers the same question as the table sweep for the ranking task, which `run.py` does not cover. For each split seed
`rank-tables.py` cuts MSLR-WEB10K (the validation fold, 2,000 queries of up to 100-odd documents with 136 features) into a train file of
600 queries and a test file of 150 other queries; `mloop train --group-column qid` and `mloop evaluate` are scored against
`automl-legacy`'s ranking experiment (the same time budget, the same files) by ML.NET's own ranking evaluator, so both numbers are NDCG at 10 with the
same gains. `summarize` reads the pair as `run.py summarize` does. Build `automl-legacy/` first; needs pandas, numpy and pyarrow.

```bash
python benchmarks/engine-compare/rank.py run --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" --source validation.parquet --seeds 0,1,2,3,4 --budget 60 --out rank.jsonl
python benchmarks/engine-compare/rank.py summarize rank.jsonl
```

`run` takes `--systems` (run one side only; a later row of the same system and seed replaces an earlier one in `summarize`) and `--mloop-name` (record
MLoop under another name, to keep two builds apart). Read the pair with the legacy search's own noise in mind: its search is multi-threaded and
bound by the clock, so repeating a seed gives another model - seed 3 scored 0.4731 and, run again, 0.4494 - and one run per seed carries about 0.02 of NDCG
at 10. A gap of 0.008 over five seeds is inside what a repeat of the legacy side alone can move; more seeds or repeated legacy runs are needed to call it.

The ranking evaluator groups consecutive rows with the same query key. A key mapping fitted on the train file alone turns every unseen test
query into the missing key and the evaluator then scores the whole test file as one list (NDCG at 10 of 0.61 instead of 0.46 on the same model) -
`automl-legacy` fits its mapping on both files for that reason.

## Learner parity and the cost of a large model

`parity/` answers two questions that a gap between systems raises. `parity.py` trains one fixed set of hyperparameters with the LightGBM
library and with ML.NET's LightGBM regressor (`MlnetLgbm`) on numeric regression tables and reports the held-out RMSE side by side — if they
agree, a gap is a matter of search, not of the learner. `sizeprobe.py` trains ML.NET LightGBM at growing sizes (trees × leaves) and reports the
model file, the time to save and load it and to score the held-out rows. Both run in the `LightGBM parity` workflow (`workflow_dispatch`).

## Where a forest gap comes from (`forest-probe/`)

When FLAML's winner on a table is a random forest or extra trees, `forest-probe` asks whether MLoop's gap to it is the forest or the settings its search can reach.
`forest.py` scores each split seed with scikit-learn's RandomForest and ExtraTrees (300 trees, library defaults) and with ML.NET's FastForest at fixed
configurations, in two groups: `reach-*` set only what the search space exposes (trees, leaves, the per-tree feature fraction) and `open-*` also set the
minimum examples per leaf, the bagged share of the rows, the per-split feature fraction and the bin count. It prints the mean, the standard error and the paired
difference to scikit-learn's forest per system. Two things to know when reading it: a configuration with no randomness left (every row bagged, every feature tried at every
split, every feature in every tree) grows identical trees, so it scores like one tree; and the binary tables' labels are written as 0/1 for the probe.

```bash
dotnet build benchmarks/engine-compare/forest-probe -c Release
python benchmarks/engine-compare/forest-probe/forest.py --mlnet "dotnet $PWD/benchmarks/engine-compare/forest-probe/bin/Release/net10.0/MlnetForest.dll" --seeds 0,1,2,3,4 --out forest.jsonl
```

`flaml` rows also record the winning settings (`best_config`, `best_loss`, `best_config_train_time`, and the per-learner best loss and config), and `flaml-lgbm` / `flaml-forest`
(named in `--systems`) restrict FLAML to its LightGBM learner or to random forest and extra trees, to attribute a gap to a learner family.

`python run.py pair --base a.jsonl --variant b.jsonl` reads a variant build of the search against a baseline run of the same system (both from the same harness, seeds and
budget): per table the paired verdict and signs, the trials each ran, the runs over budget, and the trainers that took most of the trial seconds.

`mloop-refit` (named in `--systems`) is `mloop` with the held-out share cut from 5% of the train file to 10 rows, so the model is fitted on nearly all of
the train file, as FLAML and AutoGluon fit theirs. Read a gap against them on this arm: the `mloop` arm fits on 95%. `summarize` prints it next to
`mloop` and `flaml` with `refit vs mloop`, `refit vs flaml` and `mloop vs flaml`, so the share of a gap the held-out 5% explains is one number.

`autogluon` (named in `--systems`) runs AutoGluon-Tabular with `time_limit` set to the budget and its default preset. It has no seed of its own, so a table's
split seed is the only thing that changes between its runs. The workflow installs the package only when it is named, and AutoGluon pins its own
versions of LightGBM, XGBoost and scikit-learn: read an AutoGluon dispatch against the `flaml` and `mloop` rows of the same dispatch, not against another one.

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
