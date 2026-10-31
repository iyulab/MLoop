# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`mloop train` warns of a column of numbers written for people.** A feature column whose values are numbers with a thousands separator, a percent sign or a
  currency sign (`"1,234.5"`, `12%`, `$3`, `₩5,000`) is read as text, so the model gets its characters and not its size: a regression on a price written
  with its separator scored R² -0.004 where the plain number scored 0.9997, and the only trace was a quality gate refusing the model, with no word of why. The
  data check before training now names each such column, shows a value, and says to write the plain number in the file or convert it in a preprocessing
  script. A column is judged on its first thousand rows: at least five non-empty values, nine in ten of them numbers in a written form, at least one carrying
  the formatting. The label column and tasks that read no feature columns are not asked. `mloop info` warns of the same columns (a price whose values are all different
  also looks like an identifier, and is then dropped altogether), so it is known before a run.
- **`mloop train --epochs <n>`** (also `training.epochs` in `mloop.yaml`, and `epochs` in a `POST /train` request) sets how many epochs a deep-learning task
  trains for - text classification, sentence similarity, NER, question answering and object detection. Those fits run a fixed number of epochs and take no time
  budget, so `--time` cannot bound them, and the count was a constant 10: an object-detection epoch over a few hundred images takes minutes on a CPU. The
  default is still 10. A task that does not count epochs (image classification stops on its own once accuracy stops improving; tabular tasks search trainers)
  refuses the setting, as `--max-trials` refuses a task that runs no search, and `mloop validate` checks it. The training summary lists the count when it is set.
- **`mloop drift` counts the values past the training range as a signal of its own.** The PSI folds them into its two end bins and barely sees a share of a column's values leaving the range: with a tenth of them drawn three times as far out, a window of 1,000 rows reads moderate about half the time, and with a twentieth almost never. Every numeric column now carries how many of the window's values are below the training minimum and above the training maximum, and the exact chance of that many with nothing changed (`C(m, k) / C(n + m, k)` for independent draws from one distribution); the columns where it is under 5 % shared over both sides of every column are named under the table, in `range` / `rangeNotable` of `--json` and of `GET /drift`, and in the message of `trigger check`. With nothing changed some column was named in at most one window in a hundred at 200 columns (about one in twenty at a thousand); with a tenth of a column's values thrown out it is named in every window of 1,000 rows. Like every number `drift` gives it assumes independent rows: rows that arrive in blocks sharing a cause make it name columns far more often (and the PSI read those windows moderate or worse every time). It is read apart from the PSI: levels, the retraining trigger and exit codes are unchanged. Profiles written before the range was kept carry no signal. `FeatureDrift` has a new `Range` member (a constructor argument with a default, and one more value when deconstructed), and `DriftReport.RangeNotable` lists the named columns.
- **`mloop explain` shows which input columns a trained model relies on.** `mloop analyze importance` reads the data - how strongly each column relates to
  the label, whatever model is trained on it - and nothing so far read a model. `explain [experiment-id] --data <file>` scores the model on a labeled file, then
  shuffles one feature column's values among the rows at a time and scores again; the fall of the score is how much the model needs that column (permutation
  importance), in the units of the metric `mloop evaluate` reports for the task, mean and spread over `--repeats` shuffles (3), on at most `--rows` rows
  (2,000, sampled with `--seed`; the same seed gives the same numbers). `--json` writes a progress line to stderr at each tenth of a long run. A column the model did not read as a feature (excluded or ignored at training) is listed
  as not shuffled; the label and the weight column are not features. It is the question to ask when a score looks too good - which column carries it? - or a column
  is costly to collect - does the model use it? The training warning about a near-perfect held-out score, which said what to check but not where, now suggests it. Two correlated columns share what either would show alone. Binary, multiclass and regression models; `--json`
  carries `features` (`column`, `importance`, `standardDeviation`), the `metric`, its direction and the `baseline`. The table lists the 20 columns the model relies on most and says how many it left out (`--json` has every column) - on a table of hundreds of columns the rest fall the score by about nothing.

### Changed

- **`mloop drift` no longer reads a window of unchanged data as "moderate" because the table is wide or the window short.** The PSI levels (0.1, 0.25)
  are the reading for large samples. A window of 300 logged predictions against a training profile has a PSI of about 0.03 in each column with
  nothing changed, and a bin that expects a handful of rows - the missing values' - reads 0.1 on that bin alone when the window holds none of them; on
  a table of fifty columns one such window in five read "moderate", on two hundred one in two, and on a thousand nearly all (a thousand rows: none).
  The level of a column is now read against the larger of the usual threshold and `FeatureDrift.Bound` - the PSI that a window of its size would
  stay under in 95 % of windows across all the profile's columns (`DriftMeasure.NoDriftBound`; `noDriftBound` in the `--json` signal and the REST
  answer) - and the window's shares are counted with half a row more in every bin, so a bin that expects six rows and gets none is not a distribution that
  left. With nothing changed, 2-10 % of windows read "moderate" at any width - about 10 % for two hundred columns against a profile of 5,000 rows, about 4 % against one of 500 (the profile is itself a sample, which sets the rest); a move of half a
  standard deviation, a variance up by half, a strong shift in a category's shares or missing values going from 2 % to 10 % in one column is still caught in nearly
  every window of 300 rows (0.3 standard deviations: six in ten, from seven). The reported `psi` is the smoothed one and is a little under the old one on a column with
  empty bins; "significant" is unchanged in practice for a training profile of thousands of rows - it had no false alarms in the measurement (a small training set and the `input_drift` retraining condition: see Fixed).
  `FeatureDrift` has one more member (`Bound`).
- **A trainer that fails three times in a row without ever scoring is retired from the search, and a search whose every trainer is retired ends at once.**
  A run where nothing works - a native library that is not installed, a column no trainer can read - used to keep trying until the whole
  time budget was gone, and a trainer that could not run on this machine kept being proposed beside the ones that could. A trainer that has
  scored is never retired: its failures belong to the settings that failed, which the search already steers away from. The error that ends a
  search says it stopped early, names each retired trainer's failure, and no longer advises a longer time budget
  (`NoSuccessfulTrialException.AllTrainersFailed`).
- **The search keeps the top-scoring trial unless its model is very large.** It used to replace a top model of 100,000 leaves (trees × leaves) or more
  with the smallest model within one standard error of its score; an ordinary model of a few hundred trees reaches that, and on sixteen public
  tables the swap cost score (up to 0.0066 RMSE and 0.0063 AUC, twice the standard error on two tables, at 60 and 240 seconds) with no cost
  in time to give back. The rule now starts at 10 million leaves (about 200 MB on disk), where keeping the model is what costs. A model you
  trained before is unchanged; a new search can pick a larger model than it did, and `trials.ndjson` marks the trial the model was refit from.
- **`mloop analyze` and `info --analyze` say when an analysis did not run or left columns out.** DataLens answers an analysis it cannot run with an empty
  result and a warning of its own, and the empty result read as a finding: `analyze correlation` on a two-row file printed "No high-correlation pairs
  found", and on a file with one numeric column the same. The warnings that mean the analysis did not run (too few rows, too few numeric columns,
  every pair undefined, a failure), or ran on less (constant or empty columns set aside, named), are now printed and carried in `warnings` with `--json`.
  Warnings that describe the data rather than the analysis (repeated rows, a column pair that is one column, a covariance that may be singular) are not.

### Fixed

- **`mloop train --analyze-data --json` answered with nothing.** The data check is that run's whole job — it trains nothing — and it showed its findings
  only on the console, which `--json` silences, so a consumer read an empty stdout and a zero exit: the same as data with no findings. The findings are now
  one `analysis` event (`issues`, each with `type`, `severity`, `column`, `description` and `suggestedFix`; the list is empty when the check found nothing),
  and a script written by `--generate-script` is a `script` event with its path.

- **A file that does not split into columns says which line is the odd one.** "Unable to split the file provided into multiple, consistent columns ... check for
  a consistent number of columns" left the reader to find one short or long row in a file of thousands - in `train`, `info` and every command that reads the
  file's layout. The refusal now ends with the first such line - "Line 10 has 3 field(s) where the header has 4." Lines that belong to a quoted field spread over
  several lines, and blank lines, are not counted as short rows, and a file whose header is a single field (not comma-separated) gets no hint.
- **A table's blank line is not a row in the training-time checks either.** The size warnings ("only N rows", "N samples for M features"), the "no data rows"
  check and the row count the training-time estimate is made from counted a blank line as a row, while the loaders skip it: three trailing blank lines turned
  a critically small file of eight rows into one of eleven, and a file of a header and blank lines was reported as having no valid label values rather than
  no rows. They now ask the same question the loaders, `info`, `predict` and the training summary ask.
- **`mloop predict` reads a file of fewer than a hundred rows that has a blank line.** A prediction file has no label column, so predict wrote a copy with an
  empty one in front of every line - a blank line became `,`, a line of the wrong width, and among fewer than a hundred rows it failed the split of the file into
  columns ("unable to split the file into multiple, consistent columns", with no word about the blank line). Blank lines are left out of the copy, as everywhere
  else they are no row.
  `--include-features` wrote each row's features beside the prediction of the previous row from the first blank line on, because it paired the input with the
  predictions by line number and the blank line has no prediction; the blank line is now left out of the pairing too.
- **`mloop info` describes a file of one column, and `info`, `analyze profile` and the data summary `train` prints no longer count a table's blank lines as rows.** A one-column file (a series) failed with "unable to split the file into multiple columns" - the layout training splits a file by - although `analyze profile` and `detect` read it; `info` now reads such a file from its sample. Blank lines after or between the rows of a table were counted as rows with every value missing (3 rows and two blank lines reported 5 rows, a column 40% missing), while training and `sample create` skip them (the summary `train` prints said 303 rows for 300 and two blank lines, beside a label check that said 300); all three skip them too, by one rule. In a file of one column a blank line stays a missing value, the way the series commands read it.
- **`mloop detect` and `mloop analyze series` no longer close up a one-column file over a blank line.** The reader takes a blank line for nothing, so a series
  stored one reading per line with a reading missing came out shorter and every reading after the gap sat at the wrong row - `detect` named the wrong rows.
  A blank line inside a one-column file is now the same error as a non-numeric value: the series must be fully numeric with no gaps. (In a wider file the
  empty cell already failed.)

- **`mloop info --analyze` and `mloop analyze` no longer run change-point detection nobody reads.** DataLens switches on every analysis it has unless told otherwise, and change points - added in a later version than the code that builds the options - ran on each call, quadratic in the number of rows: `analyze correlation` took 10 seconds on 50,000 rows and about a quarter of an hour on a million, for four numeric columns, to show a correlation matrix. The options are now built in one place from everything off, naming what each command shows, so a switch added later stays off until it is asked for: `analyze correlation`, `distribution` and `importance` take 0.4 to 0.7 seconds on 50,000 rows and about 2.5 seconds on a million. `info --analyze` and `analyze outliers` still take seconds to minutes on large files because the local outlier factor DataLens runs there is itself quadratic in the rows (8 seconds at 25,000 rows, 34 seconds at 50,000).

- **A model trained on a few hundred rows no longer reads windows of unchanged data as drifted, and `input_drift` no longer asks for a retraining on the noise of a small
  training set.** Three things were wrong. The two bins past a numeric column's training range expected nothing, but a new draw lies past the largest (or under the
  smallest) of n training values with probability 1 / (n + 1) on each side - a window of 300 rows carries a stray value or two in most columns of a 500-row profile; the
  bins now expect that share. The training profile's shares were taken as exact, though a bin that held two or three training rows - 2 % missing values over 150 rows - or
  none, which counted as 0.0001, made a window with nothing changed read 0.1 on that bin alone; they are now counted with half a row more in every bin, as the window's
  already were (over thousands of rows this changes nothing). And the `input_drift` condition fired at the bare threshold; it now needs a column over the threshold *and*
  over the bound the levels are read against, and a condition that is over the threshold but within the bound says so in its details. Simulated, windows of 300 rows with
  nothing changed against 8 training profiles each, fifty columns: with 150 training rows 4 of 200 windows ask for a retraining (9 with 1 % categories; 0.48.0: all of 200
  in one simulation), with 500 rows none (0.48.0: 2 of 200, and 38 at two hundred columns); "moderate" or worse is read in 6 of 200 windows at 150 rows and 7 at 500
  (0.48.0: 134 at 500 rows). One column moved by a standard deviation is still caught in every window; by half a standard deviation in about as many as before (107 of 200
  against 120 at 150 rows, a count that included the false alarms). Weakest case measured: 100 training rows, two hundred columns, 1 % categories - 38 of 200 windows ask
  for a retraining. A drift answer over a training profile of fewer than 150 rows says so, and that a "moderate" reading is weak evidence.

## [0.48.0] - 2026-10-30

### Added

- **`mloop train --max-trials <n>`** (also `training.max_trials` in `mloop.yaml`, and `maxTrials` in a `POST /train` request) stops the trainer
  search after `n` trials, within the time budget. The time budget caps the clock and this caps the work; the search's seed is fixed, so with it
  set the search is the same on a fast machine and a slow one — a pipeline that trains in CI and on a laptop, or a test that must not depend on the
  runner's speed. A value below 1, or a task that runs no search (clustering, anomaly detection, recommendation, forecasting, deep learning,
  image anomaly), is refused with the reason; `validate` reports it. A trial limit replaces the automatic time estimate (which would run the
  same fixed-seed search twice): without `--time` the budget is the default ceiling, and `--auto-time` with a trial limit is refused. `TrainingConfig.SearchMaxTrials` was an internal property and is now the
  public `TrainingConfig.MaxTrials`.
- **Settings in `mloop.yaml` that nothing reads are reported.** The file has always been read leniently, so a misspelled key was dropped
  without a word — a `no_worse_than_prod` meant a promotion rule that was never judged, a `time_limit_secs` an automatic time budget. `validate`
  now lists each unknown key as a warning with its path and the nearest setting it does read, and every command that loads the file says so
  once per run. Model names, column names and other keys of your own are never reported.
- **`mloop analyze series <data> [--column <col>] [--max-lag N] [--around-trend]`** reads a numeric column as an ordered series
  and reports its structure for choosing a model order or a transform: the autocorrelation and partial autocorrelation functions
  with the white-noise band and Bartlett's limits, the augmented Dickey-Fuller test (H0: a unit root) and the KPSS test
  (H0: stationary) read together at 5% (`stationary`, `non-stationary`, or `unsettled` when the two disagree or neither rejects),
  and the dominant period. The tests are evidence, not advice: whether to difference, detrend or leave the series alone stays the
  caller's choice, so no step is proposed. `--around-trend` tests around a linear trend instead of a constant level. `--json`
  answers the usual analyze envelope; a series with fewer than 10 observations, a gap or a constant value is refused with the
  reason.
- **`mloop threshold --precision <p> --guarantee <δ>`** chooses a decision threshold whose precision is certified, not merely
  observed. The usual choice is the lowest threshold that reached the target on the file — the luckiest reading of its rows,
  whose true precision lands below the target in about half the files (measured on synthetic calibrated scores). With
  `--guarantee`, thresholds are tested from the strictest down with an exact one-sided binomial test at risk δ, the walk
  stops at the first it cannot reject, and the last rejected one is chosen: with probability at least 1 − δ its true precision
  reaches the target. It needs enough rows (29 right answers for 0.90 at δ = 0.05) and is refused, writing nothing, when the
  file cannot carry it; a precision of 1 is never certified. `decision-threshold.json` gains a `guarantee` block (`delta`,
  `hits`, `rows`, `lowerBound`). Without the option nothing changes.
- The guide's scripting section says plainly that the C# files under `.mloop/scripts/` run, with the user's permissions, whenever the
  command that uses them runs — training a project you cloned runs its scripts, and nothing asks first.
- Dependency updates: FilePrepper 0.13.0 → 0.13.1 and DataLens 0.20.0 → 0.21.0 (UInsight 0.38.0), which carry the series statistics above.

### Changed

- **The trainer search keeps the training rows in memory instead of parsing the file again on every read.** A trial fits its pipeline once per fold and each fit
  reads the rows several times; from a CSV every read parsed them again, so a linear, SDCA or forest trial that needed a fraction of a second of
  learning spent several on reading. On sixteen public tables the same minute holds 28 % more trials on average (up to 2×), with no table's score worse
  by more than its standard error's double; a gradient-boosted tree trial, whose cost is the learning, is unchanged. The rows are kept when
  rows × columns is at most 50 million cells (a few hundred megabytes); a larger table is read from its source as before.
- **`drift` and the retraining trigger compare from 300 logged predictions, not 100.** A column is cut in ten bins, and with no change at all the PSI of
  a window of n rows is about 9 ÷ n. At 100 rows ten columns read "moderate" in nearly every window and "significant" — the level the trigger acts on —
  in one window in twelve, thirty columns in one in four, a hundred columns in more than half (simulated, no drift); at 300 rows a hundred
  columns reach "moderate" in one window in fifteen and never "significant". A window or a slice under 300 rows now answers "not enough
  predictions" instead of a level.
- **A class limit in a `gate:` policy must name its `max`.** A limit written without one (or with the key misspelled) used to read as 0% and
  refuse every model without saying why. `validate` now reports it as an error, and a promotion judged against such a limit is refused with
  the reason stated.
- **LightGBM's search may try as few as 2 examples per leaf (was 20).** The lower bound of `MinimumExampleCountPerLeaf` was LightGBM's own default and the floor of the declaration this space was ported from; FastTree already searches down to 2. The starting value stays 20. On 16 public tables (3 splits, 60 s) the change left 14 tables unchanged and improved two (wine quality RMSE -1.3%, credit-g AUC +0.0025, both near the noise of three splits) and worsened none, so it is taken as a widening that costs nothing measured rather than a proven gain. A seeded search that picks LightGBM follows a different path from this version on.

- **A gate that sets the candidate against production on its own evaluation set (`gate.data` with `gate.no_worse_than_production`) now decides the promotion, in place of the comparison of recorded scores.** The recorded scores are each model's score on the held-out part of the data *it* was trained on; once the data has moved they describe different distributions, and the model that has stopped fitting keeps the better-looking number — a model retrained on the new data was kept in staging with `it does not beat production on accuracy: 0.855 vs 0.89` while the row-by-row comparison on the new data said the candidate was better (error rate 0.12 against 0.47). A project with such a gate is unaffected for a candidate production beats on the gate data (still refused, whatever it recorded) and gains the promotion of one it does not. A project without one keeps the recorded-score rule; the reason it prints (`keptInStagingReason`, `mloop train`) now says the two numbers are each model's own held-out score and names `mloop evaluate <exp> <data> --against production` and the gate as the way to compare them on the same rows.

- **⚠ Breaking: a binary classifier's prediction CSV calls the raw trainer score `RawScore`, not `Score`.** `Score` meant two
  values — `--json` and `POST /predict` answer the positive-class probability under `score`, while the CSV wrote the trainer's
  raw margin under `Score` (for example −3.43 beside a probability of 0.335) and the probability under `Probability`. The
  probability is still `Probability`; only the margin column is renamed, and it keeps its place between `PredictedLabel` and
  `Probability`. Migration: a script that reads `Score` from a binary classifier's CSV and wanted the probability should read
  `Probability` (or `score` from `--json`); one that wanted the margin should read `RawScore`. Other tasks are unchanged — a
  regression's `Score` is its prediction on every path, and multiclass `Score.<class>` columns are as before.
- The `docker-compose.yml` that `mloop docker` writes says why the `./models` mount exists and that its host folder must be one
  the Docker VM shares: an unshared folder mounts empty, hides the models the image was built with, and `/predict` answers
  "No production model found".
- Dependency patch updates: `Microsoft.Extensions.*` and ASP.NET Core packages 10.0.13 → 10.0.14, `System.CommandLine` 2.0.13 → 2.0.14.
- **`mloop train --task image-anomaly --bank-size N` keeps a coreset, not a random subsample.** The N patches are chosen greedily —
  each one the farthest from those already kept (PatchCore's k-center choice, measured after a random projection to 128 numbers) —
  so a texture seen on a few pictures survives a small bank where a random draw mostly loses it; `selection` in
  `image-anomaly.json` says `coreset` (models trained before keep `random` and are read as before). On synthetic textured pictures
  with three rare normal textures, a bank of 1,024 of 46,080 patches scored AUROC 0.975 as a coreset against 0.835 as a random
  subsample (all patches: 0.982); on pictures without rare textures the two were within one missed defect. Training costs more:
  138,240 patches down to 15,360 took 12 s end to end on 30 threads, 0.8 GB resident. From more patches than fit in memory at
  once (about 2 GiB of numbers) the bank is still a seeded random subsample, and training says so.
- **`mloop compare` with no experiment ids and no `--name` compares the default model's experiments** — as `list`, `feedback`,
  `trigger check`, `drift` and every other command that takes a model name already did; it used to refuse with "Model name
  is required". A project with no experiments answers an empty comparison, not an error.

- **`POST /predict` tells the caller what a row left unsaid.** A feature column a row does not carry, and a value in a numeric
  column that is not a number, were read as missing and the model answered as confidently as for a complete row. The response's
  `warnings` now names each such column once, with how many rows are affected (and an example of the unreadable value); the
  prediction is still made — a missing number stays a missing number.
- **`mloop promote` of the experiment already in production is a stated no-op** (`status: "unchanged"` under `--json`): no backup
  of the model onto itself, no new entry in the promotion history, no confirmation to answer. `--force` promotes it again.
- **`GET /status` no longer reports a time budget the project never set** — an unset `time_limit_seconds` is left out of the
  model's `config` entry (it was 300), as `mloop status` says "auto".

### Fixed

- **`drift` sees a column leave the range it was trained on.** A numeric column is cut in ten bins that are open at both ends, so values far past the
  training maximum sat in the end bin beside the rows that were there — and when the maximum was tied across many rows (a capped column), a window
  that had all moved past it changed that bin's share very little and read as moderate at most. The profile now keeps the training minimum and maximum
  (profiles written before read as they did) and counts the rows below and above the range as bins of their own, reported as `< min` / `> max`;
  a constant column that takes a new value drifts.
- **`drift` no longer reports a shift in inputs that did not move when a column's values differ only below single precision.** The model reads every
  number as a single-precision float, and the prediction log records that value — a training file's `0.999999993939` is the `1` in the log, and
  `15.0013` written as `15.001299999999999` is `15.0013`. The profile's cut points were compared in double precision, so rows drawn from the
  training data fell in other bins than the ones the profile counted them in: a wind speed in steps of two and a ratio column rounded to nine
  digits each read as significantly drifted (PSI 0.6 and 12), and the retraining trigger reads the same level. Values and cut points are now
  compared as the model reads them; profiles already on disk read the same way.
- **A model promoted while `mloop serve` answered could fail the requests in flight.** `promote`, `rollback` and `auto-promote` wrote the
  production files in place — `model.zip` was removed and rewritten, `metadata.json` truncated — so for the length of the copy a prediction found
  no model (HTTP 404 `No production model found`) or a missing file (500). Files other processes read (the production slot, the decision
  threshold, the calibration) are now replaced through a temporary file beside them and renamed over the old one, so a reader sees the old
  content or the new, never neither. Found by a new lifecycle check under load: sixteen clients at once, a promotion and a rollback while eight
  clients keep predicting.
- **`train` printed its "Removed excluded column(s)" line twice.** The training and test slices each load the file with the exclusions
  already decided, and each announced them again. The decision is announced once, where it is made.
- **`compare` recommended experiments that `promote` would refuse.** It ranks by metric, and a model that answers one class for every row can
  lead on accuracy; the recommendation named it anyway, and the `promote` it suggested was refused. The recommendation is now the best
  experiment the promotion gates accept (the fixed quality gate and the project's `gate:` policy), a line names each one ranked ahead that
  they refuse and why, and `--json` carries `recommended` and `held` beside `best` (the metric's first, unchanged). `promote --best` likewise
  sets aside a pick the project's `gate:` policy refuses and takes the next best, as it already did for the fixed gate.
- **A trial the time budget cut off could be recorded as a failed trial.** When a trainer's fit was stopped by the budget and the stop
  surfaced as an exception other than a cancellation, the search listed that trial with its error and no metrics (in `trials.ndjson`, the
  leaderboard and the trial events) as if the trainer could not fit. A failure after the budget has been cut is now the cut: the trial is
  dropped and the search ends with the best it has. A trainer that fails before the cut is still recorded as failed.
- **A binary trainer without a probability whose search kept no out-of-fold answers had its probability calibrator fitted on the test
  rows.** The reported metrics then came from rows the calibrator had been fitted on. The calibrator is now fitted on the training rows
  and a warning says so. The usual path — the calibrator fitted on the kept trial's out-of-fold answers — is unchanged.
- **The "no successful trial" error no longer names AutoML.** Its message now says the trainer search finished without completing a
  trial; the cause it names (budget too small, or every trial failed) is the same.
- **`mloop info` called a two-value string column "Text | Text Feature" while training one-hot encodes it.** Without `--label` the
  last column is handed to ML.NET's type inference as the label and left out of it, so that column was read from a 200-row
  sample instead. The type and purpose now come from the decision training makes (`Categorical Feature` for a category,
  `Text Feature` for language), asked of the whole file; a column the user named as the label, typed, or training drops is read
  as before.
- **`mloop docker` for an image-anomaly project left the container unable to score.** The generator switched on ONNX Runtime and the
  encoder mount only for models that read a text column through an encoder; a project holding an image-anomaly model got both
  lines commented out, and its container answered every picture with "Encoder … is not installed". It now reads the encoder
  each model needs from the record beside it, text and image alike, and turns the runtime install and the `/encoders` mount on
  (the usage text names the `mloop encoder install` lines to run on the host). Checked by building the generated image and
  scoring pictures through `POST /predict` in the container with the host's encoders mounted; without the mount it answers 409 naming the missing encoder.
  For an encoder the catalog does not list (one added with `mloop encoder add`), the usage text now says to add it, not to `install` it.
- **An embedding no longer leaves the server's cores busy after the answer is sent.** ONNX Runtime's intra-op threads spin while
  idle by default; every encoder session (`POST /embed`, `mloop embed`, image-anomaly scoring, text encoders) now turns that off
  (`session.intra_op.allow_spinning=0`, made in one place — `OnnxSessionOptions`). Measured on a 30-thread Apple-silicon machine
  with a small image encoder: the CPU time the server spent after the response fell from a median 145 ms to 18 ms per
  `/embed` request, against a median call time of 37 ms → 40 ms (patch mode 38 → 49 ms, post-response CPU already low there).
  A caller that uses the same machine right after the answer no longer competes with the server for the cores.
- **`mloop predict --log` logged no confidence.** The prediction log entry took `null` whatever the output file's `Confidence`
  column said (a served prediction's entry already held it); `mloop logs --json` now shows it.
- **The "no input column in common" refusal listed columns training had left out.** Its expected-column list and its overlap check
  now use the columns the model reads, not a timestamp or identifier it dropped — so a row carrying only such a column is refused,
  and the message no longer asks the caller to supply it.
- **The prediction preview's "not shown" note no longer says every column is in the saved file** — the label column of an input
  that had one is not (the prediction takes its place).
- **The Swagger UI is served at `/swagger`**, where the `serve` banner and the guide say it is (it was at `/`, and `/swagger`
  answered 404).
- **A valid token without the admin role gets a 403 that says what the endpoint needs** (the admin role, `mloop token --role
  admin`) instead of an empty body, as a missing token gets a 401 that says how to mint one.
- **`mloop docker` mounted encoders from `~/.mloop/encoders` whatever the host used.** The generated `docker-compose.yml`
  and the `docker run` line printed for a project whose models read text through an encoder now mount the directory the host
  keeps its encoders in: the compose file takes `${MLOOP_ENCODERS:-~/.mloop/encoders}` (so a host that moved its encoders
  with `MLOOP_ENCODERS` mounts them, and one that did not gets the same default as before), and the printed `docker run`
  line and install hint name the host's actual encoder directory. Before, such a host mounted an empty directory and the
  container refused to load the model.

## [0.47.0] - 2026-10-27

### Added

- **`image-anomaly`, a task for pictures of normal items alone.** `mloop train datasets/normal --task image-anomaly` keeps
  the patch vectors of the normal pictures (the vectors `mloop embed --patches` gives) as a bank — all of them, or
  `--bank-size N` of them as a seeded subsample — with the encoder's identity beside it, and scores held-out normal pictures
  against it for a baseline. `mloop predict <pictures>` gives each a score (its strangest patch's cosine distance to the
  nearest normal patch) and the 16 × 16 patch distances over the part of the picture the encoder reads, as JSON;
  `mloop evaluate <exp> <folder>` takes normal and anomalous subfolders and reports the AUROC and the anomalies missed at a
  5 % and a 10 % false-alarm rate; `POST /predict` answers pictures sent by content the same way. It needs no label, no
  time budget and no new runtime or package; a model refuses to score with another encoder than it was built with. The
  task count is 16.
- `benchmarks/engine-compare` and the `Engine compare` workflow (`workflow_dispatch`): MLoop's trainer search against
  FLAML and stand-alone LightGBM / XGBoost on 16 public OpenML tables, from the same train and test files, with the
  protocol and how to read it in the folder's README. Nothing is stored in the repository — the tables are fetched by
  OpenML data id.
- `--json` on `mloop init`, `mloop feedback add` and `mloop sample create` — the last commands that handed a caller
  prose. Stdout is one document at every exit (`init`: `project`, `path`, `model`, `task`, `label`, `kept`;
  `feedback add`: the recorded `predictionId`, `actualValue`, `source`, `weight`; `sample create`: `outputPath`,
  `sampled`, the totals and, from logs, `withFeedback` and the label/weight columns), a failure is `{"error": …}`
  with exit 1, and the narration goes to stderr.
- `NOTICE`, carrying the MIT notice of the ML.NET AutoML code the trainer search's tuner and schema proposer are
  ported from and the papers behind them. It ships in the `MLoop.Search` package.

### Changed

- `mloop threshold --precision` says how much the file proves. The threshold is the lowest one that reached the
  target on that very file, so its measured precision is the luckiest reading of the rows it answers; the output now
  adds how many rows it answers as the class and the precision they certify at 95 % confidence (one-sided Wilson
  bound, weighted rows counted as their effective number), and warns when that is more than five points below the
  target — about a hundred answered rows at a target of 0.9. `decision-threshold.json` and `--json` carry `answered`,
  `classRows` and `precisionLowerBound`; a threshold written before this release reads as without them. The
  threshold chosen is unchanged.
- **The trainer search tries a tree trainer's own defaults before it starts walking.** The search began every trainer
  at a deliberately cheap point (4 trees of 4 leaves) and walked outwards a tenth of the range at a time, so a trainer's
  library defaults were never tried: on a 1,000-row table the search ended at an RMSE of 6.0 after 240 seconds where
  ML.NET's own FastTree defaults give 3.8 in a fifth of a second. FastTree, FastForest and LightGBM now start from
  their defaults (100 trees; FastTree 20 leaves at rate 0.2, FastForest 20 leaves on 70% of the features, LightGBM 31
  leaves at rate 0.1) as the trials right after the cheap start — a quarter of the trees first, then all of them, so a search's first
  trial is still its cheapest and a slow machine, whose budget the whole defaults of a three-class one-versus-all
  trainer can fill, still gets the quarter — and the local search begins from the best of them (FLAML's
  `points_to_evaluate`). Over 16 public tables × 3 splits at a 60-second budget, against the search before the
  change: better on 9, equal on 6, worse on 1 (`benchmarks/engine-compare`); the regression tables gain most
  (RMSE 6.05 → 4.17 on concrete strength, 2.00 → 1.35 on airfoil noise, 595 → 546 on diamonds). The tuner's trace
  against ML.NET AutoML's is unchanged when no starting point is given.
- **A classifier's search optimizes a smooth score unless a metric is named.** With `metric: auto` (what `mloop init`
  now writes for binary and multiclass tasks, and what an absent `--metric` means) a binary search optimizes AUC and a
  multiclass one log-loss, where it optimized accuracy: accuracy at one threshold is flat between models that rank
  rows differently, and choosing by a smooth score found the better model by accuracy too (public tables, equal time
  budget: ahead on accuracy on 4 of 7, behind on none) as well as by AUC and log-loss (ahead on 6 of 10 against the
  accuracy search, behind on none). Promotion and the quality gate are unchanged — they still read the task's primary
  metric — so what a promotion compares does not move. A project that names `metric: accuracy` keeps optimizing it.
- **Breaking — a short option letter now means one long option across the whole CLI**, and `--model` means a model
  *file* wherever it appears. Before, `-f` was `--force`, `--from` or `--feedback`; `-l` a label, a limit or a log
  flag; `-m` a model name, a model path or a metric — so a letter learned on one command did something else on the
  next (`ShortOptionMeaningContractTests` now reads the live command tree and refuses a second meaning). Moving over:
  - the model *name* is `--name` / `-n` on every command: `feedback list|metrics`, `sample create|stats` and
    `trigger check` took `--model` (and `-m`) before — use `--name`. `--model` stays the path of `predict --model`
    and `encoder add --model`;
  - these long options lost their short letter (the long form is unchanged): `trigger check --accuracy|--feedback`,
    `list --all`, `info --analyze`, `train --balance|--metric`, `compare --best|--metric`, `update --check`,
    `serve --detach`, `sample create --from|--rows|--strategy`, `feedback list|add --limit|--prediction-id|--source|
    --actual-value`, `logs --limit`, `predict --log`, `token --subject`, `prep plan --columns`.
  Kept: `-n` name, `-o` output, `-l` label, `-t` task, `-m` model file, `-f` force, `-p` port, `-v` verbose, `-q` quiet,
  `-w` weight, `-e` experiment, `-d` data, `-c` column, `-i` input, `-r` role.

### Fixed

- A type override that names no column of the data — `columns: {typo_col: {type: ignore}}`, a misspelling or a column
  the file lost — was accepted without a word and had no effect. `mloop train` now warns, naming the columns, so an
  `ignore` that was meant to keep a column out of the model is not taken as having done so.
- `mloop train` ended with "Standard data file(s) not used: test.csv" in every project `mloop init` made — the file
  `data.test` declares and `mloop evaluate` reads. The files evaluate and predict read (`datasets/test.csv`,
  `datasets/predict.csv`, or where `data.test` / `data.predict` point) are no longer counted as unused.
- A numeric column declared `type: ignore` (or dropped with `mloop features select --drop`) was not ignored: the
  loader had folded it into the merged `Features` vector, so no column of its name existed for the ignore to apply
  to — the model read it, nothing said so, and the saved schema (which marks it excluded) made `mloop evaluate`,
  `predict`, `threshold`, `calibrate` and `export onnx` drop it from every file, so a model that trained and
  promoted could not be read afterwards (`Column 'Features' item type 'Vector<Single, N>' must be … the same size as
  … 'Vector<Single, N-1>'`). Since 0.40.0. An ignored column is now loaded as a column of its own, training logs it
  (`Ignored column(s) (override)`), and the model does not read it. A model trained with such a declaration before
  this release read the column and cannot be read by those commands: retrain it. A column that really carried the
  signal will no longer — the new model's scores can differ.
- A column a prep step declares — `normalize`, `scale` or `fill-missing` with `method: mean` — could not be read after
  training: `mloop evaluate`, `predict`, `threshold`, `calibrate` and `export onnx` failed with `Could not find input
  column 'a'`, and training warned that it could not profile the model's answers. The model's featurizer reads that
  column by name, so training loads it as a column of its own; nothing recorded that, and every reader loaded it as a
  slot of the merged feature vector. The saved schema now marks such a column (`readByName`), file readers and
  `POST /predict` read it as training did. A model trained before this release has no such mark: retrain it.

## [0.46.0] - 2026-10-26

### Added

- `mloop train --json` ends a run with a `promotion` event after `result`: `promoted`, and `keptInStagingReason` when
  the experiment stayed in staging (the quality gate, the gate policy, not beating production, a post-train hook,
  `--no-promote`) — the same facts, by the same judgment, a `POST /train` job reports. A stream consumer could not
  tell a promoted model from one held back before.
- Text encoders: a pretrained sentence encoder turns text — Korean and some fifty other languages — into one
  L2-normalised vector per text, through the encoder store image encoders use (`MLoop.Core.Embedding`:
  `TextEmbedder`, `TextPreprocessor`). An encoder definition says what it reads (`kind`: `image`, the default, or
  `text`); a text encoder keeps its SentencePiece tokenizer model (`tokenizer.model`) beside its graph, both
  SHA-256s recorded, and its vectors are the mean of the encoder's output over the text's tokens. The catalog gains
  `multilingual-minilm` (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 384 dimensions, 128 tokens), pinned
  to one revision of its repository. A text longer than the encoder reads is counted (`truncated`), not cut
  silently. Tokenization follows the encoder's own tokenizer, including one unknown piece for a run of characters
  the vocabulary lacks (`Microsoft.ML.Tokenizers` gives one per character).
  `mloop embed --text <text>` (repeatable) and `--text-file <file>` (one text per line) embed texts;
  `POST /embed` takes `{"texts": [...]}` (at most 256). `mloop encoder add` takes `--tokenizer` for a text
  encoder. The encoder in an answer carries `kind` and, for text, `tokenizerSha256`; a text answer carries
  `truncated`. Without `--encoder` / `?encoder=`, the one installed encoder of the kind asked is used, and an
  encoder of the other kind is refused (409).
- A text column can be trained on through a text encoder: `type: text` with `encoder: <id>` under the model's
  `columns:` in `mloop.yaml` (binary classification, multiclass classification, regression). Training embeds the
  column's texts once and the trainers read their sentence vectors (`<column>.embedding`) beside the column's own
  n-gram features, and two columns of one encoder also their cosine (`<first>~<second>.cosine`) — a regression over
  two such sentence columns is a sentence-similarity model for Korean and other non-Latin text. The text
  classification and sentence similarity warnings for unreadable text name this route. The experiment's `report.md`
  names each encoded column and its encoder. The experiment records the encoder
  beside its model (`text-encoders.json`, copied on promote), and predict, evaluate, calibrate, threshold and
  `mloop serve` read the column's text through the same encoder. A model whose encoder is missing, or installed with
  other weights, tokenizer or preprocessing, refuses to score and says which. `mloop validate` refuses an encoder on
  a column that is not `type: text` or for another task; `mloop export onnx` and `portable:` refuse such a model.
  `mloop docker` generates, for a project whose models read text through an encoder, the ONNX Runtime install, the
  encoder directory variable and its mount (commented otherwise), and names the encoders to install on the host.
- An experiment keeps the warnings its run raised — about the data, the run and the trained model, as `mloop train`
  printed them — as `warnings` in its `metadata.json` and a «Warnings» section of its `report.md`, failed runs
  included. Before, they were on the terminal and nowhere after.

- Training a text model (text classification, sentence similarity, NER, question answering) warns, per text column,
  when more than 5% of its characters are ones the model cannot read, with the share. ML.NET's text trainers read
  text through an English tokenizer that drops every character above U+00FF on the way in, so Korean, Chinese,
  Japanese or Cyrillic text reaches the model as nothing (82–89% of the characters on Korean news, dialogue and
  sentence sets) — the README said such text trained "from English sub-word pieces". Before, only question
  answering's answers were checked. For text classification the warning names the tabular classifier, which
  featurizes text by word and character n-grams over every character (on a Korean four-class sentence set: macro
  accuracy 0.72 against text classification's 0.25).

- `ICsvHelper.ReadRowsAsync` and `WriteRowsAsync` (preprocessing scripts' `context.Csv`): stream a CSV's rows as
  arrays in header order, under the reading rules `ReadAsync` has, and write them back — for a file too large to
  hold as `ReadAsync`'s list of dictionaries; scripts compile against `System.Linq.AsyncEnumerable`, so LINQ works on
  the stream (GUIDE, Data Preprocessing). A class implementing `ICsvHelper` itself must add both.
- `mloop evaluate` and `POST /evaluate` warn when the file is the one the experiment was trained on — the
  same content as its recorded `dataFileHash`, whatever the file is called: most of its rows were in the
  training partition, so the test metrics overstate how the model does on unseen rows.

- `mloop embed --patches` and `POST /embed?patches=true` add each image's patch vectors from the same encoder run —
  one unit vector per patch of the centre crop (`dinov2-small`: 16 × 16 patches of 14 pixels), row by row, with the
  grid and the crop the grid covers in the image's own pixels. One vector per image averages a small local difference
  away under normal variation; a vector per patch keeps it and locates it. At most 8 images per call with patches. An
  encoder definition declares its patch grid with `patchSize`; an installed catalog encoder without it takes the
  catalog's.

- The encoder identity in every `mloop embed --json` and `POST /embed` answer carries `preprocessor` — the revision of
  the arithmetic that turns an image into the encoder's input (`pillow/2`). The same weights give other vectors when it
  changes, as 0.45.0's handling of translucent pixels did under an unchanged `sha256`; vectors are comparable only when
  `id`, `sha256` and `preprocessor` all match.

- `predict --json` and `POST /predict` name the answering model by its content: `modelSha256`, the model file's
  SHA-256, beside `experimentId` — which is unique only inside one project. (`predict --json` for a forecasting model
  now carries `experimentId` too.)

- `POST /predict` answers an object-detection model: images by content (`{"images": [base64…]}` or a multipart form,
  as for image classification) in, and for each image its objects — label, score and box in pixels — named by the
  image's SHA-256 and logged under it with a `predictionId`. Rows sent to such a model are refused with how to send
  images.

- Each trial's record (`trials.ndjson`, `leaderboard.json`) carries `validation.standardError`: how far each metric's
  mean could move on other rows — the margin within which the search keeps a smaller model. A holdout's (one fold, no
  spread) comes from its slices.

### Changed

- FilePrepper 0.13.0 and DataLens 0.20.0: a table is held by column, so `mloop info` and `mloop analyze profile` on a
  million-row, six-column CSV peak at about 1.2 GB, and 2 million rows at about 2.3 GB. `docs/GUIDE.md` states the envelope and the sampling route for a file larger than memory allows.
- Feedback names the experiment that answered (`experimentId`, copied from the prediction log), and the accuracy
  retraining condition judges the production experiment's answers alone. Before, every answer under the model name
  counted: after a promotion the replaced model's errors were read as the new one's accuracy, and a shadow
  candidate's (`predict --experiment`) as production's. Feedback recorded before this release is read against its
  prediction's log line; only what no log line records any more (pruned) is left out, and the details count it. `feedback list --json` and `feedback metrics --json`
  carry `experimentId` (and `unattributedFeedback`).

- The accuracy retraining condition (`mloop trigger check`, `GET /trigger`) recommends retraining only when the
  drop is established: at least 30 rows of feedback, and the 95% Wilson interval's upper end below the threshold.
  Before, the point estimate decided — three answers with two wrong read as 33% accuracy and recommended
  retraining. The details say which case holds (too few to judge, below on its face but not established, below).
  Feedback metrics gain `effectiveSampleSize` — the row count, or Kish's (Σw)² ÷ Σw² with weights, which is the n
  the interval uses.

- **Breaking** — `mloop serve` limits requests **per caller** (the token's subject), not per address:
  `Serve:RateLimit:PermitsPerMinute`, default 600, `0` for none; `/health` is not counted; a refusal is `429` with
  `Retry-After` and an `error` naming the setting. Before, every caller behind one address shared 60 predictions and
  100 requests a minute, so a backend predicting faster than one image a second lost predictions to `429`; the
  address also came from a client-supplied `X-Real-IP` header. Migration: the `IpRateLimiting` section is no longer
  read (the server warns at startup when it is set) — set `Serve:RateLimit:PermitsPerMinute` instead. The
  `AspNetCoreRateLimit` dependency is gone; the limiter is ASP.NET Core's own.

- Dependencies: Spectre.Console 0.57.2, System.CommandLine 2.0.13, Microsoft.CodeAnalysis.CSharp (scripts) 5.9.0,
  Microsoft.Data.Analysis 0.23.1, Swashbuckle.AspNetCore 10.3.1, System.IdentityModel.Tokens.Jwt 8.23.0, DotNetEnv
  3.2.0, YamlDotNet 18.1.0 (mloop.yaml reads and writes as before), Microsoft.OpenApi 2.13.0 (the 2.x line
  Swashbuckle is built against — 3.x still breaks `/swagger/v1/swagger.json`).

- A binary model's probabilities are calibrated when it is trained. The search keeps the chosen trial's answers on the
  rows each fold model did not see, fits a Platt calibration to them and saves it beside the model
  (`calibration.json`, `fitOn: "out-of-fold (training partition)"`), where predict, serve, evaluate, threshold and
  export already apply one. A trainer's own probability is not fitted to the data — FastTree's is a fixed sigmoid of
  its score, so a model of a few trees that ranked every test row right answered 0.66 where it should have answered
  0.99, and a caller that routes on `confidence` read it as unsure. The ranking (AUC) is unchanged; the answer is the
  calibrated probability's at 0.5, and the run's metrics — accuracy, F1, log loss, ECE — are the calibrated model's.
  `mloop calibrate` refits it on rows of your own; `mloop calibrate --clear` returns to the trainer's probability.

- A multiclass model's class probabilities are calibrated the same way, by a temperature fitted on the out-of-fold
  answers. One-versus-all trainers normalize fixed per-class sigmoids, and their probabilities were far from how often
  they are right (top-label ECE around 0.4 where a maximum-entropy model scored under 0.001). A temperature keeps every
  row's answer and order of classes: accuracy does not move, the log loss and the ECE do.

- `mloop sample create --from` reads a relative path as the project's, as every other command does — it used the
  current directory, so the same `--from datasets/train.csv` found the file from the project root and not from
  `datasets/`. Outside a project, where it also works, a relative path is still the current directory's.

### Removed

- `mloop preprocess --incremental` (with `--skip-hitl` and `--auto-approve`) and the pattern detectors it read from
  `.mloop/scripts/detectors/` (`IPatternDetector`). Rule discovery never reached application — every rule type answered
  "not implemented", and a run that found rules wrote no cleaned data and exited 1. Transform data with a script in
  `.mloop/scripts/preprocess/` (`mloop preprocess`) or declare steps with `mloop prep plan` / `mloop prep run`; patterns
  in the data are reported by `mloop info --analyze` and `mloop analyze`. `mloop validate` warns when a detectors folder
  still holds scripts.
- `precision`, `recall` and `totalPredictions` from `mloop feedback metrics --json` (and `FeedbackMetrics`). The first
  two were never computed — always `null` — and `totalPredictions` repeated `totalFeedback`; the count of logged
  predictions is `mloop sample stats`. Per-class precision and recall on labeled rows are `mloop evaluate`.

### Fixed

- `mloop serve` says the server started only once it answers `/health`. It used to say so as soon as the server
  process launched: on a port another process already held, the new server exited moments later — after
  `--detach` had reported it running in the background with its PID and exited 0 — and in the meantime the other
  process answered requests in its place. A port in use is now refused before starting
  (`Port <n> is already in use`, exit 1), and a server that exits while starting fails `--detach` with its exit code.

- `mloop serve --host` takes only a host name or address (or `*`): the value is part of the API process's command line,
  so a host with spaces in it added arguments of its own — another `--urls`, a signing key.
- An error or warning naming a path or name the user gave with square brackets (`data[1].csv`) printed
  "Unbalanced markup stack" — or failed with an unhandled exception — instead of the message: the text was read
  as console markup. Every value interpolated into an error or warning line is now shown as given — the lines take
  their `$"…"` as markup in the literal and text in the values, so a new message cannot forget to escape one.
  Progress lines and tables that show the data's own names — column names in `mloop info`, the label, a data or
  output file, a class value — show them as given too: `mloop info` on a file with a column named `score[1]` stopped
  at "Encountered malformed markup tag".
- `mloop train <file>` with a file that does not exist said "No data file specified and datasets/train.csv not
  found"; it now says the file was not found, as `--data` does.
- An encoder's definition file no longer carries `grid`, which is derived from its patch size.

- A `POST /train` job's `warnings` carry every warning training raised, as `mloop train` prints them — about the
  data (a row-number column, text the model cannot read, a class too small) and the run — followed by the
  performance diagnostics'. The job reported only the diagnostics', so a client training through the API never
  saw the rest. Each job collects its own: a prediction served meanwhile adds nothing to it.

- `mloop logs` without `--name`, `GET /logs` without `name`, and every reader of more than one model's logs return
  the newest entries across models. Reading stopped at the first model directory that filled the limit, so another
  model's newer predictions were left out — which model depended on the order the directories were listed in.
- Prediction logs and feedback are read a line at a time and only the newest `limit` entries are held: `mloop logs`
  (default 20) read and parsed the whole day file, and recording feedback (`mloop feedback add`, `POST /feedback`)
  read every line of each day file to find one prediction. Measured on a model logging 80,000 predictions a day
  (20 MB day file): `logs --limit 5` 282 → 83 MB, `feedback add` 136 → 85 MB. `logs --json` writes its document as
  it is serialized.
- `mloop sample create --model`, `mloop sample stats` and `POST /samples` read the logged predictions as a stream and
  hold only the sample: every prediction over the retention period was held with its inputs parsed before the
  strategy chose its rows. Measured on 800,000 logged predictions (ten days of 80,000): 883 → 104 MB, 2.7 → 1.8 s,
  the same rows for `low-confidence`. `feedback-priority` draws uniformly among the rows with feedback when they
  outnumber the sample (it took whichever were read first), and `stratified` gives a slot its rounding left over to a
  class in proportion to its rows not yet picked — the distribution the earlier uniform fill had.

- `mloop train` held the whole data file as dictionaries to count its classes and missing labels before training
  (and again to drop the missing ones). Those passes stream now: training on a million-row, six-column CSV (27 MB)
  peaked at 1.6 GB and now peaks at about 330 MB, with the same model. Merging several files (`--auto-merge`,
  `additionalDataFiles` of `POST /train`) streams too: two 500,000-row files, 830 MB → 330 MB. `mloop detect` keeps only
  the series it reads and writes its `--json` document as it serializes it rather than as one string first
  (2 million rows: 4.5 GB → 1.5 GB, the same bytes).
- `mloop sample create --from` holds the sample, not the file: it read every row into memory first (1.6 GB for
  a 55 MB, 2-million-row CSV — sampling a file too large to profile was no way out) and now streams it (about
  150 MB for a 100,000-row sample of the same file). Random and stratified samples keep the rows in their file
  order (a random sample came out shuffled), and a stratified sample gives every class a row when the sample has
  room for one — rounding the largest classes first could leave a rare class with none.
- `mloop predict --json --log` logs: the JSON branch returned before logging, so `--log` did nothing under
  `--json`. Each row now carries the `predictionId` it was logged under, as `POST /predict` answers — the id
  `mloop feedback add` takes. `feedback metrics` shows `effectiveSampleSize` (and, weighted, the effective rows).

- A failing command always says why on stderr, and under `--json` answers with one `{"error": …}` document.
  `features select`, `prep plan`, `prep run`, `info` (empty file), `validate`, `preprocess` and the `predict` /
  `evaluate` schema check printed their error to stdout only, so the exit code was 1 with nothing on stderr;
  `prep plan --json` with an unknown model left stdout empty; `compare --metrics-file` wrote its error to the
  JSON document only; and a model name containing `[` crashed the message instead of printing it.

- `mloop train --task image-classification --balance <ratio>` balances: the trainer oversamples the smaller classes of
  its training split (after the split, so no copy of an image is among the held-out ones), and the configuration table
  shows the balance. The trainer's own early-stopping slice (10% of what it is given) is set aside per class before
  balancing and passed as its validation set, so no copy is both trained and validated on. Before, image
  classification accepted `--balance` and trained unbalanced without a word. The rule
  — which classes, by how many — is the same one the CSV classifiers use. `--balance` with object detection is refused
  as before (its classes are on boxes, not images).

- `mloop train`'s update of `mloop.yaml` (a label, task or ranking group given on the command line, or a
  model the file did not define), `mloop prep plan --set/--remove` and `mloop features select` rewrote the
  whole file from its settings, dropping every comment — the ones `mloop init` writes, the guide to
  `time_limit_seconds` among them — and the user's key order. They now change only their own lines (the
  label or task line, the model's `prep:` or `columns:` block) and append a new model to `models:`; a
  layout they cannot edit safely is rewritten as before, with a warning that its comments were not kept.
  A file with both CRLF and LF line endings is read as lines and written with CRLF.
- A prep step written to `mloop.yaml` carried every option of every step type — `remove_original: false`,
  `keep_first: true`, `window_size: 0`, `count: 0`, `seed: 42` under a `fill-missing` step. Options at their
  default are no longer written; a file that has them reads as before.

- `--json` documents left out the warnings their commands raised — a column that no longer reads as its
  trained type, a weight the file lacks, a calibration worse than the model's own probabilities — writing
  them only to stderr with the narration. `evaluate`, `predict`, `calibrate`, `threshold` and `compare`
  documents carry them as `warnings` (`evaluate`'s as `POST /evaluate` does); only `train`'s event stream
  had them before. So do `info`, `analyze`, `prep plan` and `features select`. The warnings the data loaders
  raise — a possible multi-row or missing header, a column holding lists, images or annotations skipped — were
  printed as plain lines even then (`mloop info --json` reported success with a multi-row header warning on
  stderr only); they now reach the document's `warnings` and a training run's `warning` events.

- `mloop validate --json` on a `mloop.yaml` in the old single-model format printed the format warning on stdout
  ahead of its document, which no JSON parser read, and reported no warnings. The warning is in `warnings`
  (path `mloop.yaml`) and on stderr.

- An exported binary or regression graph declared the label column as an input although no output
  reads it and the manifest did not list it: the label was dropped from a view over the data, while the
  converter declares a graph's inputs from the data's source. A caller feeding what the manifest names
  was refused by runtimes that require every declared input (ONNX Runtime's Python binding:
  `Required inputs (['y']) are missing`). The graph now declares exactly the manifest's `inputs` and
  `unusedInputs`. `examples/onnx-consumer` scores an exported model from the graph and the manifest
  alone, as `mloop predict` answers.

- `mloop serve` passes a stop it receives (SIGTERM, SIGINT, SIGQUIT) on to the server it runs. It used to exit alone:
  `docker stop` — which signals `mloop serve`, the container's first process — skipped the server's graceful shutdown,
  and outside a container the server went on running and held the port.

- Training an object-detection model recorded no metrics, so the quality gate had nothing to judge and no detector was
  promoted without `--force`; the diagnostics block printed an empty warning line. Training now scores the test images
  with mAP — `map_50_95` (COCO's, the task's metric when `metric: auto`) and `map_50` — and the gate refuses a detector
  whose boxes match nothing (mAP 0), saying so and suggesting another run — on CPU the same data and seed can give
  mAP 0 on one run and a working detector on the next. `mloop init --task object-detection` writes `metric: map_50_95`.

- `mloop predict` with an object-detection model read only an annotated dataset — a folder of new pictures stopped at
  «No COCO annotations file found». It now predicts over images alone: a folder at any depth, one image, or a
  dataset's pictures (its annotations unread). A model trained with a label column of its own name scored with the
  default name and failed.

- `GET /compare` averaged every metric's raw value — accuracies with RMSEs — and called the candidate better when it won
  on more than half of them; a metric one experiment lacked was read as 0, a win for any lower-is-better metric. It now
  judges on one metric, the one the candidate's run optimized (the task's primary metric for `auto`), in that metric's
  direction, as promotion does; the response names it (`metric`), `improvement` is positive when the candidate is
  better whichever way the metric points, and `metricDetails` lists only the metrics both experiments recorded.

- On a training partition of 15,000 rows or more the search validates on one holdout and measures its standard error
  on five slices of it. The slices were cut in the file's order, so a file sorted by its label gave slices of one
  class: the AUC's standard error was missing and the others measured the sort (accuracy's 0.02 on 17,000 rows
  instead of about 0.002), widening the margin within which a smaller model was kept. The slices are now stratified
  random shares of the holdout, dealt with it.

- A binary trainer that gives no probability (FastForest) had its Platt calibrator fitted on the test rows, so the
  probability metrics reported for it were measured on rows the calibrator had seen. It is now fitted on the search's
  out-of-fold answers.

- Platt calibration (`mloop calibrate --method platt`, and the calibration above) fitted labels of 1 and 0, which has no
  finite fit on rows the model separates: the slope grew until probabilities reached 0 and 1. It now fits Platt's
  targets, (N₊ + 1) / (N₊ + 2) and 1 / (N₋ + 2), and stays finite.

## [0.45.1] - 2026-10-25

### Security

- `POST /train` and `POST /evaluate` kept their data files inside the project by comparing the path with the
  project root as a string prefix, so a directory beside the project whose name begins with the root's
  (`/srv/proj-other` beside `/srv/proj`, reached as `../proj-other/train.csv`) passed, and the server trained
  on or evaluated a file outside the project. The check now compares path segments, with the file system's own
  case rule. Both endpoints require the `admin` role.

## [0.45.0] - 2026-10-23

### Added

- `POST /predict` takes an image-classification model's images by their contents — `{"images": ["<base64>", …]}` or
  files as multipart form data, at most 64, as `POST /embed` takes them. Rows with an image path still work where the
  server can read the path; content needs no shared disk. An image that cannot be decoded is a 400 naming its
  position, images sent to a model of another task a 400, more than 64 a 413. Served predictions of images are logged
  by each image's SHA-256 (`imageSha256`).

- `mloop encoder install <id>`: downloads an encoder from MLoop's catalog, checks its pinned SHA-256 and installs it.
  The first entry is `dinov2-small` (DINOv2 ViT-S/14, Apache-2.0, 384 dimensions). `EncoderCatalog` and
  `EncoderStore.InstallAsync` are the library side. A missing encoder's message now names the install command.

### Fixed

- A training that failed inside TensorFlow or TorchSharp ("… exception triggered while loading model") was followed by
  "Train a model first" and "Promote an experiment"; it now points at the runtime (`mloop runtime list`, reinstall).

- Text classification (NAS-BERT) reported the network's logits as class probabilities: `probabilities` held negative
  values that did not sum to one, `confidence` was near 0 for a confident answer, and the log loss recorded at training
  and by `mloop evaluate` was meaningless (15.7 at accuracy 1.0). A classifier that scores with logits now answers in
  probabilities (softmax) on every path — predict, serve, evaluate, training's metrics, calibration. Experiments trained
  earlier keep the log loss they recorded; `mloop evaluate` gives the right one.
- Text tasks on macOS failed to load TorchSharp's library ("Library not loaded: @rpath/libtorch.dylib"): the macOS
  runtime did not preload `libtorch.dylib`. It does now.

- **Image classification could not train on Linux or macOS**: it failed with "Tensorflow exception triggered while
  loading model". The runtime's libraries were preloaded from `~/.mloop/runtimes` by their full path, but a P/Invoke
  asks for `tensorflow` by name, and outside Windows nothing pointed the loader at that directory (an
  `LD_LIBRARY_PATH` set while the process runs is not read again). Preloaded runtime libraries are now handed to the
  imports that ask for them. Measured on Linux x64: training, `mloop serve` and `POST /predict` with image contents.

- The TensorFlow runtime (`mloop runtime install tf`, image classification) listed macOS and a Linux package that do
  not exist: TensorFlow's redistributable 2.16.0 carries Windows and Linux x64 libraries only, and the GPU package
  named for Linux has no 2.16.0. On macOS the install downloaded 182 MB, left 656 MB of other platforms' files behind
  and ended in a stack trace; on Linux it could not download at all. The runtime now lists the platforms its package
  has (Linux uses the same package, with `libtensorflow_framework.so.2`), an install that finds no library for the
  platform removes what it unpacked, `mloop runtime install` reports a failure as a message, `mloop runtime list` says
  "Not available for <platform>" (and `available` in `--json`), and a task whose runtime has no build for the platform
  says so instead of suggesting an install.

- `mloop embed` and `POST /embed` read an image with transparency premultiplied: every translucent pixel was darkened
  by its own alpha, so the vector differed from the one the encoder's own image processor gives (cosine 0.965 on a
  screenshot with a translucent layer). Images are now decoded unpremultiplied and the alpha dropped, as converting to
  RGB does.

## [0.44.0] - 2026-10-23

### Added

- `mloop embed <images…> [--encoder <id>] [--json]` and `POST /embed`: one L2-normalised vector per image from a
  pretrained image encoder. Every answer names the encoder — id, version, weights' SHA-256, dimension — since vectors
  from different weights are not comparable. `POST /embed` takes the images' contents, either a JSON body of base64
  strings or multipart files, never a path. At most 64 images per call. An undecodable image is a 400 naming its
  position, a missing encoder or ONNX Runtime a 409, too many images a 413.
- `mloop encoder add <encoder.json> --model <model.onnx> | list [--json] | remove <id>`: the encoders `embed` runs.
  They are kept outside any project in `~/.mloop/encoders/`, or the directory `MLOOP_ENCODERS` names. A definition
  gives the graph's input and output, the vector length, the pooling (`first-token` or `none`) and the preprocessing.
  `add` records the graph's SHA-256 and refuses a definition that cannot run, or a graph whose SHA-256 differs from
  one the definition pins. When only one encoder is installed, `--encoder` may be left out.
- `mloop docker` writes the lines a container needs to serve `POST /embed`, commented: an ONNX Runtime install in the
  Dockerfile, and the encoders' variable and mount in `docker-compose.yml`.
- `MLoop.Core.Embedding` (library): runs a pretrained image encoder in ONNX Runtime. `EncoderDefinition` describes
  which graph, its input and output names, its dimension and its preprocessing. `ImageEmbedder.Open(definition, path)`
  then `Embed(images)` returns one L2-normalised vector per encoded image (PNG, JPEG, …). `ImagePreprocessor` resizes
  the shortest edge, crops the centre and normalises per channel. It resamples bicubically in Pillow's fixed point, so
  its input is the one the encoder's own image processor computes. An undecodable image is reported by its position
  (`UnreadableImageException.Index`).

### Fixed

- **Yes/no feature columns** (`yes`/`no`, `true`/`false`, `t`/`f`, …) were lost on the way to a prediction:
  - With two or more of them, training never read them: the loader folds them into one boolean vector, and the
    trainer search gave that vector no role, so every model was fitted without those columns.
  - With one, training read it but every reader after it did not: the saved schema recorded the column as a
    category, `mloop predict`, `mloop evaluate` and `mloop serve` loaded it as text, and the model saw a missing
    value in every row. A model whose signal was that column gave one answer for every row — accuracy 1.0 at
    training, AUC 0.5 at evaluation, no error.

  The search now reads boolean vectors, the saved schema records such a column as `Boolean`, and serving folds
  the columns into the vector training read. Models trained by earlier versions with one yes/no column predict
  correctly without retraining (their schema's yes/no values identify the column); models with two or more
  should be retrained to use them.

- `mloop serve` logged every health probe twice at Information — the request line and a "Health check requested"
  line — so a container HEALTHCHECK (the one `mloop docker` writes) filled the log with probes. A successful
  `/health` request is now logged at Verbose, below every configured level; a failing one is still an error.

- `mloop drift` and `GET /drift` counted a yes/no column's values by spelling: a training file's `yes` and a JSON
  request's `true` (logged as `True`) fell in different bins, so a column whose values never changed read as
  drifted. In a yes/no column, values are now binned by what they mean.

- `mloop evaluate --slice-by` printed one table with every metric, and at 80 columns the numbers themselves wrapped
  (`0.862` on one line, `1` under it). The metrics are now split into tables that each fit the console, every
  table keeping the slice and row columns.

- Loading a data file no longer leaves full copies of it in the temp directory. Every preparing step — re-encoding to
  UTF-8, flattening quoted line breaks, removing index, excluded, date-time, sparse, constant and identifier columns,
  converting Excel or Parquet — wrote a copy that nothing deleted, up to one per step for every `train`, `predict` or
  `evaluate`. Now each copy is deleted as soon as the next step has written its own. The last one, which ML.NET reads
  lazily, is deleted when the process ends (`MLoop.Core.Data.ScratchFiles`).
- `mloop train` now warns about a column of strictly increasing integers (a likely row number). The check ran
  on the training partition after the stratified split had shuffled it, so the column no longer increased and the
  warning never fired: a 1..N id was trained on as a feature without a word. It is judged on the whole file now, and
  arrives as a warning event under `--json` too. Loading a file for `predict` or `evaluate` no longer prints it.
- Anomaly detection, clustering and ranking fit one pipeline and take no time budget. Under auto-time (`mloop train`
  without `--time`) they were trained twice, once as the probe and once as the main pass, and `mloop train` showed a
  "Time Limit" that nothing enforced.

### Changed

- Anomaly detection, clustering and ranking read yes/no feature columns as 0/1 in their feature vector; they used
  to warn that such columns were not used. A clustering model now carries its feature step with it (it held only
  the trainer), so it reads the loaded columns as every other model does. Retrain to include yes/no columns;
  models trained earlier keep predicting as before.

- **Breaking (library):** `OnnxVerification.Unavailable()` and `OnnxVerification.NativeLibraryPath()` moved to
  `MLoop.Core.Runtime.OnnxRuntimeNative`. Export verification and the image encoders run on the same ONNX Runtime and
  share one place that decides whether it can run.
- **Breaking (library):** the training runner is named for what it does. `MLoop.Core.AutoML` is now
  `MLoop.Core.Training`, `AutoMLRunner` is `TrainingRunner` and its result `AutoMLResult` is `TrainedModel`; the
  members are unchanged. The training runs MLoop's own trainer search, so the old names described a library it no
  longer uses. Migration: replace the namespace and the two type names — there are no aliases.

### Removed

- The `fellback` phase of `mloop train --json` and `TrainingPhase.ProbeFellBack`, and `TrainerFallbackKind`'s
  `MetricSubstituted` and `AutoMLUnavailable`: nothing has produced them since the trainer search replaced AutoML.

## [0.43.0] - 2026-10-23

### Added

- `explode` and `split-list` prep steps: list columns (a list per cell, as Parquet and JSON lists are read) become one
  row per element, and a list column of one length becomes one column per position (`features.0`, `features.1`, …).
  A file holding a group per row — a query with its documents' grades and feature vectors — can now be trained on.
- `TimestampKeys` (`MLoop.Core.Storage`): the times written into names a program reads back — the day-file name of a
  log or feedback entry (`Day`, `TryParseDay`, `DayFileCovers`) and a time to the second (`Stamp`,
  `yyyyMMdd-HHmmss`) — in the invariant culture and the Gregorian calendar. The prediction logger, the feedback
  collector and the retention pruner read one day format from it, where each had its own copy. A retraining sample
  written by `mloop sample create --model` / `POST /samples` is named `<model>_yyyyMMdd-HHmmss.csv` (was
  `_yyyyMMdd_HHmmss`).
- `ColumnLayout.FileColumns` and `ColumnLayout.FileColumn(name)` (`MLoop.Core.Data`): every column of the file by
  its own name, with the kind it loads as and its purpose — what a column folded into the `Features` vector was read
  as, which `ColumnLayout.Columns` says only of the vector. `ColumnLayout` takes it as a fourth constructor argument.
- `MLoop.Search` package, published with `MLoop.Core` at the same version and a dependency of it: the trainer
  search's own types — `ColumnRoles` (one role per column: numeric, categorical, text or ignored, and the label),
  `TrialResult` and `ITrialSink` (a finished trial and where it is reported), and `IMetricPolicy` (the metric trials
  are ranked by and which way is better). It depends on nothing else of MLoop. `MLoop.Core` supplies
  `RankingPolicy`, MLoop's metric names and directions as an `IMetricPolicy`.
- `MLoop.Search.Tuning`: the hyperparameter tuner MLoop's trainer search will run — `SearchSpace` and
  `SearchDimension` (ranges, log scales, choices, starting values), `CostFrugalTuner` (CFO / FLOW2) and
  `EciSchemaProposer` (estimated cost for improvement across trainers). A trial's cost is in whatever unit one
  search keeps to — the search passes each trainer's relative cost rather than measured seconds, so one seed
  proposes the same trials in the same order on a busy machine as on an idle one. Not used by training yet.
- A trial record (`TrialRecord`, a row of `trials.ndjson`) can carry `error` — a trial that failed, with no
  metrics, ranked below every trial that scored — and `validation` (`folds` and each metric's
  `standardDeviation`) when its metrics are a mean over folds. Both are absent where they say nothing, so
  existing rows keep their shape. The `train --json` `trial` event carries `error` and no `value` for such a
  trial, and it counts among the `trials` like any other. Only MLoop's own trainer search reports failures;
  AutoML's does not. `TrainingProgress.Error` carries it in process.
- `AutoMLResult.Search`: the seed and thread count MLoop's own trainer search ran with, recorded as `search` in
  the experiment's `config.json` — what repeats the run.
- `ColumnLayoutInference.Infer` (`MLoop.Core.Data`): the column layout of a delimited file with a header — the
  `TextLoader.Options` (separator, quoting, columns, numeric columns folded into one vector) and each loaded
  column's purpose (`ColumnLayout`). It gives the layout ML.NET AutoML's `InferColumns` gives, field for field,
  except that a file over 8 MB is sampled from fixed positions (the same file always gives the same layout) and no
  column is read as an image path. Every command reads its layout from it (`train`, `predict`, `evaluate`, `info`).
- `EciSchemaProposer.Propose(allowed)`: a proposal among the schemas a predicate admits (`null` when it admits
  none). `TrialRunner` uses it near the end of a budget — a schema whose last trial no longer fits the time left is
  passed over for one that does, instead of the search ending with the budget unspent.

### Changed

- When the top trial's model is large — 100,000 tree leaves (trees × leaves) or more — the trainer search keeps
  the smallest model within one standard error of its score (the one-standard-error rule). Measured on a 43,000-row
  regression: a forest of 1,135 trees of 1,510 leaves led one of 27 trees of 252 leaves by 0.006 R², inside its
  holdout's standard error; it was a 193 MB `model.zip` whose save took 10 s and ran a 60 s budget to 79 s. Now 1.0 MB, 62 s, R² 0.9677
  → 0.9645. The run says which trial it kept over which; that trial carries `"selected": true` in `trials.ndjson`
  and `leaderboard.json` (ranked by score as before), and `mloop list --trials` marks it `(model)`. A holdout's
  standard error is the spread of five slices of it.
- The search reserves the refit's time by the share of rows a trial fitted on — for a large dataset's holdout,
  nine tenths — where it reserved one fold-fit's time, so a refit on all the rows no longer runs past the budget.
- The trainer search one-hot encodes categorical columns and no longer also tries each trainer with its categories
  hashed (`OneHotHashEncoding`, 2¹⁶ slots) — a schema is a trainer, so a budget runs about twice the trials per
  trainer. The hashed vector is never the narrower one for a column MLoop calls a category, and on a 500-value
  category it scored AUC 0.60 against one-hot's 0.71. A trial's parameters no longer carry `CategoricalEncoding`.
- An empty numeric field is read as missing (NaN) by every loader — training, `predict`, `evaluate`, `serve` and the
  ONNX export's data — instead of as 0, ML.NET's default. A missing reading used to train and predict as a reading
  of zero. The trainer search now replaces a missing number by its column's mean (fitted on the training rows only)
  rather than by zero, and adds an indicator feature (`<column>.missing`) for each column that had missing values in
  training. Clustering, anomaly detection and ranking replace a missing number by the mean. Forecasting and
  time-series anomaly detection refuse a series with a gap, naming the column and the `fill-missing` step that
  fills it. A clustering, anomaly-detection or ranking model trained before this release reads a missing input as
  NaN — retrain it.
- The trainer search scales features to [0, 1] ahead of its linear trainers (L-BFGS and SDCA, every task); trees read
  them as before. Unscaled, a wide column next to narrow ones made those solvers crawl and learn little — on a
  16,730-row four-class table three linear trials took half of a 60-second budget at macro accuracy 0.25. Measured on
  six tabular sets at 60 s: four better (macro accuracy 0.80 → 0.97 on that table), the rest equal.
- `log_loss_reduction` can be the optimizing metric of multiclass classification. It was refused while ML.NET AutoML
  searched — AutoML minimized it although higher is better, and chose the model that had learned least; MLoop's
  search ranks it the right way.
- **MLoop no longer depends on ML.NET AutoML** (`Microsoft.ML.AutoML`). With it go the transitive TorchSharp,
  `Microsoft.ML.TorchSharp` and `Microsoft.ML.Vision` a tabular-only consumer of `MLoop.Core` used to receive —
  nothing to exclude any more (`docs/EMBEDDING.md`). **Breaking (in-process API):** `AutoMLRunner.BuildManualBinaryPipeline`
  is removed (it built the fallback that answered AutoML's AUC failure, which MLoop's search does not have);
  `WeightedResampling` is `RowWeights` — `RowWeights.Read` where you called `ReadWeights`, and `Resample` / `DrawRows`
  are removed (the search weights the trainers instead).
- **Binary classification, multiclass classification and regression train with MLoop's own trainer search**
  (`MLoop.Search`) instead of ML.NET AutoML's experiment. What a run records changes with it: each trial is scored
  as a mean over stratified folds (with its spread) and the best is refit on the whole training partition; a trial
  names its trainer (`LbfgsLogisticRegression`, `FastTree`) rather than an AutoML pipeline string; the seed and
  thread count go into `config.json`; a weight column weights the trainers instead of resampling rows; and the
  search spends its whole time budget. Saved models load and predict as before — the column layout is unchanged.
  Data with so few rows of one class that a fold holds none no longer falls back to a single SDCA fit: such a fold
  is scored with the metrics it defines.
- **Breaking (in-process API):** `MLoop.Core` no longer hands out ML.NET AutoML column types.
  `AutoMLRunner.BuildColumnInformation` is `AutoMLRunner.BuildColumnRoles` and returns a `ColumnRoles` for every
  table — read a role's columns with `roles.Of(ColumnRole.Text)` where you read `info.TextColumnNames`, and the label
  with `roles.Label`. It no longer returns null for a table of numeric columns: such a table is all
  `ColumnRole.Numeric`. `AutoMLRunner.BuildManualBinaryPipeline` and `PortableTextFeaturizer.FitAndApply` take and
  return `ColumnRoles`. `TrialProgressReporter<TMetrics>` is internal.
- **Breaking (in-process API):** `CsvDataLoader.ReconcileInferredSchemaForInference` and
  `CsvDataLoader.ApplyColumnPreservation` take the `TextLoader.Options` they adjust — pass `layout.Options` from
  `ColumnLayoutInference.Infer` where you passed an `InferColumns` result. No public member of `MLoop.Core` hands out
  an ML.NET AutoML type any more. A file whose layout cannot be inferred fails with `ColumnLayoutException`, with the
  messages it had.

- ONNX Runtime moves to 1.31.0 (from 1.23.2). `mloop export onnx --verify` needs the native library at the same
  version, so an installed 1.23.2 runtime now reads as needing reinstall: `mloop runtime list` shows the version on
  disk and the one needed, `--verify` says so, and `mloop runtime install onnxruntime` replaces it.
- `MLoop.Core` declares the ML.NET components it calls — FastTree, LightGbm, TimeSeries, Recommender, ImageAnalytics
  and the ONNX Runtime managed API — instead of receiving them only through `Microsoft.ML.AutoML`. An in-process
  consumer gets the same assemblies; the package's dependency list now names them.

- **Breaking (in-process API):** `TrialLedger.IsReportable` takes the search's `TrialResult` (a trial is
  recorded when it scored or failed with a reason) instead of AutoML's metrics object.

### Fixed

- The trainer search refits its chosen trial in an MLContext of its own. It reused the trial's context, and a trial
  that finished as the budget's cut arrived left that context stopped: the refit failed with "Operation was
  canceled" and the run with it — timing-dependent, seen on a CI runner.
- Numbers are read from text with the invariant culture everywhere, as ML.NET's loader reads a file — not the
  machine's. On a machine whose decimal separator is a comma, regression refused every label ("No valid numeric
  values found in label column"), and the saved schema judged the numeric columns folded into the feature vector by
  that parse — as text or categories, by how many values they had — so `serve` would load them as strings. The saved schema now takes such a column's type from the column layout,
  by its own name (`ColumnLayout.FileColumns`); prep's `filter-rows` comparisons and the data-quality checks parse
  the same way.
- Dates in file names are written and read in the Gregorian calendar whatever the machine's culture. Under a culture
  whose calendar differs (th-TH counts Buddhist years) the prediction-log and feedback day files were named `2569-…`,
  which `logs prune` — reading names with the invariant culture — took for a date five centuries ahead and never
  pruned; promotion backups, prediction outputs, `POST /train` job ids and retraining samples were named the same
  way. `GET /logs` `from` / `to`, the experiment record's timestamp and date detection in prep read dates with the
  invariant culture too.
- Auto-time (`mloop train` without `--time`) judges its probe by the search's validation score, not the test
  partition's: whether the probe has converged (score above 0.95), and whether the probe's model or the main run's is
  kept. Reading the test score let the score the run reported choose its own model. It also read an error metric
  (rmse, mae) as if higher were better — an rmse of 13,000 counted as "converged", and the worse of the two runs was
  kept. A run whose validation score is below 0.95 now goes on to main training where its test score once stopped it.
- `POST /predict` (and `PredictionService.Predict` in process) matches a row's fields to the model's columns ignoring
  case. A body whose keys differ in case from the columns — `x1` for `X1`, as a serializer with camelCase names
  writes them — passed the check that a row shares a column with the model, and then every one of its values was
  dropped: the row was scored as all zeros and answered with 200.
- **Clustering without a label could not train, and anomaly detection without a label could not predict** — on a
  file of plain numbers. A label-less file was loaded with its first column standing in as a label, set apart from
  the `Features` vector, while every prediction path rebuilt the vector with that column inside it: clustering failed
  at the end of training (`Incompatible features column type: Vector<Single, 2> vs Vector<Single, 3>`), anomaly
  detection at `mloop predict` (`expected Vector<Single, 3>, got Vector<Single, 4>`). A label-less file is now loaded
  without a label, so training and prediction read one vector. A yes/no column, which also failed the fit, is not
  read by these tasks (nor by ranking), and training names the ones it leaves out.
- `mloop prep run` and the prep step of `mloop predict` reported warnings straight to the console, so with
  `--json` they never became `warning` events; `mloop preprocess` and `mloop train` passed messages into console
  markup unescaped, so a message with a bracket in it (a file name such as `[train].csv`) threw instead of
  printing. Every command now logs through one console logger: information as narration, warnings through the
  warning channel, errors to stderr, each printed as written. `mloop train` no longer prints debug detail (a
  hook's metadata values, the clean-up of preprocessing files), as no other command did.
- A weighted run could report an unweighted `negative_precision` beside weighted metrics. When no row was
  predicted negative, the weighted value is undefined and the unweighted one stayed under the same name; it is now
  left out, as the other undefined weighted metrics are.
- `mloop export onnx`: the manifest's `label.decision` left out a calibration (`mloop calibrate`) — it called the
  graph's probability "calibrated" and, with a threshold, told the consumer to compare it directly — so a consumer
  following it answered differently from `mloop predict`. The sentence now gives the whole rule in `mloop predict`'s
  order (the calibration with its parameters, then the threshold, or 0.5 when there is none), and `--verify` applies
  that rule to the graph's outputs and compares it with `mloop predict`'s answer (`verification.layered`); a
  disagreement fails the export like any other.
- `mloop threshold` chose a threshold equal to a row's probability. Models with few distinct probabilities put many
  rows on that value and left their answer to the last float digit: an ONNX graph answered 39 of 300 rows the other
  way. The threshold now sits halfway between the last probability answered one way and the next, so in-sample
  answers are unchanged and another runtime's rounding keeps them. A threshold chosen before stays as written.
- `mloop runtime install` and every runtime check counted native libraries of any version as installed. A runtime
  is installed only at the version this mloop needs; reinstalling over another version replaces its files.
- **A multiclass file sorted by its class ids could fail at the end of training** with `Could not apply a map over
  type 'String' to column 'class' since it has type 'Single'`. Each partition of the split typed the label from its
  own first rows: the training partition, all class `0` there, read it as text; the test partition, reaching `1`
  and `2`, as a number. Every load of a run — both partitions, `mloop evaluate`, `mloop predict` — now types the
  label as the saved schema recorded it (`LabelKinds`).
- **`mloop train` showed its performance diagnostics only for a weak model**, so a warning about a strong one never
  appeared — and the training row count the diagnostics read was always 0, from a file-backed view that does not
  know its length. Both are fixed, and a new warning says when a held-out score is near-perfect on 500 rows or more
  (`r_squared`, `auc` or `macro_accuracy` ≥ 0.999): check that no feature is computed from the label or recorded
  after it.
- The performance diagnostics' warnings reached only a person at the terminal: a `--json` run now reports each as a
  `warning` event, and a `POST /train` job carries them as `warnings` (`GET /jobs/{id}`).
- **Each column dropped from the features was announced twice** during `mloop train`; it is announced once, with
  what the reason means (`Excluded from features (Identifier): customerID — a distinct value in every row`).
- **`mloop init .` replaced a project's own `mloop.yaml`, README and `.gitignore` with the template** — in a cloned
  repository, which carries those files but not the `.mloop/` marker, the first command run wiped the project's
  models, tasks and labels without a word. `init` now writes only files that are missing and lists the ones it kept;
  `--force` rebuilds `.mloop/` and leaves the project's files alone too. COCO annotations already under
  `datasets/coco/` are kept the same way.

### Removed

- The unpublished `agent/` package, the agent example and its prompt files, and documents describing the
  `mloop agent` command, which was removed earlier — AI agents use MLoop through [mloop-mcp](https://github.com/iyulab/mloop-mcp),
  the CLI or `mloop serve` ([docs/AI-AGENTS.md](docs/AI-AGENTS.md)). The `.env.example` now lists the variables
  MLoop reads.
- The equipment anomaly detection example. Its training file held one class, so it could not train, and its
  preparation script wrote a fixed sample instead of labeling the readings it copied.

## [0.42.0] - 2026-10-22

### Added

- NER models report `entity_precision`, `entity_recall` and `entity_f1` beside the word accuracies, in training
  and `mloop evaluate` — whole entities with their type, the score NER is usually reported by. A model that finds
  half of every entity no longer reads as half right. Tags are read as the IOB family (`B-`/`I-`, BIOES, bare types).

- `POST /train` takes `balance` — class balancing as `mloop train --balance` (`auto`, `none` or a target ratio):
  the test part is split off (stratified) first and only the train part is balanced. `GET /jobs/{id}` shows it, and
  a job notes what balancing did and a class imbalance in its status.
- `POST /train` takes `promote` — `false` keeps the experiment in staging, as `mloop train --no-promote`, for an
  application that decides promotion itself (its own gate, a review) and then calls `POST /promote`. A completed job
  reports `promoted` and, when the experiment stayed in staging, `keptInStagingReason` (`GET /jobs` lists `promoted`).
- `mloop validate` compiles the project's scripts — hooks in each stage folder, metric scripts, pattern detectors —
  and reports one that does not compile or implements nothing it would run as an error, and a hook outside every
  stage folder as a warning. The guide showed this output before any of it existed.
- Retention for prediction logs and feedback: `retention: days: N` in `mloop.yaml` keeps today (UTC) and the N days
  before it. `mloop serve` removes older days at start and once a day; `mloop logs prune [--days N] [--dry-run]
  [--json]` does it on demand. Whole day-files leave, logs and feedback together; nothing is removed without a period.
- `GET /experiments` rows are the rows of `mloop list --json`: each carries its `metrics`, `qualityGate`,
  `qualityWarning` and `isProduction`, and the answer names each model's `production` experiment — one call where a
  client used to fetch every experiment separately. Rows are newest first, as in the CLI; `mloop list --json` rows
  gain `labelColumn` and `trainingTimeSeconds`.

### Documentation

- The guide covers `mloop new` (templates, stages, what each writes) and `mloop preprocess` (scripts, order, `-o`,
  the incremental mode), and its Hooks section names the stage folders and what a hook gets at each stage — the old
  example used a namespace hooks are not in and a folder that runs a hook at both stages. Every command now has a
  section.
- The guide covers `mloop trigger`, `mloop prep` (plan · run · step types · where each step runs), `mloop features`,
  `mloop runtime` and `mloop update`.

### Changed

- Hooks run only from their stage folder (`hooks/pre-train/`, `hooks/post-train/`). A hook directly in
  `.mloop/scripts/hooks/` used to run at both stages, from a second discovery inside training; it no longer runs, and
  `mloop train` warns with its name and the folders to move it to. `AutoMLRunner` no longer takes a project root (it
  was used only for that discovery), and `ScriptDiscovery.DiscoverHooksAsync` / `GetHooksDirectory` are gone —
  `HookEngine` is where hooks are found and run.

- A training job reads `Completed` only after its promotion is decided; before, a client that polled to completion
  could read production before the job's experiment had been promoted.

- A post-train hook that returns `HookResult.Abort` keeps the model out of production: it is saved in staging and not
  promoted automatically, by `mloop train` and `POST /train` alike (the job's status says why). It used to be a
  warning while the model was promoted anyway, which made a performance-gate hook a no-op.
- `mloop serve` no longer takes `-h` for `--host`: `-h` is help on every command, and `mloop serve -h` failed with
  "Required argument missing for option: '-h'". Write `--host <address>`.

### Fixed

- Metric scripts in `.mloop/scripts/metrics/` were never computed — nothing ran them, although `mloop init` scaffolds
  the folder and the guide describes them. Training now computes each over the held-out rows and records it with the
  run's metrics (`metrics.json`, results table, `list --json`, post-train hooks), and `mloop evaluate` / `POST
  /evaluate` compute them over the evaluated file. A name the run already reports, or a non-finite value, is skipped
  with a warning. A script that did not compile used to vanish without a word when loaded through `ScriptDiscovery`;
  the compiler's message is reported now. The shipped examples (`docs/examples/metrics/`) did not compile
  (missing `using` directives) and checked a task name MLoop never passes (`BinaryClassification` →
  `binary-classification`); the guide's and `mloop init`'s examples showed an interface that does not exist.
- Warnings that carried their own sign read `Warning: ⚠ …` — the prefix and the sign saying the same thing. The
  warning seam drops a leading `⚠` (a severity colour such as `🟡` stays).
- A script's compile errors named their line and column counted from 0 (`0,30:`); they read `line 1, column 31:` now.
- What a hook or metric script logs (`ctx.Logger`) reached stdout under `--json`, ahead of the document — `mloop
  evaluate --json` with a metric script, `mloop train --json` with a hook that logs, could not be parsed. It goes
  where the command's other narration goes.
- The hooks README `mloop init` writes told users to put hooks in `.mloop/scripts/hooks/` itself; it names the stage
  folders now.

- Feedback for a prediction logged more than 7 days earlier was refused as "prediction not found" (`mloop feedback
  add`, `POST /feedback` 404) although the prediction was in the logs. Every logged day is searched now, newest
  first; how far back the logs reach is the retention period.
- `mloop preprocess` said "Running preprocessing scripts" twice and "Preprocessing complete" three times.
- `mloop new hook` wrote hooks that did not compile — every template, for every stage (`Exception` without
  `using System;`), and the validation template printed `{rowCount:N0}` instead of the count. The training engine then
  reported "No IMLoopHook implementations found", because script compile errors were never shown; they are now, for
  hooks, custom metrics and preprocessing scripts. A test compiles every template for every stage.
- With a pre-train hook, `mloop train` always failed with "Specified argument was out of the range of valid values.
  (Parameter 'Max')": hooks ran twice — once from the command, once from the training engine, whose data loader for
  the hook was invalid. They now run once, in the training engine, for `mloop train` and `POST /train` alike; a
  pre-train hook gets the training data as a `DataView` and its path as `Metadata["DataFile"]`.
- The `performance-gate` and `deploy` templates read metrics as ML.NET metric objects, which hooks never receive, so
  the gate always passed; they read the run's metric dictionary and name the metric they hold the model to.
- `mloop new hook --help` said the default template was `basic`; it is `validation` for pre-train and pre-predict and
  `logging` for post-train and post-evaluate. An unknown `--template` is refused instead of writing the basic template.
- `mloop train` checked the label (missing values, classes, a single class) on the file before the `prep:` steps of
  `mloop.yaml` ran, so a step that drops or reshapes rows was trained on after the raw file's classes had been reported.
  The steps now run first and the checks see the rows training gets.
- A ranking trainer whose native library does not load (LightGBM on osx-arm64) printed the loader's full search, dozens
  of lines; it now says in one line that the trainer was skipped and why, and training continues with the others.
- `mloop validate` refused a `sample` prep step that `mloop prep run` accepts and runs. The step names and what each
  step needs now come from one place (`PrepStepRules`) that the step runner and both validators read, and an unknown
  step type is reported with the full list everywhere.
- Training jobs survive a `mloop serve` restart. They were held in memory, so after a restart `GET /jobs/{id}` answered
  404 for a job whose experiment was on disk. Each job is now a JSON file under `.mloop/jobs/`. A job that was running
  when the server stopped reads as `Failed` with a message saying it was interrupted, and a job still queued is trained
  when the server starts again.
- `mloop init` writes `.mloop/temp/` and `.mloop/merged/` (copies of training data the API merges) to `.gitignore`.
- A `POST /train` job on classification data with one class failed inside training with an unhelpful error; it
  now fails with the reason, as `mloop train` does.
- `mloop trigger check` showed an accuracy nobody measured (no feedback yet) as `0.0%`, which reads as a model that
  failed; it shows `-`, and `currentValue` is `null` in `--json` and `GET /trigger` for every condition that could
  not be measured (no feedback, too few logged predictions, a condition the trigger does not evaluate).
- `mloop prep plan` showed a step's columns in doubled brackets (`columns: [[x]]`).
- The prep leakage warnings, a `validate` note and a `train` line were in Korean while the rest of the output is
  English; they are in English.
- `mloop runtime install` / `remove` help listed `tf, torch` and not `onnxruntime`; it lists what the registry knows.
- `mloop train --balance` with a value that is not `auto`, `none` or a positive ratio trained unbalanced after a
  grey note; it is now refused before training. So is `--balance` on a task that is not classification (it was a
  warning). A ratio reads the same in every locale (`2.5`).

## [0.41.0] - 2026-10-21

### Added

- `mloop feedback add --weight <w>` (and `weight` on `POST /feedback`) records how much the row counts — for a
  row reviewed with probability *p*, 1/*p*. Finite and not negative; absent counts 1. `feedback list` and
  `GET /feedback` show it, and `feedback metrics` counts each prediction by its weight when any entry has one
  (`weighted` in `--json`).
- `POST /samples` — a retraining set from logged predictions and feedback over HTTP, as `mloop sample create
  --model` (admin). `POST /train` takes `additionalDataFiles`, merged after `dataFile` as `mloop train --data a b`
  (a file without the weight column counts 1), and `weight` — the column that weights the rows, as `--weight` (the
  `weightColumn` `POST /samples` answers with); `GET /jobs/{id}` shows both. A relative path is the project's.
- `POST /evaluate` takes `sliceBy` — per-value metrics, as `mloop evaluate --slice-by`.
- `mloop sample create --model` strategies `stratified` and `low-confidence` — the sampler had both, the command
  did not accept them.

### Changed

- **`mloop sample create --model` writes a training file.** The feedback's actual value goes under the model's
  label column (from its production experiment, else `mloop.yaml`), with a weight column when the model trains
  weighted or any feedback carries a weight. The `Output` (the model's own answer) and `ActualValue` columns are
  gone: every column in the file is a feature to the next training, and a model given its previous answer learns
  to copy it. To migrate, train on the file with the model's label — no `--label ActualValue`, and no `Output`
  column to drop.
- `IFeedbackCollector.RecordFeedbackAsync` takes `weight` before the cancellation token; `FeedbackEntry` carries
  `Weight`; `IDataSampler.SampleAsync` takes the `RetrainingColumns` to write under.
- `ICsvMerger.MergeAsync` and `ValidateSchemaCompatibilityAsync` take `columnDefaults` — columns a file may lack,
  with the value its rows take — before the cancellation token.

### Fixed

- `mloop analyze` on numeric columns in a small unit (values near 1e-9) reported an infinite condition number and
  permutation importances of 0, and PCA, Mahalanobis and VIF read the columns as degenerate — fixed by depending on
  DataLens 0.19.2 (UInsight 0.36.2).

- A `POST /train` job trained on its file as given: `maxRows` was recorded and never applied, and rows with an empty
  label were trained on (an empty label a class of its own). A job now prepares its data as `mloop train` does —
  read as a table, sampled to `maxRows`, rows without a label dropped where the task drops them.
- `POST /evaluate` read the test file as CSV whatever it was and skipped the schema check `mloop evaluate` makes: a
  Parquet or JSON file was taken as text, and a file missing the model's columns failed as a server error (500). It now
  reads the file as a table and refuses a schema mismatch with `400`, naming the missing columns. A relative
  `testDataPath` is the project's, not the server's working directory.
- `POST /train` without `timeLimitSeconds` or `testSplit` used 300 s and 0.2 over what `mloop.yaml` set. Both are
  optional now, and an absent one is left to `mloop.yaml`, then the default — as an absent `--time` is.

- `mloop train --data a.csv b.csv --weight w` refused to merge when only some files had the weight column. A file
  without it is unweighted: its rows now count 1, so original training data and weighted feedback merge as they are.
- A column was excluded as constant when its first 200 rows agreed, and as sparse when its first 200 rows were
  mostly empty — even when the rest of the file said otherwise (rows appended after the original data). Both rules
  now read every row; the constant rule stops once each column has shown two values.
- A weight column of mostly 1s was removed from the data as a constant feature, and weighted training then failed
  with «The test data has no weight column». No feature-exclusion rule removes a column claimed for a role (weight,
  ranking group, recommendation user and item, pre-featurizer inputs, typed overrides) — before, only the
  structured and identifier rules respected that.
- MCP `mloop_feedback`, `mloop_sample`, `mloop_detect` and `mloop_prep` passed options the CLI does not have
  (`--value`, `--name`, a data-file positional, `--period`, `--config`), so those calls failed. They now pass the
  CLI's own options; `mloop_feedback` gains `source` and `weight`, `mloop_sample` `modelName`, `output` and `label`,
  `mloop_prep` `modelName`, `input`, `output` and `dryRun`, and `mloop_train` `additionalDataFiles`. The MCP server's
  `npm run check:cli` compares every tool's command lines with the CLI's own options.
- A retraining set wrote logged inputs as the client spelled them — `x1` from a JSON client of a model trained on
  `X1` — so it did not merge with the training data. Inputs are now written in the model's spelling.
- `mloop feedback list` printed the feedback's own text (prediction, actual value, source) as console markup — a
  `[` in a label broke the table.

## [0.40.0] - 2026-10-21

### Added

- `mloop predict --experiment <exp-id>` predicts with a given experiment — a staging candidate
  answers the same rows production would, and production does not move. Rows logged with `--log`
  carry the candidate's id, so `mloop drift` (which reads production's rows) and feedback keep the
  two apart.
- `mloop predict --json` names the model that answered in `experimentId` (production's experiment
  by default).
- `mloop calibrate [exp-id] --data <labeled> [--method platt|isotonic|temperature] [--weight <col>]` fits
  a classifier's probability calibration on labeled rows without retraining — binary by Platt (default)
  or isotonic, multiclass by temperature (its answers unchanged, the class names kept) — and every path that scores
  — predict, serve, evaluate (its `ece`) and a threshold chosen afterwards — reads the calibrated
  probability; `promote` carries `calibration.json` with the model, and the ONNX manifest names it.
  Refused while a decision threshold stands (calibrate first, then choose the threshold).
  MCP: `mloop_calibrate`.
- `POST /predict?experiment=<exp-id>` — the same shadow prediction over HTTP; logged rows carry the
  candidate's id. `404` for an unknown experiment or one without a trained model.
- `mloop threshold --precision-of <class>` targets the precision of a named class — the negative
  one included, for when a wrong «no» is the costly answer. The threshold is then the highest one at
  which that class's answers reach the precision; the class is recorded in `decision-threshold.json`
  (`precisionOf`; absent = the positive class, as before).
- `mloop drift` measures free-text columns: training profiles each text feature by its length and by
  the share of words its vocabulary never saw, reported as `<column> (length)` and
  `<column> (unseen words)` — so `input_drift` fires for a text classifier whose inputs change subject.
  Models trained earlier keep their profile and are measured as before; retrain to add the text columns.
- `mloop evaluate` reports `negative_precision` and `negative_recall` for binary models — the
  former was already a metric name `train --metric` accepted, but no evaluation reported it.

### Changed

- `--metric log_loss_reduction` and `--metric top_k_accuracy` (and `metric:` in `mloop.yaml`) are
  refused for training, with the reason. Neither ever optimized what its name says: the AutoML engine
  minimizes log-loss reduction although higher is better, so a run chose the model that learned
  least (trials reached 0.96, the result was 0.0); and top-k accuracy needs a K that trials are not
  scored with, so it was 0 for every model. **Migration:** use `log_loss` — on the same rows it ranks
  models exactly as its reduction does — and for top-k, `macro_accuracy` or `micro_accuracy`.
  `log_loss_reduction` is still reported for every multiclass model.

### Fixed

- A model whose only input is free text wrote no drift profile at all, so `mloop drift` said "no
  profile" and its answer distribution — the one drift signal it has — was never recorded. The
  answers are now profiled whenever they can be.
- `POST /evaluate` for an experiment without a trained model evaluated the production model and
  reported it under the experiment's id; it is now `404` with the reason.
- Commands and endpoints that take an experiment to score with (`predict --experiment`, `evaluate`,
  `export onnx`, `threshold`, `calibrate`, `POST /predict`, `POST /evaluate`) say the same thing when it
  does not exist or has no trained model — `evaluate` used to print the model file's absolute path.
- `log_loss_reduction` was read as lower-is-better wherever experiments are ranked or compared
  (its name contains "loss"); it is higher-is-better.
- A multiclass result did not carry `log_loss_reduction`; regression and multiclass results, like
  binary ones, now come from one mapping per task family.
- A binary model trained with `--metric auprc` or `--metric negative_precision` recorded no value for
  the metric it optimized: training reported neither (and `evaluate` did not report `auprc`). Every
  binary result — training, each trial, the small-data fallback and `evaluate` — now reports the same
  set, the metrics `train --metric` accepts for a binary model.
- The guide's `mloop predict` synopsis showed the model as a positional argument and a staging path
  without the model name; the model is named with `--experiment` or `--model`.

## [0.39.0] - 2026-10-21

### Added
- **`mloop drift` compares the model's answers too.** Training a classifier or a regressor now also
  records, in `drift-profile.json`, how the model's own answers were distributed over the sampled
  training rows — the predicted class, or the predicted value cut at its deciles — read the way a
  prediction log records them. `mloop drift`, `GET /drift` and the `input_drift` condition's `drift`
  object carry it as `prediction` (PSI, level, most moved bin), and the text output adds an
  **Answers** line. It is kept apart from the inputs' level, which still alone decides
  `input_drift`. `prediction` is `null` for models trained earlier and for tasks whose answer is
  neither a class nor a value.
- **`mloop evaluate --against <experiment|production>` compares two models on the same rows.** The
  other model is scored on the same test file with the same weights; its metrics are shown beside
  the candidate's, and a classifier or regressor is paired row by row — the weighted mean per-row
  loss (wrong class, or squared error) of each, their difference, and a 95% paired bootstrap
  interval with a verdict (`CandidateBetter`, `BaselineBetter`, `Indistinguishable`). `--json`
  carries it as `comparison`. A model trained for another task or label, and the candidate itself,
  are refused. `MLoop.Core.Evaluation.PairedComparison` holds the measure.
  `POST /evaluate` takes `against` too.
- **`mloop evaluate` shows a classifier's confusion matrix** — actual × predicted, in the label's own
  class names, each cell a weighted sum under `--weight`. `--json` carries it as `confusion`
  (`classes`, `counts`, `total`) overall and per slice, and so does `POST /evaluate`.
  `MLoop.Core.Evaluation.ConfusionCounts.ShareOf` reads any class-set measure from it — of the rows
  labeled with one of a set of classes, the share answered with one of another.
- **`mloop threshold` — a binary classifier's decision threshold for a target precision.**
  `mloop threshold [experiment] --precision 0.95 --data <labeled file> [--weight <column>]` chooses the
  lowest probability threshold whose positive precision reaches the target on those rows (keeping
  the most recall; rows of one probability are never split) and writes it beside the model as
  `decision-threshold.json` — on production too when the experiment is in production. Every path
  that loads the model then answers at it: `predict` (CSV and `--json`), `mloop serve` (its model
  cache reloads when the threshold changes), `evaluate` and the gate policy. `promote` carries it
  with the model; a promoted model without one answers at 0.5. Without `--precision` it shows the
  current threshold; `--clear` removes it. The ONNX graph keeps the trainer's own decision, and the
  manifest carries `decisionThreshold` and says to apply it to `Probability`.
- **A project's own gate policy** (`gate:` under a model in `mloop.yaml`): rules a model must also
  pass to be promoted, judged on an evaluation set (`data`, optional `weight`) when `promote`, a
  training run or `POST /promote` would promote it — `no_worse_than_production` (refused when the
  row-by-row comparison with production says production is better on those rows) and class
  `limits` (`actual`, `predicted`, `max`: of the rows labeled with one of `actual`, at most `max`
  answered with one of `predicted`), also per `slice_by` value with 30+ rows. A refusal names the
  rule; `--force` (or `"force": true`) overrides and the history records it as `forced`. A rule that
  cannot be judged refuses. `mloop validate` checks the policy.
- **`mloop promote --history` lists every change of production**, with what each one replaced, how
  it happened (`promote`, `auto-promote`, `rollback`) and the quality gate's outcome at that moment
  (`passed`, `forced` with the refusal, `not-applied`) beside the project metric's value. `--json`
  returns it as `promotions`, and `GET /promotions?name=` returns the same.

### Fixed
- **`evaluate` counts the answers a binary model gives.** ML.NET's binary evaluator re-derives each
  answer from the score at its own threshold and ignores the model's `PredictedLabel`; for a model
  answering at a chosen decision threshold, accuracy, precision, recall, negative recall and F1 are
  now counted from its answers, as the confusion matrix is. AUC does not depend on the threshold.
- **A model promoted by a training run is in the promotion history.** Only `promote`, `rollback`
  and their REST twins recorded an entry, so after `mloop train` (or `POST /train`) replaced the
  production model, `mloop rollback` answered that the history did not record what production held
  before. The production slot now records every change where it happens — the registry's
  promotion — so the history has no gaps.
- **`POST /evaluate` scores a weighted experiment by its weight.** It took no weight and evaluated
  every experiment unweighted — a model trained on weighted rows was judged on the distribution it
  was trained away from, and the response did not say so. It now takes `weight`, defaults to the
  experiment's training weight when the test file has that column, says in `warnings` when it does
  not, and returns `weightColumn` — the same choices `mloop evaluate` makes, from one place.

## [0.38.0] - 2026-10-21

### Added
- **A drift reference is saved with every model.** Training writes `drift-profile.json` beside
  `model.zip`: for each numeric, categorical and boolean feature, how the training inputs were
  distributed — numeric columns cut at their deciles (a repeated value stays in one bin),
  categorical ones by their 20 most common values plus an "other" bin, missing values in a bin of
  their own; built from a fixed-seed sample of up to 20,000 rows, so a large file costs flat memory
  and the same file gives the same profile. `promote` carries it to production (and drops the
  previous model's when the new one has none). `MLoop.Core.Monitoring` holds the profile and the
  population stability index (`DriftMeasure` — under 0.1 stable, 0.1–0.25 moderate, 0.25 and
  above significant), which reads logged JSON inputs the way the training file was read.
- **`mloop drift`** compares the production model's own logged predictions (the most recent 5,000
  by default, `--rows`) with that profile: each column's PSI, its level, and the bin that moved
  most (`> 13.6: 10% → 45%`, a category, `(other)`, `(missing)`); `--slice-by <column>` repeats it
  per value of an input column, so a moved site or device stands out from steady ones; `--json`
  and `GET /drift` give the same document. Computed when asked — nothing runs in between.
- **Input drift is a retraining condition.** `mloop trigger check` and `GET /trigger` evaluate
  `input_drift` (met at PSI ≥ 0.25 on any column) beside accuracy, feedback volume and age, and
  carry every column's PSI under `drift` in their JSON; a model with no profile, or too few logged
  predictions, is reported as not measured rather than as stable.

### Fixed
- **`mloop status` and `GET /status` ranked weighted and unweighted experiments against each other.**
  The best-metric value is now the best among experiments measured like the newest one — the same
  metric and the same row weighting. The table names the metric beside the value and notes the
  weighting and any experiments left out; `--json` adds `bestMetricName`, `bestMetricWeightColumn`,
  `bestExperimentId` and `experimentsNotCompared`. Experiments optimized for different metrics were
  compared too, and no longer are.
- `mloop compare --name <model> --best N` takes its N from experiments measured like the newest one
  first, rather than mixing weighted and unweighted values into one top-N.
- The experiment index records each experiment's weight column (index format 2). An older index is
  read by filling the column from each experiment's own record, so nothing needs migrating.

## [0.37.1] - 2026-10-21

### Fixed
- **`mloop export onnx --verify` crashed the process on Windows.** Windows carries its own, older
  `onnxruntime.dll` for Windows ML; the system's library search found it, and ONNX Runtime's managed
  half failed against it with an access violation — not an error message, the end of the process.
  The native library is now loaded only by explicit path: the one installed with
  `mloop runtime install onnxruntime` (or one shipped beside the application). Without either,
  `--verify` refuses with the install command before loading anything.

## [0.37.0] - 2026-10-21

### Added
- **`mloop export onnx [exp-id]`** writes a trained model as an ONNX graph (`model.onnx`) with a
  manifest (`model.onnx.json`): each graph input and the data columns behind it, the outputs, the
  classes and how to read the answer, the experiment's metrics and weight column, and SHA-256 of the
  graph and of the source `model.zip`. The input schema is read from data (default: the experiment's
  training file) through the loader prediction uses, so a column the model does not read is not an
  input. Binary classification, multiclass classification and regression. A model with a step ONNX
  cannot express — a string column featurized as free text — is refused with its string columns
  named and nothing is written.
  `MLoop.Core` carries the conversion (`MLoop.Core.Export.OnnxExport`, `OnnxManifest`).
- **`mloop export onnx --verify`** runs the exported graph in ONNX Runtime on up to 1,000 rows and
  compares it with the model (`Score` / `Probability` within 1e-4, `PredictedLabel` exactly); a graph
  that disagrees is not kept, and the manifest records the comparison. ONNX Runtime's native library
  is an on-demand runtime: `mloop runtime install onnxruntime` (`MLoop.Core.Export.OnnxVerification`).
- **Portable training** — `mloop train --portable`, or `portable: true` under the model in `mloop.yaml`:
  train only with steps an ONNX runtime computes exactly, so the model exports. Text columns are
  featurized as counts of character n-grams (lengths 1–4, case-sensitive) instead of ML.NET's text
  featurizer, fitted once on the training rows and saved in front of the model AutoML picks. On the
  UCI SMS spam data: accuracy 0.9848, AUC 0.9948, ECE 0.009 (default featurization 0.9865, 0.9874,
  0.009), and the exported graph agrees with the model to 1e-7. Binary classification, multiclass
  classification and regression; refused on other tasks. The experiment records `portable: true`.

### Changed
- **String columns of categories are one-hot encoded instead of featurized as text.** Every untyped
  string column used to go through text featurization (word and character n-grams), so a column
  like `sex` or a log level was read as a one-word document, while the saved schema called it
  `Categorical`. Now a string column whose values do not read as language and that has at most 100
  distinct values (or distinct values on under a tenth of its rows) is a category: one-hot encoded,
  printed as `Categorical column(s): …`, and recorded as `Categorical` in the schema; any other string
  column is text and recorded as `Text`. Decided once from the whole training file. A declared
  `type:` still wins. Measured on the UCI Adult census data (eight category columns, 32,561 training
  rows, 60 s): test AUC 0.9140 → 0.9171, accuracy 0.8683 → 0.8666, calibration error (ECE)
  0.088 → 0.017 — and the model now exports to ONNX. Retrain to get the new featurization; existing
  models are unchanged.
- ML.NET 5.0.1 (AutoML, TorchSharp 0.23.1); new dependency `Microsoft.ML.OnnxConverter` 0.23.1.

### Fixed
- Training states each column-role line (`Categorical column(s): …`, `Text column(s): …`) once per
  run; a run that retried with another metric and then fell back printed it once per attempt.
- The small-data fallback trainer (used when AutoML cannot score a binary run) now applies the
  preprocessing steps that run inside training (statistical `prep` steps); it trained without them.
- `mloop promote --best` no longer picks a "best" experiment across different row weightings —
  a weighted and an unweighted accuracy describe different distributions. It refuses, lists the
  candidates by weighting, and points to `mloop evaluate --weight` and an explicit promote.

## [0.36.0] - 2026-10-21

### Added
- **Weighted training** — `mloop train --weight <column>` or `weight: <column>` in `mloop.yaml`: each
  row counts by its weight (for a row sampled with probability π, 1/π). AutoML accepts a weight
  column but does not pass it to the trainers it builds, so the training rows are split into fit and
  selection rows, each resampled in proportion to the weight, and AutoML is given the selection rows
  as its validation set. The test metrics are weighted too (binary accuracy, precision, recall, F1,
  AUC, ECE; multiclass accuracy, log loss, F1, ECE; regression R², RMSE, MAE), the experiment records
  the weight column, and the weight is never a feature. A weight the run could not honour — another
  task, `--balance`, the label itself, a column declared a feature, missing / negative / all-zero
  values — is refused before training. Binary, multiclass and regression.
- **`mloop evaluate --weight <column>`** counts test rows by a column of the test file; a weighted
  experiment's column is the default when the file has it. `--json` carries `weightColumn`.
- Automatic promotion (training, `POST /train`) does not compare metrics weighted one way with a
  production model's measured another way — it leaves production unchanged and says why; an explicit
  `mloop promote` still decides. `mloop compare` warns about such a comparison and carries
  `weightsDiffer` and each experiment's `weightColumn` in `--json`.

### Fixed
- **`POST /train` trains under the project's model definition.** The API built its training
  configuration from the request alone, so the model's `mloop.yaml` entry — column overrides, the
  weight, the ranking group and recommendation user/item columns, the forecasting horizon — was never
  read: an `ignore`d column was a feature, and a weighted model would have trained unweighted. The
  request now stands where a `mloop train` command line stands, merged with `mloop.yaml` the same way,
  and one shared step turns the definition into the training configuration for both. What the
  definition asks for that the API cannot honour is refused with 400 before a job exists: a weight
  the task would not reach, and `prep` steps, which the API does not run.
- **The quality gate told a model that learned nothing from one that only sits on its decision
  threshold.** A binary model that answers one class for every row was refused as "predicts only one
  class" whatever its AUC — so a model that ranks the classes well but whose default 0.5 cut falls on
  one side (accuracy 0.969, F1 0, AUC 0.968) got the same verdict as one that ranks like chance
  (accuracy 0.991, F1 0, AUC 0.5). AUC does not depend on the threshold, so it now decides: at or
  below 0.5 (or not recorded) the model is refused as before; above it the gate passes the model and
  warns that its predicted labels carry no information while its scores do. `train` and `promote`
  print the warning; `promote --json`, `POST /promote` and `list --json` return it as
  `qualityWarning`. The floor of the project's own metric is unchanged — on `accuracy` or `f1_score`
  such a model sits on the floor and is still refused, so the warning applies to projects that
  optimize a ranking metric.
  Training's performance diagnostics read the same rule (`BinaryDecisionShape` in `MLoop.Core`), so
  they no longer call such a model "degenerate", and now also recognise a model that answers
  *positive* for every row.

## [0.35.0] - 2026-10-20

### Changed
- **`mloop promote` applies the quality gate** training applies before it promotes. A model no better
  than a trivial baseline — at or below its metric's floor, or one that predicts a single class (on
  0.6% positives: accuracy 0.993, F1 0) — is refused with the reason and exit code 1; `--force`
  promotes it and prints the reason as a warning. Before, an explicit promote put such a model in
  production without a word. `POST /promote` applies the same gate (422, `"force": true` to override).
- `mloop promote --best` picks the best experiment the quality gate accepts: on imbalanced data a
  single-class model's accuracy outranked every model that had learned.
- FilePrepper 0.11.1, DataLens 0.19.1 (Mahalanobis outliers are judged against the distance's exact
  law, so small samples flag more rows; a single-value column's standard deviation is absent rather
  than NaN), `Microsoft.*` 10.0.13.

### Added
- **`mloop evaluate --slice-by <column>`** reports the metrics for each value of a column of the test
  data, so a regression confined to one device or one log type is not averaged away. Each slice goes
  through the same evaluation as the whole file; small slices are marked.
- **Calibration error (`ece`) for classifiers.** Training and `mloop evaluate` report the expected
  calibration error of the probabilities (binary: the positive-class probability; multiclass: the top
  class's), over ten bins. Accuracy, AUC and F1 do not see calibration, so a confidence used as a skip
  threshold had nothing behind it. Reported only — not a metric to optimize.
- **`mloop rollback`** returns production to the experiment it held before the current one, from the
  promotion history; `--to <exp>` names a former production experiment. The current production is
  backed up and the rollback recorded, so a second rollback undoes the first. An experiment that never
  served is refused — that is a promotion. `POST /rollback` (admin) does the same (409 when there is
  nothing to return to). Promotion already kept backups; going back was a manual copy.
- **`mloop list --json` carries each experiment's `metrics` and the gate's verdict** (`qualityGate`,
  `null` when it passes), so a consumer deciding promotion no longer opens files in the experiment
  directory.
- **`MLoop.Core` and `MLoop.Extensibility` are published to NuGet.org** with each release, at the
  CLI's version. A host that calls the engine in process takes a `PackageReference` and receives the
  dependency floors through the package, which a `ProjectReference` to this repository never carried.
  The `MLoop.Core` README now shows the API that exists (`AutoMLRunner`, `PredictionService`).

### Fixed
- `mloop predict` and `mloop evaluate` on a file where a categorical feature holds only yes/no values
  failed with `Could not find input column '<name>'` when training saw that column with more values
  (ML.NET's column inference merges yes/no columns into one boolean vector). The saved schema records
  how many values the column had, and a column with more than two goes back to its own text column.
  A slice from `--slice-by` is exactly such a subset, which is how the release walk found it.
- `mloop evaluate` on binary test data holding a single class reports accuracy, precision and recall
  from the counts instead of failing: ML.NET computes AUC with everything else and throws when it is
  undefined.
- `mloop train` says why the gate left a model in staging when the reason is not its metric's floor
  (a single-class model with AUC 0.95 was saved to staging with no reason given), and a model sitting
  exactly on the floor gets the floor message the gate applied.
- `experiment-index.json` names the metric its `bestMetric` holds. A configured spelling the metrics
  file does not use (`f1` for `f1_score`) recorded the accuracy under the name `f1`.

### Security
- `System.Security.Cryptography.Pkcs` and `System.Security.Cryptography.Xml` resolve at 10.0.13
  (GHSA-846f-f764-76mx, GHSA-2fmv-rv2c-59x7, GHSA-7hqp-7c5m-4vq6, GHSA-r955-8q5j-3h83). 0.34.0 brought
  10.0.10 in through FilePrepper; FilePrepper 0.11.1 takes them at 10.0.13.

## [0.34.0] - 2026-10-01

### Added
- **`--records` reads a JSON file whose rows sit in a nested array** (`train`, `predict`, `evaluate`,
  `info`, `analyze`, `detect`). `--records data.paragraphs.qas.answers` reads each answer of a reading-comprehension file as
  a row that keeps its question and passage (`qas.question`, `paragraphs.context`), so such a file
  trains question answering as it is published. A JSON file that is not a table is refused with the
  paths that would read it; `--records` on a file that is not JSON is refused rather than ignored.
  This comes with FilePrepper 0.9.0 (now 0.10.0).
- **Question answering says when its answers cannot be written.** ML.NET's question-answering model
  decodes its answer one byte per character, so an answer in Korean, Chinese, Japanese or Cyrillic
  comes out with those characters missing. Training now counts such answers and says where
  `exact_match` stops; the guide's Question Answering section explains the limit.
- **An image-classification model predicts a folder of images.** `mloop predict new-photos/` scores
  every image under the folder, subfolders included; before, the images had to be listed by hand in a
  CSV with an `ImagePath` column (still read). After training, the suggested predict line names a
  folder for image models instead of `data.csv`.
- **A folder of data files is one table** for `train`, `predict`, `evaluate`, `info`, `analyze` and
  `detect`: every data file directly inside it, in file-name order, each read as it would be alone. An
  export that writes a file per record (one JSON object per measurement) could not be given at all
  before — several hundred paths on a command line exceed the operating system's length limit — and
  `info` refused a folder. Mixed formats or different columns are refused by name. This comes with
  FilePrepper 0.11.0.
- **SVMlight / libsvm data** — `label [qid:q] index:value …`, the format learning-to-rank sets are
  published in — is read by every command, under its own extensions or by its lines (`rank.train`).
  It becomes `label`, `qid` and one `f<index>` column per feature; a group-size file beside it
  (`rank.train.query`) supplies the groups. Such a file was read as one CSV column headed by its first
  row, and training went ahead on zero features. This comes with FilePrepper 0.11.0.
- **`mloop info <folder>` describes an image-classification folder** — images per class folder and
  each class's share — instead of reporting an existing folder as "File not found" with a tip about
  relative paths. Any other folder is named as a folder.

### Changed
- **Rows with an empty label are dropped by default wherever the label is a target** — regression,
  ranking, recommendation, NER and question answering, not only classification. A row without a
  target can neither teach a model nor score it: a recommendation set whose unrated visits (22% of
  rows) stayed in scored R² 0.54, and −0.64 once they were left out. A time series keeps its rows.
  `--drop-missing-labels false` keeps them.
- **A multiclass prediction CSV names each score column after its class** — `Score.cat`, `Score.dog`
  rather than `Score.0`, `Score.1` — the names `--json` already reported. The CSV said which slot
  scored highest but not which class that slot was. A model whose classes cannot be recovered keeps
  the numbered columns.

### Fixed
- **A column that holds a list in every row is not read as a feature.** A Parquet or JSON list column
  arrives as JSON text, and was featurized as text — TF-IDF over brackets and digits. It is now
  excluded like a constant or identifier column, with the reason `Structured` and a line saying it
  must be unpacked first. A label column of lists is refused with that reason instead of the advice
  to use classification.
- **`mloop info` shows the columns training will drop**, as `Excluded (Constant)`,
  `Excluded (Identifier)` and so on, from the same decision training uses. It called them features.
- **`train --group-column` is saved to `mloop.yaml`** with the label it was given alongside, so the
  next run groups the rows the same way; the scaffold's `query_id` stayed behind.
- **`predict` no longer shows an input column as the prediction distribution.** Without a predicted
  class the distribution counted the file's first column — a ranking file's query ids. It now
  appears only for a class answer. The ranking log counts feature values, not merged columns
  ("1 features" for 209).
- **`predict` no longer announces a strategy for unseen category values when there were none.** Every
  image-classification prediction printed "Low unknown value ratio (0.00%) - safely replacing with
  most frequent values"; the line now appears only when some value was actually unseen, and goes
  where the rest of the command's narration goes.
- **Image-classification training shows its progress.** The bar stood at 0% for the whole fit (ten
  minutes on 970 images on a CPU), because this trainer reports progress through a callback rather
  than its log. It now counts the images featurized, with the time left, and then the epochs, which
  are named as a cap of 200 since the trainer stops early once accuracy stops improving. `train --json`
  carries the same as `featurize` events (`step`, `steps`) and `epoch` events with `stopsEarly`.
- **Image-classification `train` and `evaluate` say what they loaded.** The image loader's count of
  images per class, and its warnings about empty or sparse class folders, a single class or severe
  imbalance, were written to no one.
- **A model no better than the mean is no longer promoted when it is judged on an error metric.**
  rmse, mae and mse have no floor of their own, so a recommendation model at R² −0.65 — worse than
  answering every row with the mean rating — reached production. R² at or below 0 now keeps such a
  model in staging, with the reason.
- **A failed training run no longer suggests what cannot help it.** The fallback advice offered
  `--time 30` to deep-learning and time-series runs, whose fit takes no time budget, and
  `mloop analyze profile` for an image directory. It now fits the task. An object-detection directory
  with neither layout also says that a YOLO dataset would be read, not only that no COCO file was found.
- **Training no longer advises about columns the model does not read.** For time-series tasks,
  recommendation and deep-learning tasks, the samples-per-feature rule ("300 samples for 35 features"),
  the suggestion to drop an index-like column and the announcements of excluded columns all counted
  columns the model never sees. They now apply only where the model reads the table's columns as
  features, and a time-series run shows its test split as not used, since a series keeps its order.
- **A time-series anomaly prediction CSV answers in two columns.** The detectors' one output vector
  matched none of the columns the CSV keeps, so the file held every input column twice, the loader's
  feature vector slot by slot, and the answer as three unnamed numbers; the preview collapsed into
  one-character columns. It now answers in `PredictedLabel` (anomaly or not) and `Score`, the columns
  anomaly detection uses, with the values `--json` reports as `isAnomaly` and `anomalyScore`.
- **Forecasting, time-series anomaly and recommendation training no longer run an auto-time probe.**
  Their fit takes no time budget, so the probe trained the model a second time (its warnings printed
  twice), a 2-second time-series fit was reported as a 19-second probe and a 114-second main pass, and
  a 2-second matrix factorization as "converged in probe phase (30s)" after a 143-second estimate. The
  "Time Limit" row now says the budget is not used and why.
- **A JSON file holding one object reads as a one-row table** (with FilePrepper 0.10.0), so a file per
  record — a sensor reading per timestamp — can be passed with `--data`. `--records` is suggested only
  when the file has rows under a path to give.
- **`mloop train --data a --data b` merges JSON, Parquet, Excel and TSV files.** The merge read every
  file as CSV text, so two JSON files of the same shape were refused as having no column in common,
  with each file's raw JSON listed as its columns. Each file is now read as a table first (with
  `--records` when given), and the merge report names the files as given.
- **`mloop info` without a label no longer calls a column the label.** ML.NET's type inference needs
  one, so `info` guesses (a common label name, else the last column) — and reported that guess as the
  column's purpose. A guessed column now shows the purpose its type gives it.
- **`mloop train` suggests a predict command that reads the data the same way.** After training on
  `--records`, the next steps said `mloop predict data.csv`; they now carry the record path.
- **A deep-learning task whose pretrained weights cannot be downloaded says how to fetch them.**
  ML.NET's downloader fails on large files (dotnet/machinelearning#6980), and its error named the URL
  but not where the file is expected; the advice after it pointed at listing and promoting models.
  Training now names the file, the folder it is looked for in (`mlnet` in the temp folder) and the
  command that fetches it, and suggests only that. A training error also no longer prints its cause a
  second time as an "Inner exception".
- **Question answering trains, predicts and evaluates.** The trainer was given only the passage and
  the question, so it read the answer and its position from columns that had to be named
  `TrainingAnswer` and `AnswerStart`; training reported no metric, so the quality gate had nothing to
  judge; `mloop predict` read the answer vector as a single number; and `mloop evaluate` refused the
  task. The answer is now the label, its start position is the column that locates it in the passage,
  the passage is the longer text column, and training and `evaluate` score the best answer with
  `char_f1` (primary) and `exact_match`, the KorQuAD rules. A prediction's `predictedLabel` is the
  answer — in the CSV as in `--json`, with its `Score`. The start position is found among any number
  of numeric columns: a file that also carries numeric ids keeps each column under its own name. See
  the guide's Question Answering section for the data format.

## [0.33.0] - 2026-10-01

### Added
- **NER works end to end.** No NER model could be trained: the data check refused a text label for
  any task that was not classification, and past it the trainer received one key per sentence where it
  needs one per word; prediction then rejected every answer as degenerate, and `evaluate` looked for a
  score column NER does not have. The label is now read as space-separated tags, one per word; the
  answer is written the same way (with the trainer's "no entity" answer written `O`); and training and
  `evaluate` score word by word with one shared rule. The data format is in the guide.
- **Deep-learning training shows how far along it is.** Text classification, NER, sentence
  similarity, question answering and object detection train for a fixed number of epochs in one
  call, and the progress bar sat at 0% for the whole of it — over four minutes on eleven thousand
  sentence pairs with nothing to say whether that was a tenth of the run or all of it. The bar now
  counts epochs (`Epoch 3/10 — about 9m left`), `train` says up front that the time limit does not
  shorten these tasks, and `--json` emits a `phase: "epoch"` event with `epoch` and `epochs` per epoch.
- **Data files in Parquet, Excel, JSON and TSV.** `train`, `predict`, `evaluate`, `info`, `analyze`
  and `detect` used to read every data file as comma-separated text, so a Parquet file reached the
  label check as binary noise listed as "available columns". Such a file is now read as the table it
  is — once per run, whichever step asks — and a Parquet struct column becomes one column per leaf
  named by its dotted path (`labels.label`). Comes with FilePrepper 0.8.0.
- **`mloop detect` marks an anomaly that sits near an end of the series.** The score is least
  reliable in the first and last few points, where the transform's boundary handling moves it most,
  and a flag seen only there used to look exactly like any other. The table now marks such an index
  with `*` and says why; `--output` gains a `NearEdge` column and `--json` a `nearEdge` field on
  every point.

### Fixed
- **`mloop trigger check`, `mloop feedback list`/`metrics` and `mloop sample` default to the `default`
  model** like every other command, instead of refusing to run without a model name.
- **Predictions served by `mloop serve` reach logs, feedback and retraining triggers.** `POST /predict`
  never logged what it answered, so `GET /logs`, feedback and triggers saw only batch predictions made
  with `mloop predict --log`, and nothing anywhere showed the id feedback needs. Served predictions are
  now logged (`Serve:LogPredictions`, on by default), each row carries its `predictionId`, and the id
  appears in `mloop logs` (an ID column and an `id` field in `--json`) and in `GET /logs`. A failure to
  log does not fail the prediction; the row comes back without an id and with a warning.
- **`POST /feedback` answers a caller's mistake as one.** An id no log holds returned 500; it is now
  404, and a request without an id or an actual value is 400. `mloop feedback add` names where ids are
  found. Recording feedback again for the same prediction is a correction: feedback accuracy now counts
  each prediction once, with its latest actual value, where it used to count every entry.
- **`mloop trigger check` and `GET /trigger` check the same conditions.** The CLI checked feedback
  accuracy and volume while the API checked only the time since training, so one model got two answers.
  Both now check all three by default, and `--accuracy`/`--feedback` replace the default they name
  instead of discarding the others.
- **`mloop preprocess --incremental` no longer reports cleaned data it did not write.** The workflow
  discovers rules with every detector, but most rule types have no application strategy yet. A run
  whose rules changed nothing still printed "Incremental preprocessing complete!" and a
  `cleaned_data.csv` path — a copy identical to the input, or a file that was never written when no
  rule was approved — and exited 0. Now a run that finds or approves no rule says so and exits 0; a
  run whose approved rules could not change the data writes no cleaned copy, says how many rules have
  no strategy yet, and exits 1. A run that did change rows reports how many.
- **Running `mloop` with no arguments lists every command.** The list was a hand-kept copy that had
  fallen six commands behind (`analyze`, `detect`, `features`, `token`, `new`, `runtime`) and it now
  comes from the command tree itself.
- **A NER model that tags every word `O` is no longer promoted.** NER's label cell holds a sentence of
  tags, and the quality gate counted the cells as classes: N was the number of distinct sentences, its
  1/N floor effectively zero, and on real data a model scoring the share of `O` (0.85 of words, 0.12
  averaged over tags) went to production. The label schema now records the tags as the classes, the
  gate's floors apply per tag, and NER's primary metric is the tag average (`macro_accuracy`).
- **Deep-learning training no longer talks about a time budget it does not use.** `train` printed an
  auto-time estimate ("⏱ Auto-time: 60s estimated") and a time limit for tasks that train a fixed
  number of epochs; the configuration table now says the limit is not used, and no estimate is shown.
- **Deep-learning training no longer runs twice under auto-time.** The probe phase is a short trial
  run for AutoML; a deep-learning fit ignores its budget, so the probe was a whole training thrown
  away before the real one (NER: 1,050 s → 719 s on the same data). These tasks now skip it.
- **Metric names say what they measure.** The image, text and NER trainers stored the class-averaged
  accuracy as `accuracy`; it is `macro_accuracy`, as it is for multiclass classification.
- **Error suggestions name commands that exist.** A failed `train` suggested `mloop experiments list`
  and `mloop experiments promote` (there is no such group — `mloop list`, `mloop promote`),
  `mloop analyze` without the aspect it requires, and `mloop init` without a project name; its
  "What you can try" list numbered every item `1.`; and a missing label column — whose message names
  the model — drew the suggestions for a missing model. Every `mloop …` a message suggests is now
  parsed against the real command tree by a test.
- **A text model reads the text, not the first string columns.** Text classification, NER,
  sentence similarity and question answering took the first string columns in the file. On a
  sentence-similarity set whose first column was a source tag, the model was trained on the tag and
  one sentence, never saw the second — and, the tag correlating with the label, scored well enough to
  be promoted. Columns declared `type: text` in `mloop.yaml` are now read first, in order; otherwise
  string columns that read as language are chosen over short codes, and a fallback pick is reported
  as a warning with how to declare it. The rule that decides "reads as language" is the one that
  already decided text featurization and `info`'s column purposes, now in one place.
- **A model with no metric to judge it by no longer replaces production.** Sentence similarity and
  NER had no primary metric, so `metric: auto` stayed unresolved; the quality gate had nothing to
  test and the production comparison nothing to compare, and promotion went ahead anyway — a model at
  R² −0.30 replaced one at 0.73, reported as "Better auto than exp-002". Both tasks now have a
  primary metric (R² and micro-accuracy), and any task still without one keeps its production model
  and says to promote explicitly. The first model of a project is still promoted.
- **`train` and `info` describe the file you named.** The file name and size in `train`'s data
  summary and configuration, and the size and modification time in `info`, came from whichever copy
  MLoop was reading at that moment — a UTF-8 conversion of a CP949 file, a flattened or split copy —
  so they could name a temporary file and report its size.
- **`mloop detect` finds the period of a series that is not a whole number of cycles long.** The
  estimate used to land a few steps short on such a series — 60 for a period of 63 over 300 points
  — because the autocorrelation it compared shrinks with the lag and nothing corrected for that.
  This comes with DataLens 0.18.0.
- **An image built by `mloop docker` for one model now serves that model.** The generated Dockerfile
  sets `MLOOP_MODEL_NAME`, and that variable existed precisely so an image serving a single model
  would not need it named on every request — but the only code reading it was in the CLI, and what
  the image runs is the API. The API worked out which model a request meant in eight places of its
  own, none of which looked at the environment, so an image built for `churn` answered any request
  without `?name=` from `default`. Which model a request means is now decided in one place, and the
  Dockerfile writes the variable name from that same place rather than spelling it again.
- **A header with a quoted comma, or in a legacy Korean encoding, no longer loses a column from the
  feature set.** The reader that supplies column names to the exclusion logic split on every comma
  and read every file as UTF-8. So `"Last, First",Age` became three names where ML.NET and every
  other reader in MLoop see two, and a CP949 header came back mojibake. That does not surface as a
  wrong list: the exclusion decision is computed by comparing these names before and after each
  removal step, and a name that does not match the real schema is skipped in silence by the code
  that applies exclusions — leaving training and prediction disagreeing about which columns are
  features. It now uses the same RFC 4180 parser the rest of the product uses, and detects the
  file's encoding the way the loader does. The removal chain's own second parser is gone with it.
- **A text classifier is now protected like the other classifiers.** Two different questions were
  being asked under one name — "does this model predict a class", and "can I count its classes in a
  column of the training file" — and the answers had drifted apart on `text-classification`. It
  counted as a classification task everywhere the first question was asked and nowhere the second
  was, so a text-classification run got an unsupervised split instead of a stratified one (a rare
  class can land entirely outside the test set, and every metric that needs it is then undefined),
  was described to the time estimator as having no classes, was sampled uniformly under
  `--max-rows`, and skipped the per-class minimum-sample check that rejects a provably untrainable
  set. It now gets all four. Tasks whose labels are directory names — image classification — still
  answer the first question and not the second, which is what they are.
- **One reading of the `task` field, instead of four.** Every part of MLoop that asks which task a
  model is normalized the string itself, and the normalizations disagreed: one accepted
  `binary_classification` and rejected a stray space, another did the reverse, a third accepted
  `BinaryClassification`, and a fourth compared without folding case at all. The visible result was
  a configuration that half the product accepted: `task: classification` — the spelling that
  predates `binary-classification` — was rejected by `mloop validate` while `train` and `evaluate`
  ran it as binary classification, and `task: binary_classification` split the other way. A task
  name is now folded in one place, tolerating surrounding space, any casing, underscores in place of
  hyphens, and the run-together spellings (`BinaryClassification`, `binary classification`). Those
  spellings are accepted, not advertised: the vocabulary shown in help and error messages is
  unchanged.
- **`mloop train` checks the task name.** It never had: `TaskTypes` was consulted there only for the
  `--task` help text, so `--task regresion` reached each predicate in turn, was answered "no" by all
  of them, and trained something nobody asked for. `init` and `validate` have always checked.
- **`mloop status` sees every kind of prediction, not just tabular ones.** Three commands write into
  `predictions/` — scored rows, per-image detections for object detection, and a forecast — and
  status looked for the first convention alone, so a project whose model does object detection or
  forecasting reported no predictions however often it had run. A second count in the same command
  had the mirror problem: it matched any `.csv` in the directory, missing detections while counting
  a stray file a user had left there. Both now ask one place what a prediction file is called, which
  is also what the writers ask, so a fourth kind cannot be added without the reader hearing about
  it. `docs/GUIDE.md` described a fifth convention, from before models were named
  (`predictions/predictions-TIMESTAMP.csv`, no model prefix); it now matches what is written.
- **`mloop evaluate`'s overfitting warning works for every task, and no longer fires backwards.** It
  resolved the metric to compare from a list of three tasks, so for the other eight — image and text
  classification, ranking, recommendation, clustering, anomaly detection, forecasting, time-series
  anomaly — it found nothing to compare and said nothing, which reads exactly like a clean model. It
  also compared the two scores by absolute difference, so a model that did *better* on the test data
  than on its training data was reported as possibly overfitting, which is the one thing that result
  cannot mean. Widening it was not a substitution: the threshold it used is only meaningful for
  metrics on a fixed scale like accuracy or R². A task scored by `rmse`, `mae` or `average_distance`
  carries the label's own units, where a fixed gap of 0.1 says more about the unit than the model —
  those are now compared as a proportion of the training score. The warning names the direction it
  found.

### Changed
- **DataLens 0.18.0** (from 0.14.3), on UInsight 0.24.0. Two of its changes reach `mloop analyze` and
  `mloop info --analyze` output: K-means and HDBSCAN clusters are numbered by first appearance (the
  grouping is unchanged; the numbers attached to it can differ from 0.32.0), and permutation
  importance no longer depends on column order or names, so its scores for the same data differ from
  earlier versions. An analysis step that fails is now reported as a warning instead of being
  dropped silently.
- **`mloop detect` runs on every platform, and finds what it used to miss.** Detection moved
  from ML.NET's SR-CNN to spectral residual scoring in the statistics engine MLoop already uses for
  analysis (through DataLens 0.15.0). The old path needed ML.NET's FFT native — absent on Apple
  silicon, and dependent on a system OpenMP runtime on Linux — so `detect` did not run there at all.
  It also scored poorly: on a period-7 sawtooth with one spike it flagged fifteen points around the
  spike, it missed a spike on a noisy sine and a level shift entirely, and given the true period of
  a seasonal series it reported 191 of 300 points as anomalies. On the same series the new path
  flags the one sawtooth spike, the sine spike, the shift, and exactly the four injected points.
  **Breaking for `detect` options and output**: `--threshold` is now the score a point must exceed,
  greater than 0 with a default of 3 (it was a value in [0, 1], default 0.3, on a different scale);
  `--sensitivity` is the coverage of the control limits in percent, default 99.73 (3 sigma);
  `--period` is removed — the score assumes no period, so a supplied one could not change the
  result, and passing it now explains that instead of failing to parse. The series' own period is
  reported (`period`, `null` when it has none). Each point carries `score`, `expectedValue` and
  `controlLower`/`controlUpper`; `marginLower`/`marginUpper` and `residualSigma` are gone — the
  margin was the old detector's decision gate and has no counterpart, and the control limits are now
  the engine's band at the requested coverage, the same robust residual scale MLoop used to compute
  itself.
- **`mloop list` and `mloop status` agree about how old something is.** Each had its own version:
  they switched from "9d ago" to a calendar date at different ages (a week against a month), only
  one of them said "just now", and — the one that made them contradict each other — one rendered its
  date in the reader's zone while the other rendered it in UTC, so the same file could be shown with
  two different dates depending on which command you asked. Ages are now rendered in one place and
  stay ages all the way out (`4mo ago`, `2y ago`) rather than becoming a date, which is what removes
  the zone question rather than managing it. The exact instant is still in `--json` and in the
  experiment report. `mloop sample stats` now says which zone its date range is in.
- **`mloop validate` now applies the model-name rule the rest of MLoop applies.** The rule existed
  in three places and one of them enforced almost none of it: `validate` accepted uppercase,
  underscores, a leading underscore, any length and every reserved name — while printing "must be
  lowercase alphanumeric with hyphens only", the rule it did not implement. A configuration naming
  a model `MyModel_v2` passed validation and then failed at the first training run, which is the
  one outcome validation exists to prevent. **A `mloop.yaml` that validated clean before may now
  report an error**; the name it names is one `mloop train` would have refused. Rejections say
  which rule was broken, so a reserved name (`staging`, `production`, `temp`, `cache`, `index`,
  `registry`) is told it is reserved rather than told a rule it already satisfies.
- **`mloop list` says which way its Metric column reads.** The trial leaderboard already said it;
  the experiment table named the metric and left the direction to be guessed, so a reader meeting
  `rmse` or `average_distance` for the first time had nothing to tell them the smallest number is
  the best one. A note under the table now gives the direction for each metric present, and says
  the list is ordered newest-first rather than ranked. A metric MLoop does not recognize is left
  unclaimed rather than assumed to be maximized.

### Removed
- **`mloop pipeline`.** It never ran a step: the handlers it dispatched to were placeholders that
  reported an experiment named `exp-pipeline-001`, a metric of 0.95 and a completed promotion without
  training or promoting anything, and the command exited 0. Each MLoop command already runs as its own
  process, so a shell script or CI job is the pipeline — `mloop train`, `mloop evaluate` and
  `mloop promote` in sequence, stopping on the first non-zero exit code.
- **`mloop extensions`.** It printed that listing extensions was not yet available. Scripts are found by
  where they are: `.mloop/scripts/hooks/`, `metrics/`, `preprocess/`, `detectors/`.

### Security
- **Build-time: `Microsoft.SourceLink.GitHub` 10.0.401** (from 10.0.301), which brings
  `Microsoft.Build.Tasks.Git` out of the range of GHSA-23fw-v26w-5fgq (CVE-2026-62900). The reference
  is build-only (`PrivateAssets="All"`) and never reached a consumer's package graph, but every build
  that included MLoop's projects reported the advisory as a warning.

## [0.32.0] - 2026-09-20

### Added
- **Every experiment records the fingerprint of the data it was trained on.** `config.json` (and
  the copy of it inside `metadata.json`) now carries `dataFileHash`, the SHA-256 of the training
  file's content written as `sha256:<hex>`, taken over the file as you have it — before any
  encoding conversion or split MLoop makes for its own use — so anyone holding the original can
  confirm that `exp-007` really was trained on it, and a regenerated `datasets/train.csv` can no
  longer pass for the one an experiment saw. The report shows it under `Data hash`, and
  `GET /experiments/{id}` returns it in the config block. Absent on experiments recorded before
  this version and on image-directory tasks, whose input is not one file.
- **Every experiment now leaves a `report.md` beside its model.** Training writes one Markdown page
  into `models/<name>/staging/exp-NNN/` that says what the JSON files in that directory say, in the
  order a person asks: status and data, the winning trainer and its pipeline, the metrics, every
  trial ranked as the leaderboard ranks it, and each input column with its role — including the
  reason a column was excluded. It renders in any Markdown viewer, including the repository host a
  project is pushed to. The page is written for failed experiments too, and it is only a rendering:
  no command reads it, so a missing or stale report breaks nothing, and a failure to write it costs
  a warning rather than the experiment. The training summary prints its path under `Model saved to:`,
  and the `--json` `result` event carries it as `reportPath` (omitted when no report was written).
  When binary classification had to optimize F1 because AUC was undefined on the data, the page's
  "Optimized for" line names the metric the search actually used and the one that was asked
  (`f1_score (accuracy was asked)`), rather than repeating the configuration.

### Fixed
- **`mloop train` could end with an unhandled exception instead of a model.** The progress bar's
  description is Spectre markup, and the trainer name was written into it unescaped. When AutoML
  cannot run and training falls back to a direct pipeline, that name carries a bracketed note
  (`SdcaLogisticRegression [manual fallback: AutoML AUC failure]`), which reads as an unknown style:
  `Could not find color or style 'manual'`, thrown on the progress refresh thread, which ends the
  process after the model was already trained. Whether a refresh landed on that description before
  training finished was a race, so it surfaced only on a slow machine. Every value the descriptions
  carry — trainer, metric, model name — is escaped now, and one place builds them.
- **The small-data fallback trainer gives the same model on every run.** When AutoML cannot run
  on a binary-classification dataset, MLoop trains an SDCA logistic regression directly. ML.NET runs
  that solver on several threads by default, and on the small data this fallback exists for it does
  not converge and its result varies between runs of the same seed. It now trains on one thread.
- **An auto-time run recorded a time limit it never ran under.** When the budget was left to
  auto-time — the default — the experiment's `config.json` and its `report.md` stated the
  configured default of 300 seconds, while the run was granted a short probe and, when it helped, a
  main pass. The record now holds the budget the run was actually granted (probe plus main) and
  says it was chosen automatically (`autoTime: true`; the report reads `auto — 77 s granted`). The
  flag also appears in `GET /experiments/{id}` and `mloop compare --json`. Experiments recorded by
  earlier versions keep the value they stored.
- **`mloop evaluate` echoed the test file with mixed separators** (`…\demo\datasets/train.csv`)
  when given a relative path; the resolved path is now normalized.
- **A training job started through `POST /train` was promoted on the wrong metric.** The job
  passed the first metric in its result to the promotion check — for binary classification that is
  always accuracy — so a job trained for AUC or F1 replaced production (or did not) by comparing
  accuracy. It now compares on the metric the job optimized, as `mloop train` does.
- **The multiclass performance diagnostic never showed a log loss.** It looked the metric up as
  `LogLoss` while the trainer writes `log_loss`; the two never met. The diagnostic now reads every
  metric by its canonical name under whatever spelling the dictionary carries, through the same
  vocabulary training uses, and its private lists of alternate spellings are gone.
- **The `mloop train` option reference in the guide named an option that does not exist
  (`--output`), gave `--time` a default it does not have, and omitted eleven options the command
  accepts** (`--name`, `--label`, `--task`, `--data`, `--max-rows`, `--sampling-strategy`, `--seed`,
  `--group-column`, `--analyze-data`, `--generate-script`, `--json`). The block is rewritten from
  the command's own option definitions.
- **A metric name the optimizer did not recognize was replaced by the task default without a
  word, and every record kept the name you wrote.** The optimizer knew `f1_score`, `f1`, `r2`;
  `mloop validate` accepted `rSquared`, `recall`, `log-loss`; the shipped examples wrote `F1Score`.
  Any spelling outside the optimizer's own short list fell through to accuracy (or macro accuracy,
  or R²) silently, so `metric: F1Score` — the churn example's own configuration — trained for
  accuracy while `config.json`, the experiment index and `mloop list` all said F1, and the index
  paired that label with the accuracy value. There is now one vocabulary (`MetricNames`), read the
  same way by training, `validate`, and the records: spelling, case and separators do not matter
  (`F1Score`, `f1-score` and `f1_score` are one name), the ML.NET enum names and the usual short
  forms are accepted, and `auto` resolves to the task's primary metric. **A name that is still
  unknown, or one the task cannot optimize (`auc` for regression), is refused before training
  starts** — by `mloop train` (exit 1, with the names the task accepts and a "did you mean" for a
  likely typo), by `POST /train` (400), and by `mloop validate` (an error, no longer a warning).
  Substituting the task default would have repeated the original defect for every typo: the run
  optimizes one metric while its records name another. `mape` joins the vocabulary, since
  forecasting reports it. The experiment config, the leaderboard and the index now
  record the canonical name of the metric that was actually optimized, so the index's score is the
  value of the metric it is labelled with. Index entries written by earlier versions are left as
  they are: their score was the metric that really was optimized, only the label was wrong, and
  rewriting them would pair old accuracy values with new F1 ones under one name. The examples now
  spell the metric `f1_score`.

### Changed
- **Identifier columns are no longer trained on.** A text column in which every row carries its
  own distinct, whitespace-free value — a customer id, an order number, a UUID — was being
  featurized as TF-IDF n-grams: ML.NET's column inference marked it as ignorable, and the training
  path then turned every text column back into a text feature, so the id column reached the
  trainer as noise, and as a leak path wherever the id encodes anything about the label. It now
  joins the DateTime, sparse and constant columns that training drops, is reported under
  `Excluded from features (Identifier)`, and is recorded in the model's schema so `predict` and
  `evaluate` drop it the same way. Numeric columns are never treated as identifiers (measurements
  are distinct per row too), a value with whitespace or a single repeated value disqualifies the
  column, fewer than 20 rows is not enough evidence, and a column you named as a ranking group,
  a recommendation user or item, or gave an explicit type in `mloop.yaml` is never dropped.
  A model trained on this version therefore has one fewer feature than the same data produced
  before; **models trained on earlier versions keep working**, because they replay their own
  recorded schema at prediction time — retrain when you want the id column gone from them.
- **`mloop serve` no longer writes fields that have no value.** Every response from the server —
  `POST /predict` rows and envelope included — now omits a field whose value would be `null`, the
  same rule every `--json` command in the CLI already applies. Until now the same prediction row had
  two shapes depending on which surface you asked: `{"predictedLabel":"cat",…,"confidence":0.5}`
  from the CLI and `{"predictedLabel":"cat",…,"score":null,"clusterId":null,"distances":null,…}`
  over HTTP, and the response contract had to tell readers to treat absent and `null` as the same
  thing. Now there is one shape. **Migration**: a consumer that tested a key's *presence* to decide
  whether a row carries a score, a cluster id or a band must test for the *value* instead — which
  `docs/PREDICT-RESPONSE.md` has asked for since the contract was written. A consumer that read
  `null` as "no value" is unaffected; one that reads responses from both this version and earlier
  ones should treat `null` as absent.
- **DataLens 0.14.3** (from 0.13.5), which follows UInsight 0.20.1. DataLens now routes on the file
  extension and refuses one it does not recognise (`.csv`, `.tsv`, `.json`, `.xlsx`, `.xls` are
  accepted); `mloop info --analyze` and `mloop analyze` report that refusal as the analysis being
  unavailable for that file, as they already did for any other DataLens failure, and the rest of the
  command runs unaffected.
- **The Jarque-Bera normality test now reports correct numbers.** It was computed from bias-adjusted
  skewness and kurtosis where the test uses the plain moment ratios, and its p-value lost the tail
  beyond a statistic of about 74. The corrected statistic and p-value appear in the distribution
  section of `mloop info --analyze` and in `mloop analyze distribution`; because a column is called
  normal only when no test rejects, a column near the significance level can now be reported with a
  different normality verdict and shape than before.
- **A converted temp file keeps its extension.** When a CP949/EUC-KR file is converted to UTF-8 the
  intermediate copy was named `.tmp`; it now carries the original extension, so anything downstream
  that routes on it sees a `.csv`.
- **The "possible ID column" hint named a flag that does not exist.** It suggested
  `mloop train ... --exclude <column>`; the command that declares a column excluded is
  `mloop features select --drop <column>`, and the hint now says so.

## [0.31.0] - 2026-09-11

### Fixed
- **Time-series commands failed on Linux and macOS with a message naming a type initializer.**
  Forecasting, time-series anomaly detection and `mloop detect` all route their FFT through ML.NET's
  MKL native library, and what that library ships is not the same everywhere: on Linux it expects an
  OpenMP runtime to already be installed on the system rather than bundling one, and for macOS on
  Apple silicon no build is published at all. What a user saw was
  `The type initializer for 'FftUtils' threw an exception` — which names neither the missing runtime
  nor a way to supply it. These commands now report which runtime is missing and how to install it
  where installing it is possible, and say plainly where it is not.
  The condition itself was already known to five separate copies of a probe in the test suites and to
  no part of the product; it now has one home that both consume, so a new call site cannot silently
  miss it. The prerequisite is documented in the README, and containers generated by `mloop docker`
  install the runtime themselves — previously an image built from this project could not train a
  forecasting or time-series-anomaly model at all.

- **A schema validation failure called itself a warning.** `predict` and `evaluate` printed the
  problem under a yellow `Warning:` label, directly below a red `Schema Validation Failed:` heading
  and directly above a non-zero exit — three answers to the question of whether the run had succeeded.
  The message now carries the severity the exit code does.

### Changed
- **`predict --json` and `evaluate --json` left stdout empty when the data did not match the
  model's schema.** The mismatch was rendered to the terminal in four parts and the command exited
  non-zero with nothing on stdout, so a consumer parsing that got the same failure as parsing prose
  — on the one failure these commands exist to report clearly. Both now emit the finding as a
  document at that exit. The check sits behind a successfully loaded production model, which is why
  the existing guard for this contract never reached it: it points its commands at a project with
  no model at all.
- **Schema validation printed its failure in one language and its advice in two.** The result
  carried a pair of message fields, and only one of them was ever rendered — so the sentence a user
  read was in Korean while the suggestions beneath it arrived in both Korean and English, one of
  them a duplicate of the other. Both `predict` and `evaluate` show this. The pair is now a single
  field, the surface is English throughout like the rest of the CLI, and the one hint that existed
  only in Korean was kept rather than dropped with its neighbours. `CONTRIBUTING.md` states the
  convention so the next surface does not have to rediscover it.
- **A generated Dockerfile pins the CLI version it installs.** It ran `dotnet tool install mloop`
  with no version, so the image was built against whatever happened to be newest on nuget.org at
  build time rather than the release that generated the file — a project on one version could
  produce an image serving another, and rebuilding an old directory would not reproduce the image
  that was tested. The generated file now carries `ARG CLI_VERSION` set to the generating CLI's
  own version, overridable with `--build-arg` for anyone who wants a different one.
- **`mloop list` marks the production row with `prod` rather than `Production`.** The whole-model
  view is tight at eighty columns and this cell spent ten of them to say one bit. The column is not
  removable: the ID cell distinguishes the production experiment by colour alone, which is absent
  under `NO_COLOR`, in a pipe, and for a reader with a colour vision deficiency — the word is that
  fact's only accessible channel, so it was shortened instead of dropped. A row whose trainer name
  is long still wraps; that width belongs to the trainer, not to this.
- **`mloop list` shows the trainer in the Trainer column, and the pipeline underneath the table.**
  What AutoML records is the whole pipeline it assembled
  (`ReplaceMissingValues=>Concatenate=>FastTreeBinary`), and printing that in a narrow column wrapped
  a single row across five lines in an eighty-column terminal, carrying every other cell of the row
  down with it. The column now shows the trainer — the last stage — and the full pipeline, together
  with the reason a fallback trainer stood in, is printed under the table keyed by the row it belongs
  to, so nothing is lost. The metric's name has likewise moved from every row into the column heading
  when all rows share one, which is what leaves the column enough width. A listing for a single model
  now fits one line per row.
- **A timestamp meant for a person now says which zone it is in.** Nine places rendered one in the
  reader's local time and none of them said so, while the directory and partition names written
  beside them are in UTC — so a single `promote` printed two times that differed by the length of a
  time zone, with nothing to indicate they were the same instant. Displayed timestamps now carry a
  numeric offset (`2026-09-09 15:36:17 +09:00`); in the dense listings the offset appears once in the
  column heading (`Timestamp (+09:00)`) rather than on every row, which keeps a row to one line in an
  eighty-column terminal. Names of files the product writes for you to find are unchanged and remain
  local, and storage keys remain UTC — the distinction is who reads the value, not whether it is a
  file name.

### Fixed
- **`mloop docker` generated a Dockerfile that could not be built.** The final stage added the
  Microsoft package feed for Debian 12 to a base image that is Ubuntu, then installed a full .NET
  SDK on top of a runtime image that already carries the runtime; `apt-get` could not resolve the
  result and the build stopped with exit code 100. The generated file now acquires the CLI in the
  SDK stage it already had and copies the tool into the runtime image, installing no .NET
  distribution at all. Nothing about this was visible from the generator's output — only from
  building it.
- **A container built from the generated files reported itself unhealthy while serving correctly.**
  The generated health probe used `wget --spider`, which issues HEAD, against an endpoint mapped for
  GET only — so every probe was answered 405 and the orchestrator saw a dead container, indefinitely.
  The probe now issues a GET, and `/health` additionally answers HEAD, so external probes and load
  balancers that use HEAD get the same answer as a browser.
- **`docker compose up` could not start the service at all.** The generated compose file selected the
  Production environment, where the API refuses to run on the default signing key, but supplied no
  key — leaving a container that crash-looped under its own restart policy. Compose now requires
  `JWT_SIGNING_KEY` and stops immediately with a message naming it, rather than starting something
  that cannot come up. A placeholder secret was rejected as the alternative: a known key in a
  generated file is worse than an absent one.
- The generated compose file no longer carries the obsolete `version:` key, which current Compose
  releases warn about.
- **`train` told the reader to promote a model it was about to promote itself.** The next steps are printed when the experiment is recorded, which is before automatic promotion runs, and the list always offered `mloop promote <exp>` — so read top to bottom the command asked for something the following lines reported as already done. The promotion step is now offered only when the command will not be taking it, which is what `--no-promote` asks for and where naming the experiment actually makes the line runnable.

- **`init` lined up its folder listing for one model name and no other.** The comment column was written into the strings, and two of the three paths carry the model's name — so `--name churn` moved every comment three characters out of line, and the default the padding was written for was one character out to begin with. The column is now computed from the widest path, which also keeps the object-detection and image-directory listings aligned.

- **The preview after a prediction showed the row that went in and not the answer that came out.** `predict` appends its answer columns to the row it was given, and the preview rendered every column of the result — so on a twenty-column dataset the table divided the terminal's width by twenty-four, elided every heading to an ellipsis, and ran each value down the page one character per line. The four columns that carried the prediction were the last four, and they were blank. The preview now keeps the appended columns unconditionally, fills the remaining width with the leading input columns so a row can still be told from the next one, and states how many columns it left out; the full row was always in the saved file and still is. A column counts as an answer when the input did not have it, so this stays correct as trainers change what they emit, and a file written without the original features loses nothing.

- **A binary model trained on `OK`/`NG` predicted correctly and then answered `1`/`0`.** ML.NET's
  binary classification takes a Boolean label, so a label written as words is converted before
  training — and nothing undid the conversion afterwards, so the class names the user wrote were
  absent from the answer. The two prediction paths did not even agree on what to put in their
  place: the CSV said `1`/`0` and the structured output (`--json`, `serve /predict`) said
  `True`/`False` for the same rows. Which value becomes the positive class is now one decision read
  from one place, by the conversion that makes it and by both paths that report it, so a prediction
  comes back in the vocabulary the model was trained on. The two orderings involved genuinely
  disagree — the conversion sorts case-insensitively while the schema's recorded value list sorts
  ordinally, which inverts a vocabulary like `a`/`B` — which is why reading the recorded list's
  first entry is not the same question and is not what happens. A model that recorded no vocabulary
  (an older experiment, or a label already Boolean in the data) still gets one answer from both
  paths, `True`/`False`. Binary probabilities are keyed by the same names, so a consumer can join a
  predicted label to its probability.

- **A multiclass prediction named its own classes in one field and numbered them in the next.**
  `predictedLabel` came back as `cat`, while the probability distribution beside it was keyed
  `class_0`/`class_1`/`class_2` — positions, whose meaning is the trainer's internal class order and
  appears nowhere in the answer. A consumer could not join the two without guessing, and the guess
  had no way to be checked. The scored model records which class each slot belongs to; that is now
  what names the keys, so the distribution reads `{"cat": …, "dog": …, "bird": …}` and every row's
  own predicted label is one of its own keys. Binary predictions were already keyed by class name,
  so the same consumer no longer meets a different rule per task type. Two annotations carry the
  vocabulary and both are read, because which one a model has depends on its label: a label written
  as words leaves per-slot names, while a label written as class ids (`0`/`1`/`2`) leaves none — only
  the training vocabulary — and reading just the first would have named the classes for one kind of
  label and silently kept positions for the other. A model carrying neither, or one whose recorded
  names do not cover its slots exactly and distinctly, keeps the positional keys it has always
  produced rather than being given invented names.

- **A schema that named a column's type the .NET way failed with a sentence about a different
  column.** A saved schema records each column's type in a small vocabulary of semantic names —
  `Numeric`, `Categorical`, `Text`, `Boolean` — and the prediction path builds a model's feature
  vector by selecting the numeric ones. A schema saying `Single` (the .NET type name) matched
  nothing, so no feature vector was built and the run died a step later on `Could not find feature
  column 'Features'`, which names neither the column at fault nor the value that was wrong. The
  failure now names them, lists the values that are recognized, and says the names are semantic
  rather than .NET types. One schema MLoop itself wrote had this defect — an object-detection
  model's bounding-box column — and it is corrected; models saved with it load exactly as before,
  since both spellings resolved to the same type on the way in.

- **`mloop predict` gave two different answers for the same input depending on `--json`.** Without
  the flag it failed (`Schema mismatch for feature column '__Features__': expected
  Vector<Single, 3>, got Vector<Single, 2>`); with it, the same file and model produced predictions
  and exited 0. The two take different paths — one infers a loader from the file, the other reads
  rows — and they disagreed about a file where a feature arrived as a different type than the one it
  was trained as. Column inference groups the columns it reads as numeric into one vector and leaves
  the rest beside it, so a numeric feature arriving as text drops out of the vector and the vector
  loses a slot; overriding the column's type, which the code already did, cannot move it back in.
  The reverse is the same defect: a feature trained as text arriving as digits is pulled into the
  vector, which then comes out one slot too wide. What the vector spans is a property of training,
  so it is now rebuilt from the saved schema rather than re-derived from the file being predicted,
  in both directions. Both paths now load the file and agree row for row. Files that already match
  are untouched.
  The same repair replaces a narrower one: label-less clustering had been putting its missing
  dimension back by re-concatenating a column in the predict path, for that one task. Rebuilding the
  range from the saved schema covers it — and covers every other task the same way — so the
  task-specific step is gone rather than left to double-count the column it restores.

### Added
- **Schema validation now reads values, not only the header.** It compared column names and never a
  field, so a column whose contents had changed type — a numeric feature arriving as `not-disclosed`
  — passed validation and the run failed later with `Schema mismatch for feature column
  '__Features__': expected Vector<Single, 3>, got Vector<Single, 2>`, a sentence that does not
  contain the name of the column that changed. The saved schema knew which column it was the whole
  time. `predict` and `evaluate` now say so first, naming the column, what it was trained as, how
  many of the sampled rows disagree and an example value. It is a warning, not a rejection: the
  loader coerces the values to the trained type and the run may still be what you want — the point
  is to know. Machine consumers receive it as a `warning` event like any other.

### Fixed
- **A path printed to the terminal was folded at the console width, inside the path.** `mloop
  promote` reported its backup location as `…/backups/exp-001-2` on one line and `0260908-064321`
  on the next; what a reader saw and copied was not a location that existed. Spectre lays every line
  out against the console width and breaks inside a token when the token has nowhere else to break —
  and the width it assumes whenever output is redirected is 80 columns, so the failure is worst
  exactly where output is captured and pasted from. Thirty-one call sites printed a path this way.
  A path (and any value a reader copies) is now written whole, at any width, while its label is
  still rendered; the value loses its colour, which is the trade: a value you have to retype is not
  decoration. Prose is untouched — a sentence reflows without harm.
- **Under `--auto-time`, a run that AutoML could not search was searched twice.** The probe phase
  exists to size the main phase; when it reports that AutoML could not run on the data at all and a
  direct pipeline stood in for it, the main phase has no more trials to buy — it takes the identical
  path, fails the identical way, and prints the identical fallback warnings a second time. Measured
  on a 50-row imbalanced binary set: four warnings for two passes, under a "Phase 2: Main training
  (130s)" line above a pass that spent ten seconds not searching. Main training is now skipped in
  that case, and the run says why instead of ending after Phase 1 with no explanation. The decision
  reads a typed signal on the trainer rather than the wording of its fallback message, so rephrasing
  a warning can no longer change what the engine does.
- **`serve`, `pipeline` and `update` exited 0 after reporting a failure.** All three had handlers
  that returned nothing, so no path through them could report anything but success — a pipeline file
  that did not exist, a pipeline run that ended `Failed`, an API assembly that could not be found, an
  API child process that crashed after starting, a self-update that did not install. Each printed a
  clear red line and then told its caller the run was clean. These are precisely the three commands
  written for a caller that reads nothing else: a scheduler, a supervisor, a provisioning script.
  All three now return an exit code, failures route to stderr rather than stdout, and the child
  API's own stderr stays stderr instead of being folded into the parent's stdout. A guard fails the
  build if a command handler is written that cannot report failure.

### Added
- **The exit-code contract is documented** (`docs/GUIDE.md` § Exit Codes). `0` means the command did
  what was asked — including outcomes that are not failures, like `evaluate` skipping a model with no
  production version. Non-zero means it failed and left a cause on stderr, and `serve`/`update` pass
  through the exit code of the process they launched. The rules were true in the code and written
  down nowhere, so a script had to discover them by experiment. Guards now assert both halves on real
  command lines: a failure exits non-zero with something on stderr, and under `--json` it still leaves
  a parseable document on stdout rather than nothing.

### Changed
- **The list of task types a project may declare is written once.** `init`'s scaffolding allowlist,
  `validate`'s config allowlist, and `train`'s `--task` help text each carried their own copy of the
  fifteen accepted values. They agreed, which is the state that makes duplication look harmless: the
  cost only appears at the next edit, and it appears asymmetrically — a task added to one and not the
  other scaffolds a project that fails its own validation, or is supported and named nowhere a user
  can find it. All three now read one list, and a guard fails the build if a fourth copy appears.
  Adding a task is one edit.
- **The changelog's comparison links stopped at `v0.6.1-alpha`.** Thirty releases were listed above
  a link block that described six, so `[Unreleased]` compared against a year-old tag and most
  headings resolved to nothing. Regenerated from the tags that actually exist; the release procedure
  now names adding the new version's link as a step, since a version's link cannot be written before
  its tag exists.

### Fixed
- **`mloop init --help` named five of the fifteen task types it accepts.** The option's description
  offered a handful as examples while the command validated against the full set, so the values a
  user could actually pass were discoverable only by guessing and reading the rejection message. Both
  `init` and `train` now list every accepted task, and the rejection message does too.
- **A deep-learning task with no runtime declaring it would have failed at native load rather than
  at the install prompt.** Which tasks need a native runtime is stated in two assemblies — the module
  that trains them and the registry that knows which runtime serves which task — with nothing
  connecting the two. They agree today; a guard now keeps them agreeing, in both directions.

### Fixed
- **`mloop evaluate --json` could not say that it had skipped.** Without a production model, evaluate
  reports that it is skipping and exits 0 — skipping is an outcome here, not a failure. The document
  it emitted, though, was every field null, which is also what an evaluation that ran and measured
  nothing looks like. The person watching the terminal was told which had happened and the consumer
  reading stdout was not. The document now carries `skipped`, naming the reason when no evaluation
  ran and `null` when one did. Human output is unchanged.

### Added
- **Published text is checked for tokens that only mean something inside the workspace it was
  written in.** This repository is public, and its changelog carried a hundred-odd work-item ids
  no reader could look up, links into a private notes tree, and absolute paths off one machine's
  disk. Two earlier passes by eye had each declared the tree clean and each missed most of it —
  the tokens look ordinary in every other context. A guard now reads every file git tracks and
  fails on an internal id, a drive-letter path in prose, a reference to the private working tree,
  or an internal cycle/sprint number; the tokens it found have been removed, and the descriptions
  that carried them now say what the class of defect was rather than pointing at a ticket. CI no
  longer skips documentation-only changes, which is what made the guard reachable at all.

- **`docs/RELEASING.md` — the release procedure, starting with a walk of the new-user path on the built binary.** `init → train → predict → train → list → promote → status` against a dataset from this repository, read as a user reads it: exit codes, whether training stayed inside the budget it printed, whether the promotion sentence is true for the situation, whether a path survives a copy from redirected output, whether stderr is empty on success. Doing this walk for the first time with the repository's own example dataset found the training hang fixed below — no test suite had. The rest of the document is what comes after the walk: suites run one project at a time, CI confirmed on the pushed commit before publishing, and how to trigger the release workflow when the version was already on `main`.
- **What a run trained is now carried in parts, not folded into one string.** `bestTrainer` was assembled by concatenation at sixteen task paths and had never been an identifier: it carries hyperparameters (`KMeans (k=3)`, `SsaForecasting (window=7, horizon=3)`, `MatrixFactorization (rank=32, iter=20)`), fallback notes (`SdcaLogisticRegression [manual fallback: AutoML AUC failure]`, `… [metric fallback: AreaUnderRocCurve→F1Score]`), or the whole assembled pipeline (`ReplaceMissingValues=>FeaturizeText=>Concatenate=>LightGbmRegression`) — a format never specified, differing per path, that a downstream consumer was parsing because it stamps the trainer into provenance. The parts exist at the moment the string is built, so they are what a run now carries: `trainer` (`name`, `params`, `fallbackReason`) appears beside `bestTrainer` in the `result` event, in `metadata.json`, and — where it matters most — on every row of `trials.ndjson` and `leaderboard.json`. A clustering leaderboard reads `KMeans (k=2)` … `KMeans (k=10)`, which as strings look like nine different trainers when they are one trainer at nine settings, and comparing a trainer across its settings is what a leaderboard is *for*. **Nothing a user or a stored experiment sees changes**: `bestTrainer` and `trainerName` are now derived from the parts by one composition, deliberately designed to reproduce every string the sixteen sites produced, with each of those seven shapes pinned character-for-character by test. Absent parts are omitted rather than blanked, matching the `phase` event's rule. The rendering itself is never serialized — writing it beside its own parts is how a display form and its parts drift apart.
- **`mloop list --trials <experiment-id>` shows the trial leaderboard a search produced.** Every AutoML search (and hand-rolled sweeps like clustering's K) has written `leaderboard.json` — trials ranked by the metric they were searched on — since 0.30.0, with no CLI surface to see it; a consumer had to read the file directly. Renders a ranked table (trainer, metric, runtime) and honors `--json` for the raw ranked data. Direction-aware: a `lower_is_better` metric like `davies_bouldin_index` ranks ascending, `higher_is_better` descending, matching the same `MetricDirection` authority the promotion gate uses — a clustering leaderboard now reads `KMeans (k=3)` before `KMeans (k=2)` when the former scores better, not just whichever ran first.
- **`mloop evaluate --json`.** `train` and `predict` have had `--json` since 0.30.0; `evaluate` — the third command that produces a result worth consuming programmatically — did not. Emits `{ model, experimentId, testDataFile, trainingMetrics, testMetrics, possibleOverfitting }` to stdout; all narration (the progress spinner, the schema-validation line, the human table) routes to stderr instead, the same reassignment `predict --json` already established for keeping stdout pure JSON.
- **`mloop validate --json`.** Emits `{ valid, errors: [{path, message}], warnings: [...] }` — the same findings the human report renders, collected once and either printed or serialized rather than computed twice. Exit code is unchanged (`1` on any error, `0` otherwise); a CI pipeline can now consume the structured reason instead of parsing the table.
- **`mloop status --json`.** Emits `{ projectRoot, models: [{modelName, totalExperiments, completedExperiments, hasProduction, productionExperimentId, bestMetric, lastPredictionAt}], dataFiles, summary }` — `dataFiles` is `null` unless `--verbose` is also passed, matching what the human report shows. `bestMetric` picks the same direction-aware best experiment the human table already displayed (a `davies_bouldin_index` clustering run isn't reported as "best" for scoring highest). All narration routes to stderr, same as `evaluate`/`validate --json`.
- **`mloop runtime list --json`.** Emits `{ runtimes: [{id, displayName, installed, sizeBytes, approximateSizeMB, requiredByTasks}] }` — `sizeBytes` is the actual on-disk size once installed, `null` otherwise (only `approximateSizeMB` is known before download).
- **`mloop prep run --json`.** Emits `{ model, dryRun, input, output, steps, errors, rowsBefore, rowsAfter }` — `steps` is the model's own `mloop.yaml` prep list serialized as-is (no separate view model to drift from it), `errors` covers every failure exit (unknown model, missing input file, step validation) with one shape, and `output`/`rowsBefore`/`rowsAfter` are omitted on a dry run since no file was written.
- **`mloop info --json`.** Emits `{ dataFile, labelColumn, labelSource, analyze, fileInfo, columns, labelDistribution, profile, analysis }` — `columns` reports the same per-column type/purpose classification the human table renders, as plain values (e.g. `"Numeric Feature"`); the human table's coloring is applied at render time from that same value rather than baked into it. `profile` is the always-on dataset profile, and `analysis` is the full `--analyze` deep-analysis result serialized as-is, present only when `--analyze` was passed.

### Fixed

- **A `--json` command could report success and hand back output that does not parse.** `mloop info --json` and every `mloop analyze … --json` sub-command began their stdout with `[Info] Removed index column(s): Unnamed: 0` whenever the data carried a column the loader drops, then printed the document after it and exited 0 — so a consumer's parse failed on a command that said it had succeeded. Two independent causes, both of which the earlier `--json` audit was structurally unable to see. First, `MLoop.Core` defaulted its diagnostic sink to `Console.WriteLine`: that is not the console a command's scope replaces, so the line went underneath the host's channel straight onto the stream reserved for the document. Core now writes nowhere without a sink — the host injects one and decides where these lines land — and the CLI's sink goes through its ambient console, so the scope a `--json` command installs governs Core's narration exactly as it governs the CLI's own. Second, `analyze` opened no output scope at all, for a flag all five of its sub-commands declare; the scope is now a property of running an analysis rather than something each handler remembers, so a sixth sub-command cannot ship without it. Human output is unchanged — the line still appears, on the stream a human reads. Three guards, each with its negative control verified: source-scanning (Core reaches for the process console again), fixture (a call site stops injecting the sink, and the message silently vanishes), and command-tree (a `--json` command loses its scope).
- **`mloop train` could run for minutes past its budget on a small dataset, and nothing could stop it.** On the repository's own `examples/customer-churn` data — 50 rows, twelve text columns — AutoML's internal AUC computation fails (the imbalance is such that a fold has no positive), the runner falls back to its manual SDCA pipeline, and that pipeline had no cache checkpoint between featurization and the trainer. ML.NET evaluates a pipeline lazily and SDCA makes many passes over its input, so every pass re-ran the whole text-featurization chain from the source rows: a 29-second probe sat in `SdcaTrainerBase.TrainCore` for five and a half minutes, then the main phase started the same fallback again. No time budget applies to the fallback, and the CLI passes no cancellation, so the only way out was to kill the process. The featurized rows are now cached once before the trainer, which is what ML.NET documents for every iterative trainer and is exactly the case the fallback exists for — small data that fits in memory. Measured on the same example: the whole `train` in 24 seconds, budget honoured, a calibrated model promoted. Pinned by counting how often training opens a cursor on the source data (37 with the checkpoint; the uncached fit did not finish in ten minutes), a bound that measures the code and not the machine.
- **A diagnostic that named a command to run folded that command across two lines.** `Error: No completed experiments found for model 'default'. Run 'mloop train --name default' first.` is longer than the 80 columns Spectre assumes when stderr is not a terminal, and it folds at the last space before the limit — inside the quoted command, so a user who copies the remedy from a log or a captured stream gets a broken one. Nothing written to the error console is a table or a panel that needs a width to lay out, so it now gets one no line reaches; a real terminal still wraps visually at its own edge, the text itself stays one line.
- **`mloop status` reported a `300s` time limit that neither the configuration nor training used.** The generated `mloop.yaml` leaves the budget unset so that training auto-estimates it from the data, and `train` says so (`auto (estimated from data size)`); `status` rendered the legacy default in its place. It now reads `auto`.
- **The text-featurization notice carried its icon twice** (`ℹ️  ℹ️  [Info] Text column(s): …`): the message added one and the logger added another. The logger owns the decoration.
- **One run reported its trial count three ways, and one of them disagreed.** In a run where AutoML failed and the manual SDCA fallback took over, `train --json` emitted one `trial` event and then summarised the window as `phase(complete)` with `trials: 0`, writing no `trials.ndjson` at all — so a consumer drawing trial nodes under a phase drew one node beneath a "0 trials" heading, and reported it. The cause was two counters for one fact, filled by different code: `complete.trials` read `AutoMLResult.Trials` while the events came from the progress channel. It was not limited to the fallback: of the **twelve** paths that report a trial, only three filled `Trials`, so every non-tabular task and all four reporting deep-learning handlers had the same disagreement. Worse, the leaderboard those paths *should* have kept was being discarded — clustering searches over K, ranking over trainers, time-series anomaly over detectors, and all three were classified as "single pipeline, no search" when the trial history landed. `trials` now means **the trials this run completed and reported**, with no per-task exemption: a reported trial always has a record, so the event count, the summary and the file count the same things by construction, and a zero-row leaderboard can no longer arise (a path that reports nothing records nothing — object detection and question answering measure no metric and so still write no files). Where the two productions cannot be merged — AutoML's `RunDetails` carries each trial's full metric set and its own runtime, which the live stream never sees — the rule that decides membership is single-sourced instead. Measured on the reported fixture: `trial n=1` with `complete.trials: 1` and a matching one-row `trials.ndjson`; and on a 240-row clustering run, nine trial events, `complete.trials: 9`, nine `trials.ndjson` rows, and a `leaderboard.json` ranked lower-is-better with `KMeans (k=3)` first on data built from three clusters — a nine-candidate leaderboard that used to be thrown away in full.
- **Warnings that said they were wired were not.** Three sources reached the terminal and never the event stream, so a `--json` consumer heard less than a human watching the same run. `AutoMLRunner` built its own console logger, which made it structurally impossible for any of its six warnings to reach the CLI's warning seam — including the entire AUC-fallback chain, so a run that fell back to the manual pipeline never said so as an event. `TrainCommandLogger.Warning` rendered straight to the ambient console, taking with it `LabelValueHandler`'s `Dropped N/M rows with missing labels` — a source listed as wired, and the one a downstream consumer specifically wanted for provenance. Two doc comments meanwhile asserted wiring that did not exist ("the one place Core warnings surface in the CLI", on a logger the runner never received; "Error/Warning go to stderr", on a warning that went to stdout). The runner now takes an injectable warning channel — only the warning channel, since info and debug are narration that machine mode discards by design and so have nowhere to arrive — and the command logger reports through the seam. Also new: **which columns featurization dropped, and why**, as a warning event (`Excluded from features (Constant): note`), raised from the structured exclusion decision rather than scraped out of the removal chain's prose, so a consumer no longer has to re-derive the post-exclusion schema to know what was trained on. And a classification run whose accuracy does not beat always predicting the most common class now says so even when the promotion gate does not stop it — the gate's `max(1/N, majority share)` floor only applies when accuracy is the metric being judged, so a run optimizing `f1_score` could promote with a meaningless accuracy figure. Measured on one 200-row run with three missing labels, two positives and a constant column: seven warning events, exit 0, stderr empty.
- **A summary line's value moved to the next line and read as missing.** `> Best Trainer: ` plus an assembled pipeline name is 84 characters against the 80 columns Spectre assumes when stdout is redirected, and the name is one unbreakable token, so the only wrap point was after the label — leaving it alone on its line. A downstream consumer reported it as a missing value; it never was, the name was on the line below. The summary now names the trainer that won (`LightGbmRegression`) rather than the pipeline that produced it, which is what a summary line should say; the assembled name is still carried in full by `--json`, `metadata.json` and `mloop list`.
- **The promotion gate measured classifiers against random guessing, which on imbalanced data is no opponent at all.** The floor for `accuracy` was `1/N` — the score of a coin flip — so a binary label that is 99.2% one class still only had to clear **0.5**. A model scoring 0.986 promoted cleanly despite being *worse than answering with the most common class every time*, and `IsClassificationDegenerateModel` did not catch it either: that check only fires on total collapse (`f1 ≈ 0`, or `negative_recall ≈ 0`), so a model that gets the minority right occasionally while still losing to a constant passed between the two checks. Reported downstream after it recurred across a real catalogue, where the consuming app had begun computing the baseline in its own display layer to compensate. The gate now compares against the stronger of the two trivial models, `max(1/N, majority-class share)`. That share was not recorded anywhere, so it is now measured during schema capture — the pass that already streams the whole file to collect categorical values counts occurrences instead of collecting a set, making it free — and stored as `MajorityClassRatio` on the label column. An experiment trained before this, or one with a numeric label, carries null and keeps the old `1/N` floor: null means "not measured", never zero. The change is deliberately confined to `accuracy`/`micro_accuracy`, because those are the two metrics a majority-class predictor actually scores; the **macro** metrics keep `1/N` and that is correct for them — always predicting one class yields recall 1 on that class and 0 on the others, so its macro accuracy is exactly `1/N` no matter how skewed the data is, and raising that floor would reject models that genuinely beat the trivial one. Blocking also explains itself now: the old advice was a flat "Model performance is near-random", which is the wrong diagnosis for a model far above chance that simply lost to a constant, and it was written to stdout where a `--json` consumer never saw it. Measured on a 500-row fixture with 4 minority rows: `majorityClassRatio: 0.992` in the schema, accuracy 0.990, and a warning event reading "below minimum threshold (0.9920) … always predicting the most common class scores 99.2% here" — with stderr still empty and exit 0.
- **A failure's guidance never reached the consumers that needed it most.** `ErrorConsole` documents that a tip is "part of the cause a machine consumer persists" — and then `Tip` wrote to stderr and returned, while only `Error` reported into the event stream. So every `--json` consumer was told *what* broke and never *what to do about it*, which is the half that resolves the failure. Sixteen call sites were affected, and eleven more had the guidance on **stdout** entirely (`AnsiConsole.MarkupLine("Run mloop init…")` after an `ErrorConsole.Error`), which also broke the other direction: a command could exit non-zero with its remedy sitting in the stream reserved for results. Fixed at the sink rather than call site by call site: `ErrorConsole.Error(cause, tip)` writes both lines to stderr and reports them as a **single** event, keeping "one failure ⇒ one error event" intact. The standalone `Tip` remains for guidance that stands on its own, now documented as terminal-only so the trap is visible where it is set. Every adjacent `Error`+`Tip` pair moved to the new shape; no call site writes a bare tip any more. Measured, running `mloop logs` outside a project: exit 1, **stdout empty**, stderr carrying both the cause and the remedy — previously the remedy was the one line on stdout.
- **Eleven commands each carried their own copy of "you are not in a project", and the copies had already drifted.** `ProjectDiscovery` throws with "Run 'mloop init' to create a new project, **or navigate to an existing project directory**"; the eleven commands that catch that exception printed only the first half, so the answer a user got depended on which command they happened to run. The wording is now stated once (`ProjectDiscovery.NotInsideProjectCause` / `NotInsideProjectGuidance`) and consumed by all twelve sites.
- **Smaller message defects of the same family.** `mloop runtime remove` rejected an unknown runtime without saying which ids exist, while `runtime install` — the sibling in the same file — listed them; both now share one rejection. The JWT key-length errors in `MLoop.API` and `mloop token` stated the constraint without naming the setting to change, unlike their neighbours; `mloop token` additionally rendered that error through `AnsiConsole`, putting a diagnostic on the stdout that `mloop token -q` reserves for the token itself and leaving stderr empty on a non-zero exit. `Model 'X' already exists.` now names both ways forward, matching the sibling rejection directly above it. And the label-not-found error folded its "Available columns:" line — the part that makes the failure fixable, and which the thrown exception has always carried — into the cause instead of an aside on stdout.
- **`mloop detect` promised control limits and shipped a band that cannot be one.** The command emitted SR-CNN's `UpperBound`/`LowerBound` under an "SPC bounds" heading, and on a 300-point sensor series **288 of 300 normal points plotted outside their own band** — charting normal operation as a wall of violations. The band was not miscalibrated; it was measuring the wrong quantity. Upstream builds it as `expected ± unit * factor(sensitivity)`, where `CalculateBoundaryUnit` derives `unit` from the series' level and trend, not its spread. Measured on one fixture and two variants of it: multiplying the noise by five left the median band width at 0.0151 → 0.0155 (**unchanged**), while shifting the level by +1000 with the spread held fixed widened it 0.0151 → 0.6151 (**~41x**). No `--sensitivity` value fixes that — the sweep only rescales a band blind to dispersion, and the settings where it *looks* right on a given file (80–90 here) are an accident of that file's level and spread happening to share a magnitude. The band also never decided anything: upstream uses it only to *clear* an anomaly flag the spectral-residual score already raised, so `IsAnomaly ⇒ outside` holds by construction and the converse does not — which is why 96% outside was never a detection defect. Both facts are now stated by the output rather than hidden by it. Each point carries two bands: `MarginLower/MarginUpper`, the detection gate that explains the verdict, and `ControlLower/ControlUpper`, real Shewhart limits at `expected ± 3σ̂`, with `σ̂` the robust dispersion (MAD × 1.4826) of the residuals with the flagged points excluded so a spike cannot inflate the band meant to exclude it. `ResidualSigma` is emitted alongside. On the same three fixtures the control band now scales exactly with the spread (σ̂ 0.2188 → 1.0938 at 5x noise), is invariant to the level, and holds **296 of 296 normal points inside** while placing exactly the four injected spikes outside. Control limits are computed in MLoop because ML.NET exposes no such surface to defer to. **Breaking for consumers of `--json` and `--output`**: the `upperBound`/`lowerBound` fields are replaced by the two named pairs above.
- **`mloop validate` crashed while listing the configuration problems it exists to report.** A finding's path and message are data, and both routinely contain square brackets — an indexed path reads `models.default.prep[0]`, and a note carries a literal `[info] ` prefix — but both were interpolated straight into a Spectre markup string, which parses brackets as style tags. The result was `Unbalanced markup stack. Did you forget to close a tag?` in place of the error list, in **human mode as well as `--json`**, on the most ordinary failure the command has: a config with an unknown prep step. `prep run` had the identical defect in its step-validation report. Both now escape the data at the display boundary. This also restored the `--json` payload on those paths: the crash happened before the emitter was reached, so `validate --json` and `prep run --json` had been exiting non-zero with an empty stdout.
- **A `--json` command could exit without ever emitting a document, and the exits where it happened were not reachable by fixing them one at a time.** `predict --json` had twelve error exits — "no production model", an unreadable data file, a schema it cannot use — each of which reported to stderr and returned, leaving stdout empty; `JSON.parse("")` fails exactly as prose does. The "you are not inside a project" exit is worse still: it belongs to shared resolution code, so no command owns the place a guard would go. Both are now covered by a scope a structured-output command opens once, which also takes over routing narration to stderr (a block that had been copied verbatim into thirteen places). The scope is a *guarantee*, not an interception: it asks at the end whether stdout received anything, and only then emits the failure that was reported. That ordering matters — a command like `info` reports the cause first and emits its own richer document second, and an intercepting sink would have written the poorer one first and left two documents on stdout. Deliberately narrow: an exit that reports no cause at all is left alone rather than given a fabricated one. `evaluate --json` separately gained the payload it never emitted when there was no production model to evaluate, with the same keys as the evaluated case and nothing measured.
- **`compare` accepted `--json`, described it as machine-readable, and then rendered the same table anyway.** The local-state path took the flag as a parameter and never read it, so `compare exp-001 exp-002 --json` returned a box-drawing comparison table on stdout — unparseable from the first byte — while the `--metrics-file` path had honored `--json` all along. It now emits the ranking decision (`metric`, `direction`, `directionSource`, `best`, `tie`, `ranking`, `excluded`) under **the same keys, computed by the same function**, as the `--metrics-file` output, so a consumer reads either without branching on which mode produced it; local state adds enrichment (`experiments[]` with task, status, timestamp, config, every metric, and the production pointer) rather than a second vocabulary for the same decision. An experiment ID that does not resolve is reported as an `excluded` entry with `reason: "not-found"` instead of a warning line on stdout, and the two states the human path stops at — nothing to compare, only one experiment — are reported as data (empty ranking with a null winner; a ranking of one) rather than prose, with exit codes unchanged. Human output is byte-identical.
- **Three `--json` commands exited on failure with nothing on stdout at all.** An error return is still an exit point, and `JSON.parse("")` fails exactly as `JSON.parse("Error: ...")` does — so "non-zero exit plus a sentence on stderr" is not a machine contract. `list --trials` (unknown experiment, and an experiment whose task ran no search so has no leaderboard) and `compare` (no model name given, and its top-level handler) now emit `{"error": "..."}` alongside the unchanged stderr diagnostic. The payload shape existed in three commands as a hand-written one-liner apiece; it is now named once so it cannot drift per command, and the error text and its tip travel together — a tip that resolves the failure is the half that a consumer receiving only the cause is missing.
- **Four commands answered `--json` with prose instead of a payload whenever there was nothing to report, and a fifth put its progress line on stdout ahead of the JSON.** `logs`, `feedback list`, `feedback metrics`, and `sample stats` each took an early return on the empty case — no logs, no feedback, no predictions yet — that printed a human hint (`No prediction logs found.` …) and never reached the `--json` branch at all, so a consumer parsing stdout got a parse error on the most ordinary state a fresh project has. They now emit the same payload shape as any other run, with the counts at zero and the collections empty; the human hint still prints when `--json` is not passed. `trigger check --json` separately wrote `Evaluating triggers for …` to stdout before its JSON, which made the document unparseable from the first byte — that narration now goes to stderr, the same routing the other `--json` commands use. Found by exercising each command's empty state through the real command tree, which is now covered by tests.
- **`mloop info` on an empty file reported the failure and then exited 0.** Every other input-rejection path in `info` (missing file, missing DataLens dependency for `--analyze`) reports the problem and exits `1`; the empty-file check printed "File is empty" and returned success anyway, so a script checking the exit code alone saw no failure.
- **`prep run` crashed instead of showing the pipeline whenever a step's column list (or the "no prep steps" example text) reached the terminal.** The steps table and one hint line embedded raw text like `columns: [pH, Temp]` in a Spectre markup string; unescaped square brackets are parsed as a style tag, so the CLI failed with `Could not find color or style 'pH, Temp'` instead of doing what `prep run --dry-run` exists to do — preview the pipeline. Reproducible on the simplest cases: any prep step with a `columns:` list (even a single column), and running `prep run` on a model with no `prep` section at all (the guidance text hit the same bug). Never caught before because the only existing coverage unit-tested the string-building helper directly, without ever rendering it through Spectre.
- **A missing file was named without saying where the CLI looked for it.** `analyze`, `info`, `prep run`, `predict`, `evaluate`, and `train --data` each reported "File/Data not found: {path}" with the resolved absolute path and nothing else — leaving a relative-path typo to guess whether that path was relative to the project, the shell's current directory, or something else. Every one of these resolves an explicit relative path by joining it against the project root before checking existence (`sample create --from` is the sole exception, resolving against the current directory instead) — now stated as a `Tip:` line alongside the error, so a user who passed a wrong-but-plausible relative path is told which base was actually used rather than left to reason about it from the absolute path alone.
- **Seven call sites across four commands had guidance that reached the terminal but not the machine channel.** `compare`, `sample create` (three call sites), `new hook` (two call sites), and `feedback add` each followed `ErrorConsole.Error(...)` with a plain `AnsiConsole.MarkupLine(...)` carrying the actual remedy — the same stdout-only leak `ErrorConsole.Error(cause, tip)` was built to close in 0.30.0, just reached through a call the earlier audit didn't search for (a raw markup line rather than a standalone `ErrorConsole.Tip`). A `--json` consumer saw the cause and never the fix. All seven now report through the combined overload, so the remedy is part of the same error event. `train --data` merging multiple files also gained two guidance lines it never had: a schema mismatch between files now lists which columns differ per file (the data already existed on the validation result and simply wasn't shown), and a merge or label-cleaning failure now says what to check next instead of stopping at the raw error.
- **`init`'s model-name rejection bypassed the stderr contract entirely, and named a rule the name could already be failing for a different reason.** Invalid task type used `ErrorConsole.Error` two lines above it; invalid model name instead wrote straight to `AnsiConsole` — stdout, not stderr, and never reported to the `--json` event stream, so a machine consumer saw no failure at all rather than an incomplete one. The message also named only the length/character rule, so a name rejected as reserved (`staging`, `production`, `temp`, `cache`, `index`, `registry`) was told a rule it already satisfied. Both fixed together: routed through `ErrorConsole.Error`, and the tip now names all three rules. `init`'s invalid-characters rejection also gained the specific characters found, rather than stating only that some existed.
- **Every command's shared top-level error handler dropped its own Suggestions block off the machine channel, and the dedicated training-failure path reported nothing to it at all.** `ErrorSuggestions.DisplayError` — what every command's outermost `catch` funnels through — called `MachineOutputScope.ReportError` with only `ex.Message`, before the Suggestions block a few lines below was even computed; a `--json` consumer got the cause and none of what resolves it, at the scale of the shared handler rather than one call site. `DisplayTrainingError` had no such call anywhere in it: a `train --json` failure reaching this path reported *no* error event at all. It also turned out to be unreachable — its own 0.28.0 changelog entry describes it as already wired into every top-level catch, but `train`'s actual `catch (Exception ex)` called the generic `DisplayError` instead, so the richer training diagnostic (model name in the heading, `mloop analyze`/`mloop status` hints) was built, tested, and never used. Both functions now report cause, inner exception, and suggestions as one combined event, and `train`'s top-level catch calls `DisplayTrainingError`, restoring the behavior its changelog entry already claimed.
- **The promotion gate's quality floor let a model that scored exactly at the floor through as if it had cleared it.** `R² > 0`, `AUC > 0.5`, and every other quality-gate threshold mean "must be better than a named trivial baseline" (mean-prediction, random guessing, the majority-class predictor) — but the comparison was a strict `<`, so a model scoring *exactly* the floor (`r_squared: 0.0`, `auc: 0.5`, …) sat exactly on the trivial baseline and was not blocked. Reported as `R² == 0.0` on a near-empty or degenerate test split — where the model has, by construction, learned nothing more than the trivial baseline it should have to beat. Changed to `<=`, uniformly across every metric this floor applies to (accuracy, F1, macro variants, R², AUC): equal to the floor is not better than the floor.
- **`docs/GUIDE.md` had seven examples the CLI would reject outright** (an earlier pass fixed 6 sites in README and 12 in this file in 0.30.0 — these seven were either missed then or drifted back since): `evaluate`/`predict` examples passed a `model.zip` path as a bare positional argument where those commands take an experiment ID or `--model <path>`; `promote --to staging` and `validate --type hooks`/`--type metrics` documented options neither command has ever had; one `train` example repeated the original defect itself, a positional label the CLI has never accepted. Each fix is verified against the CLI's real argument grammar, not just re-read. New: a test (`ReadmeCommandContractTests`) parses every concrete example in README.md and `docs/GUIDE.md` against that same grammar (`Program.BuildRootCommand`, the tree `Main` actually runs) on every build — the mechanical check the original issue's resolution asked for instead of relying on the next human reader to notice.

- **A command line that failed to parse wrote its usage text to stdout, and a caller asking for `--json` got that prose instead of a document.** An unrecognized option or command never enters a command's action, so none of the machinery that keeps stdout machine-readable is ever constructed — this was the last exit where `--json` could not be relied on. Usage and the parser's own messages now go to stderr, where diagnostics belong, leaving stdout carrying only what was asked for; when `--json` was requested, stdout receives `{"error": "..."}` naming every cause the parser found, not just the first. Explicitly requested help (`--help`, `--version`) is unaffected and stays on stdout: output asked for by name is the answer, not a diagnostic. Running `mloop` with no arguments is a parse failure, so its usage block moved to stderr as well — the banner and the exit code are unchanged.

- **A mistyped option was absorbed as a positional value, and the command then reported confidently about the wrong thing.** `promote --lastest` answered "Experiment '--lastest' not found" and `predict --jsn` answered "No production model found" — both non-zero, both naming a subject that was not the problem, sending the reader to look at their models over a typo in front of them. Any command with a positional argument accepted an unrecognized `--` token as that argument's value. The token stream is now checked once, after parsing, rather than by constraining fifteen arguments individually: an unrecognized `--` token bound as a value is reported as what it is, with the command's own `--help` as the next step. A single dash is untouched — it remains a value by convention, and one option already reads it as standard input — and everything after a bare `--` is a value by definition, which is the escape hatch for a value that really does begin with two dashes.

- **Schema validation claimed to check something it could not reach, and the check it would have run was wrong.** When an experiment records no input schema, validation fell back to comparing the prediction data against the model artifact's own schema. That path was unreachable — training records the schema unconditionally, and models are saved without one, so the fallback always returned early — and had it been reached it would have reported nonsense: the artifact's schema describes a featurized view where every feature column is folded into a single vector, so present columns read as missing, and it demands a label column that prediction data is not supposed to carry. Removed, leaving the recorded schema as the single authority for what a model was trained on; an experiment without one now says validation was skipped and why, instead of implying it passed. `SchemaValidator.ValidateAsync` no longer takes a model path, since it never read it.

- **The exit-code contract was never written down.** Every command returns `0` or `1`, and one case is easy to read as a bug without the reason: `evaluate` with no production model reports the skip, emits its usual document with nothing measured, and exits `0`, because having nothing to evaluate is an outcome rather than a failure — a caller tells the two apart by `experimentId` being `null`. README now states this alongside the stdout/stderr split and the `--` escape hatch, so a script author has the whole contract in one place. The doc-example check that keeps those snippets honest was widened to allow them: it now understands shell redirection, and lets an example document a failure instead of only ever showing the happy path.

- **Eight commands were absent from every user-facing document.** `status`, `logs`, `feedback`, `sample`, `trigger`, `token`, `preprocess` and `new` all shipped, several of them with the machine-readable output this release advertises, and a reader had no way to learn they exist. README now carries the full command list, pinned to the command tree by a test — a hand-kept list is what drifted in the first place.

- **Every number MLoop printed changed shape with the machine's locale.** The messages are English prose, but the figures inside them — an accuracy, a majority share, a row percentage, a duration — were formatted in whatever culture the host started in. On a machine set to a comma-decimal locale the same run reads `Accuracy 0,9750 does not beat always predicting the most common class (97,5 % of rows)`: English sentence, another locale's separators, and a percentage that grew a space. The same figures travel inside `--json` message fields, so a consumer parsing them got a different string depending on where the CLI happened to run — and a comparison of two runs on two machines showed differences that were not there. The formatting culture is now pinned once, at the entry point of both the CLI and the API server, to a single declared authority rather than to the host: the invariant culture, with percentages rendered the way English prose writes them (`97.5%`, not invariant's `97.5 %`) so that what users have been reading is exactly preserved. Output on an en-US or ko-KR machine is byte-identical to before; a comma-decimal machine now agrees with it instead of diverging. The test suites pin the same culture, so they assert what a user actually gets — previously they asserted the developer's locale, which is why a suite could be green on every developer machine and red on a runner.
- **`mloop serve` told the wrong thing to anyone not on Windows, and named a solution file that does not exist.** When the API assembly cannot be found, the guidance offered a `set VAR=...` command with a Windows drive-letter path — in a tool that runs on Linux and macOS — and then suggested building a solution file under a path the repository has not had. Both replaced with platform-neutral guidance and the solution file that is actually there.

- **A project's compiled scripts were cached wherever the shell happened to be, not in the project.** `.mloop/scripts/` is compiled once and cached as DLLs so later runs skip Roslyn — but the cache path was left relative, and the CLI never changes its working directory: it resolves the project root and passes it around. So `mloop train` run from a subdirectory found the scripts correctly (project root) and wrote their compiled output to `<that subdirectory>/.mloop/.cache/scripts` — a stray directory left in the user's tree, and a cache that missed on the next run from anywhere else, recompiling every script each time. Every one of the four call sites already held the project root and dropped it on the way to the loader. The path is now stated once, as `ScriptLoader.CacheDirectoryFor(projectRoot)`, and all four pass the root they have; a caller with no project root still gets the previous behaviour, now spelled out as the current directory rather than implied by a relative path.

- **The two environment variables MLoop's own generated Dockerfile sets were read by nothing.** `mloop docker` writes `ENV MLOOP_PROJECT_ROOT=/app` and `ENV MLOOP_MODEL_NAME=<model>` into the image, its compose file repeats the second, and `mloop serve` sets the first before starting the API — and no code anywhere read either one. The images worked by coincidence: the project root because the generated `WORKDIR` happened to be where the project was copied, so walking up from the working directory landed on it anyway, and the model because it happened to be named `default`. An image built for a differently-named model served `default` regardless, and moving the entry point or the working directory would have broken the first silently. Both are now read where the answer is decided, once each: an explicitly configured project root takes precedence over walking up from the working directory — which is what a server or a container has instead of a meaningful current directory — and a value naming something that is not a project fails saying so, and naming the variable, rather than falling back to whatever directory the process started in. It applies to every `mloop` command, not only the server: exporting it in a shell pins that shell's commands to the named project. `MLOOP_MODEL_NAME` supplies the model when a command does not name one; an explicit name still wins.
- **The API server started fine without a project and failed on the first request that needed one.** Six service registrations and three endpoints each resolved the project root for themselves, and five of those wrote `FindRoot() ?? Directory.GetCurrentDirectory()` — a fallback that could never run, since `FindRoot` throws rather than returning null. So there was no fallback: outside a project the server started cleanly and then answered a 500, from whichever endpoint first touched a service that needed the root, with the cause buried in a stack trace. The root is one fact for the life of the process, so what is registered is now the answer rather than the question: it is resolved once, before the first request, and logged. A deployment that cannot find its project says so while an operator is still watching the log, in the same wording the CLI uses — and nine repeated walks up the filesystem go with it.

- **The reason given for a promotion described a comparison that had not happened.** A project's very first training has no production model to beat, and the line printed under `Model promoted to production!` said `Better accuracy than current production model` regardless — among the first things a new user is shown, one command into their first project, and untrue for exactly that run. The presenter already received the production model, or its absence, and did not look at it. A first promotion now says so; a replacement names the experiment it displaced and shows both figures (`Better accuracy than exp-004: 0.7100 -> 0.8500`), so the claim can be checked rather than taken — the same rule the quality-gate messages already follow. Where the displaced model recorded no value for the metric, the experiment is still named and no numbers are invented.

- **The path to the model just trained was folded in half.** `Model saved to:` printed the absolute path, and an absolute path plus that label is longer than the 80 columns Spectre assumes when stdout is redirected — which is what a script, a CI log, or a piped terminal gets. There is no space to break at inside a path, so the fold landed mid-token and put `model.zip` on a line of its own: a user copying the value out gets a path that does not exist, with nothing to indicate the line continued. It now shows the path from the models directory down (`models/default/staging/exp-001/model.zip`) — the part that identifies the artifact, in the same vocabulary every command that takes an experiment already uses, and short enough to fit. A path that does not run through the project's layout is shown whole rather than guessed at, and the value is escaped, so a project living under a directory with a bracket in its name no longer breaks the summary the way `validate` and `prep run` used to break their own reports.

### Security

- **A central version pin is no longer mistaken for a way to ship a security floor.** Three separate advisories have now been answered by raising a `PackageVersion` in `Directory.Packages.props`, and twice that fixed only this repository's own restore: central package management governs restore here and is not part of the dependency graph a project hands to whoever references it, so a project-referencing consumer silently re-inherited the version the pin excluded. The rule was already written down for the one package it was first learned on, which is why the next package that needed it did not inherit the lesson. It is now enforced: every entry marked with a `Security:` comment must also be a direct dependency of a library under `src/`, or else be removed in favour of advancing the dependency whose own manifest carries the floor. A pin written for build coherence rather than for an advisory carries no such marker and is unaffected. `docs/EMBEDDING.md` states the mechanism in general terms rather than as a fact about one package.

- **The explicit `System.Security.Cryptography.Xml` override is gone — the floor it enforced is now carried by the dependency that needed it.** 0.30.0 shipped an interim `Directory.Packages.props` pin at 10.0.10 to clear four High advisories reaching MLoop transitively through `FilePrepper` → `EPPlus` (GHSA-cvvh-rhrc-wg4q, GHSA-g8r8-53c2-pm3f, GHSA-23rf-6693-g89p, GHSA-8q5v-6pqq-x66h against 10.0.7), because FilePrepper's own manifest at the time didn't carry the fix. FilePrepper 0.7.4 now declares that dependency directly at 10.0.10+, so the pin duplicated a floor the dependency already guarantees. Bumped `FilePrepper` 0.7.3→0.7.4 and `DataLens` 0.13.3→0.13.5 (both already published, MLoop had not caught up), removed the now-redundant override, and confirmed the resolved graph still lands `System.Security.Cryptography.Xml` at 10.0.10 with `dotnet list package --vulnerable --include-transitive` reporting zero vulnerable packages across all 15 projects in the solution.

## [0.30.0] - 2026-07-21

### Added
- **`mloop train --json` streams the run as newline-delimited JSON.** Every other long-running command already had `--json` (`analyze`, `compare`, `predict`, `list`, `logs`, `promote`); the one that takes minutes and most needs watching did not, so a consumer running mloop as a subprocess could show nothing between "started" and "finished". Measured on a 20-second run captured to a file: the rich output gave **one** line for the whole training (`Training model…: 0%`), while `--json` gave **nine trial events plus a result event**, each flushed as it happened. Events are `phase` (training-window boundaries), `trial`, `warning`, `result` and `error`. Phases share one vocabulary across budgeting modes, so a consumer never branches on how the run was scheduled: auto-time emits `probe` → `main` (or `converged`), a fixed `--time N` run emits `main` directly with its budget, and every successful run closes the stream of phases with `complete` (trial count as persisted in `trials.ndjson`, elapsed time) before the post-training steps — save, evaluate, promote — run. Fixed-budget runs originally emitted no phase at all, which a downstream consumer measured as "62 seconds of complete silence" in their tree UI; now both of their execution modes produce the same node structure. A phase event carries only the facts that boundary has — a fixed-budget run has no `probeTimeSec`, nothing has a `metric` before training starts — and omits the rest rather than zero-filling, so an absent value is never confusable with a measured zero. Warnings initially had no common seam the way failures have the stderr sink — they were raised from scattered render and log calls across three assemblies, all discarded in `--json` mode, so a machine consumer saw *fewer* warnings than a human at the terminal. One seam now exists (`WarningConsole`, hooked by the same scope that owns the stream), and the known sources report through it: data-quality findings (class imbalance, starved classes), the missing-label drop notice, the promotion quality gate blocking a model, a post-train hook abort, and Core's logger warnings including the undefined-metric sanitizer. Measured on an imbalanced fixture: the `--json` stream opens with `warning` events carrying the same imbalance findings the terminal shows, and a success still leaves stderr empty. Two channel contracts are kept by construction rather than by convention: stdout carries nothing but the event stream — a single scope silences Spectre rendering and plain narration for the duration, since `train` renders in some sixty places and narrates from three assemblies — and the 0.28.0 rule still holds, so a non-zero exit still leaves its cause on stderr *and* now reports it as an `error` event. Warnings carry no error code: MLoop has no such vocabulary, and inventing one per call site would give consumers a code they cannot rely on.
- **An experiment now keeps the leaderboard its search produced.** ML.NET hands back every trial's trainer, validation metrics and runtime on `ExperimentResult.RunDetails`; MLoop read `BestRun` off it and dropped the rest, so each run destroyed the only record of what else was tried and how it scored — the best trainer and its final metrics were the maximum resolution available afterwards. Searching tasks (binary classification, multiclass classification, regression) now write two files into `models/{model}/staging/{expId}/`: `trials.ndjson`, one JSON object per completed trial in completion order, and `leaderboard.json`, the same trials ranked by the metric the search actually optimized. Ranking direction comes from the `MetricDirection` authority rather than a local test — the same knowledge that, duplicated, once ranked the worst clustering model first — and a metric that authority does not recognize leaves the trials unranked and says so instead of guessing a direction. Tasks that fit a single explicit pipeline write neither file: an empty leaderboard would claim a search happened and found nothing.

### Fixed
- **Training progress on the three most common tasks was reported to nobody.** `AutoMLRunner`'s binary-classification, multiclass-classification and regression paths each accepted an `IProgress<TrainingProgress>`, threaded it down through every layer — and never called `Report` on it, while the six non-tabular tasks did. The user-visible result was a progress bar frozen at 0% for the whole run, an auto-time probe summary that always read `0 trials`, and a static `Training <model>…` line: three symptoms, one cause. Every completed trial is now forwarded (`TrialProgressReporter`), with the metric read from the ML.NET enum the experiment actually ranks by rather than from the requested metric string — binary classification switches to `F1Score` mid-run when AUC turns out to be undefined, so the two can disagree.
- **The unused-data warning named the file the run had just trained on.** Training with `--data data.csv` could end with `Standard data file(s) not used: data.csv`. The scan was right — the unused file was a *second copy* under `datasets/`, genuinely untouched — but every display path printed the bare file name, so two distinct files collapsed into one on screen and the message read as a contradiction. Unused files whose name is shared with a file the run used are now shown as a path (`datasets/data.csv`), and the warning says outright that a different file with that name was used. Names that are unambiguous stay bare: paths everywhere would be noise, and the collision is what justifies them. Fixed in the scanner rather than the renderer, so the warning text and the file list agree and any other consumer of the scan sees the same names.
- **A trial line could report a metric no trial had measured.** The seven task paths that fit one explicit pipeline instead of running AutoML — anomaly detection, clustering, ranking, forecasting, time-series anomaly, recommendation, the manual SDCA fallback — plus the six deep-learning handlers, each built their progress update by hand with `ElapsedSeconds = 0`, and most of them sent it *before* fitting with `Metric = 0`. That renders as `Trial 1: RandomizedPca - detection_rate=0.0000` on a bar that cannot move, because the CLI computes its percentage from `ElapsedSeconds`. All of them now report through a single `TrialProgressChannel` that owns the trial ordinal and the elapsed clock, once the trial has a result to report. A pipeline that produces no metric (object detection, question answering) reports nothing rather than reporting a zero.

## [0.29.2] - 2026-07-20

### Fixed
- **A run that completes no trial now explains itself, instead of being advised to shorten the time budget.** When AutoML ends without a single successful trial it throws ML.NET's own `TimeoutException` ("Training time finished without completing a successful trial"), wrapped in an `AggregateException` by its internal threading. That message contains neither "timeout" nor "memory", so every suggestion rule missed it and the generic training fallback answered with `Try with a smaller time limit first: mloop train --time 30` — advice that makes the very failure it is answering worse. The wrapper is now unwrapped where AutoML failures are already interpreted (`AutoMLRunner`, alongside the AUC-undefined recovery) and the condition is recognised **by exception type**, not by matching an ML.NET English string that a localization or wording change would break. The diagnosis deliberately names **two** causes — a budget too small, or trials that all died before finishing — because in this path MLoop genuinely cannot distinguish them: when no trial completes, AutoML's `progressHandler` reports nothing at all (measured against Microsoft.ML.AutoML 0.23.0). Naming only one would be a guess presented as a diagnosis. The suggestions cover both directions: raise `--time`, or reduce the work per trial with `--max-rows` / `mloop sample create` / fewer columns.
- **Out-of-memory failures reached the user with no memory advice at all.** The suggestion layer had a memory rule since early on, but it tested `ex is OutOfMemoryException` against the *wrapper*: AutoML wraps the failure in an `AggregateException` and the CLI wraps again as `Training failed for experiment {id}: …`, so the rule never fired on a real training run. Exception-type rules now walk the inner chain. Native trainers that report exhaustion in their own words (`bad allocation`, `bad_alloc` — LightGBM) are recognised too, for the cases where that text reaches a managed message.

### Known limitation
- A **native** out-of-memory crash (the trainer's own allocator aborting the process) still cannot be attributed: no managed exception is raised, so no MLoop code runs to diagnose it. Measured, for the record: LightGBM's native warnings go to **stdout (fd 1)**, bypassing `MLContext.Log` and managed `Console` redirection entirely — so they cannot be captured as a diagnostic signal without OS-level fd interception, and equally they do **not** break the "exit != 0 ⇒ stderr carries the cause" contract (stderr stayed empty across the measurements).

## [0.29.1] - 2026-07-19

### Fixed
- **Feature exclusion is decided once per run instead of per data slice.** Which columns featurization drops (DateTime / sparse / constant) is a *data-dependent* judgement, and it was being made independently in three places: the saved-schema capture (which carried its own mirror of the loader's constant/sparse rules), the training-partition load, and the test-partition load. A *near-constant* column — two distinct values across the file, the minority present in only a handful of rows — is genuinely constant inside one partition and not the other, so the partitions ended up with different feature widths and the pipeline fitted on one failed on the other with `Schema mismatch for feature column 'Features': expected Vector<Single, 30>, got Vector<Single, 29>`. Data that is perfectly trainable died on an internal inconsistency. The decision now has one authority, `CsvDataLoader.DetermineExcludedColumns`, computed once from the full dataset and handed to every consumer — both partitions and the saved schema that predict and evaluate replay. The authority derives the set by running the real removal chain and diffing headers rather than restating the rules, so a second implementation cannot drift from the first; the three mirror methods in `TrainingEngine` are gone. This also closes a quieter defect the same drift produced: the saved schema could describe a column as a `Feature` that the training partition had actually dropped, promising predict a column the model was never fitted on. Reported from downstream (a 357×48 set with 3 positives) against 0.29.0, and reproduced against the published 0.29.0 binary with a synthetic 357-row fixture carrying one numeric near-constant column: `expected Vector<Single, 7>, got Vector<Single, 8>`, exit 1 — the same run exits 0 here. (The column has to be *numeric* to reproduce it: `InferColumns` merges adjacent numeric columns into the ranged vector, so dropping a text column changes the column list without changing the vector width.)

This closes the **exclusion axis** of that error signature. A second, independent way to reach the same message remains: `LoadData` runs `InferColumns` separately per file, so a column inferred as numeric in one partition and text in the other still yields different vector widths. No such case has been observed, and addressing it means inferring the loader schema once and reusing it — a larger change than this one, deliberately not attempted here.
- **A training failure's diagnosis is no longer printed twice on stderr.** The wrapping site builds its message as `Training failed for experiment {id}: {inner.Message}`, and the display layer then printed the inner message again on an `Inner:` line. Both layers were individually reasonable, and while exception messages were one-liners the repeat was invisible; 0.29.0's untrainable-class diagnosis (class distribution plus remediation options) made it a duplicated block — 1375 bytes of stderr for roughly half that much information. The display layer now suppresses the `Inner:` line when the outer message already contains it, and still shows it when the inner exception adds something new.

### Changed
- `IDataProvider.LoadData` takes an optional `featureExclusions` argument — the decided exclusion set to apply verbatim, instead of re-deriving it from the file being loaded. Non-tabular loaders (image directories, COCO/YOLO) ignore it, as they already do `GetMergedColumnGroups`.
- When a model has **no recorded exclusion schema** (experiments trained before schema capture), predict and evaluate still reconstruct the exclusions from the inference data — there is no recorded decision to apply — but they now say so instead of doing it silently, since a reconstruction that disagrees with training surfaces only as an opaque feature-vector size mismatch.

## [0.29.0] - 2026-07-19

### Fixed
- **Classification train/test splitting is now stratified.** The general split path drew rows uniformly, so a rare class could land entirely on one side: with 3 positives in 357 rows, a 20% split produced a positive-free test partition about half the time, leaving every both-class metric (AUC, F1, precision/recall) undefined and sending the run through the AUC → F1 → manual-SDCA fallback chain — which swaps the *metric* axis and could therefore never address a *split* problem. `TrainingEngine` now routes binary/multiclass runs through the existing `CsvSplitter.StratifiedSplit` (which guarantees each class reaches both partitions) and hands the result to AutoML as a pre-split pair. The split files are written to a temp directory and deleted afterwards, so `datasets/` is not polluted; `--balance`, pre-split (`--test`), time-series, and directory-based tasks are untouched. Per-class counts are reported (`Stratified split: train=…, test=… ('A' 48/12, …)`) so the split is auditable. Note that ML.NET's `samplingKeyColumnName` is *grouping*, not stratification — passing the label there would force every positive onto one side; the fix deliberately does not use it.
- **Untrainable class shapes are rejected before training instead of failing deep inside ML.NET.** A class with fewer than 2 samples cannot appear in both partitions (provable from the split arithmetic). When removing such classes leaves **fewer than two classes to distinguish**, training now stops immediately with the class counts and the reason, rather than consuming the entire time budget and dying inside AutoML. Shapes that still have two or more viable classes (e.g. multiclass 80/80/1) continue to train with a warning naming the starved class — measured against the previous release, which trained and promoted that shape successfully, so the check does not remove working capability. The per-class minimum-sample check is no longer multiclass-only, so binary datasets now surface it too; its messages became task-neutral (`🔴 Multiclass training at risk` → `🔴 Training at risk`, `🟡 Multiclass training may be unstable` → `🟡 Training may be unstable`).
- **`TrainingConfig` silently lost fields when re-wrapped, disabling prep pre-featurizers on the default path.** Three hand-written copy sites dropped `TestDataFile`, `PreFeaturizer`, and `PreFeaturizerColumns`; two of them sit on the auto-time path taken whenever `--time` is omitted, so a run using `prep` without an explicit time budget could have its pre-featurizer silently discarded. `TrainingConfig` is now a `record` and the copies use `with`, making the omission structurally impossible rather than fixed one site at a time.

### Known limitation
- AutoML's **internal cross-validation remains unstratified**. Under extreme imbalance a run therefore still exhausts the time budget once before falling back to manual SDCA (results are correct; wall-clock is roughly doubled). Injecting a stratified validation set via `experiment.Execute(trainData, validationData, …)` would change validation for *every* classification run and is intentionally out of scope here.

## [0.28.0] - 2026-07-19

### Fixed
- **`exit != 0` now guarantees a non-empty stderr.** Domain errors were written entirely to stdout while stderr stayed empty, so a machine consumer following the POSIX convention (capture stderr on failure) recorded a blank cause even though the CLI had produced an excellent diagnostic — the parse errors it *did* see on stderr came from System.CommandLine, not from MLoop. This is fixed at the boundary rather than message by message: a new `ErrorConsole` (stderr `IAnsiConsole`) backs `ErrorSuggestions.DisplayError`/`DisplayTrainingError`, through which every command's top-level catch already passes, and the scattered early-exit `Error:`/`Tip:` lines across 18 command files were converged onto it. `TrainCommandLogger.Error` and `MLoop.Core`'s `ConsoleLogger.Error` also route to stderr; narration (Info/Warning/Debug) and the rich human-facing output stay on stdout. `promote` and `detect`, which used their own `WriteError` helper, mirror the cause to stderr while keeping the JSON envelope clean on stdout — so the contract holds in `--json` mode too. This makes the CLI obey a rule it had already written down for itself (D19, `Program.cs`: "Diagnostics must never pollute stdout").
- **A metric-selection sentinel was persisted as the metric's name.** Experiments recorded `metricName: "auto"` instead of the resolved metric, which broke metric provenance for consumers and defeated the 0.27.0 `directionSource` chain (an unrecognized name resolves to `default`, so the direction stamp is withheld). The discriminator was not `--name`, as first suspected, but whether the model had a `training.metric` entry in `mloop.yaml` — `init` writes a resolved name for the models it creates, so `train --name <unregistered-model>` (a form `init` itself recommends) produced the defect while the default model happened to work. `ConfigMerger` now resolves the sentinel through the `TaskMetadata` single authority at the one point where the task is known, so *any* path — registered or not — records the real metric; a yaml that already contains `metric: auto` recovers on the next training run with no manual edit. Tasks with no canonical primary metric (object-detection, unknown) keep `auto` and its deferred semantics. The promotion message no longer leaks the sentinel either (`Better auto than…` → `Better accuracy than…`), which also restores staging-reason messages that were silently vanishing when `result.Metrics["auto"]` failed to resolve. Optimization targets are unchanged — this is a provenance fix, and scores from before and after remain comparable.

## [0.27.1] - 2026-07-19

### Changed
- **CP949/EUC-KR encoding detection now delegates to FilePrepper (0.7.2 → 0.7.3), removing a duplicated implementation.** `MLoop.Core.Data.EncodingDetector` carried its own ~180-line copy of the byte-pattern detection (BOM / UTF-8 validity / Korean-pattern) that FilePrepper — the file-prep layer MLoop.Core already depends on — also implements. The two could drift; MLoop's convenience surface (`DetectionResult`, and the `ConvertToUtf8WithBom` policy ML.NET's TextLoader needs) is now a thin adapter over FilePrepper's single authority. Detection results are unchanged, verified by the existing detector tests (UTF-8 with/without BOM, UTF-16 LE/BE, CP949, and the KAMP-style Korean-header + ASCII-body case). FilePrepper 0.7.3 additionally fixes a boundary bug where a large UTF-8 file whose 64KB detection sample cut a multibyte character was misdetected as CP949 — MLoop inherits that fix.
- `DetectionResult.Confidence` removed (no consumer read it; the delegated authority reports a decided encoding, not a score). `HasBom` now recognizes UTF-16 LE/BE marks in addition to UTF-8.

## [0.27.0] - 2026-07-19

### Added
- **`mloop compare --metrics-file` now exposes judgment provenance** — three additive JSON fields so a consumer delegating best-selection can keep its own conservative policy instead of trusting a silent default:
  - `directionSource: "known" | "default"` — whether the `MetricDirection` authority actually recognized the metric's direction, or fell through to the `maximize` default. Previously an unrecognized *lower-is-better* custom metric was silently ranked as maximize, making the **worst** model "best" with no signal. Backed by a new `MetricDirection.IsKnown(metric)` on the single-source authority.
  - `excluded: [{"id","reason":"metric-missing"}]` — candidates lacking the requested metric are now reported explicitly rather than silently dropped from `ranking`, so a consumer can distinguish "won" from "was the only one left".
  - `tie: true|false` — set when the top two values are equal (best is otherwise input-order-dependent), letting a consumer apply its own tie policy (e.g. "keep champion").

  All three are additive (existing `metric`/`direction`/`best`/`ranking` unchanged). Addresses the empirical finding (All.Models TH-D4 adoption) that a provided-state judgment surface must expose provenance for a conservative consumer to delegate to it — a `maximize` default indistinguishable from an authoritative determination blocked full best-selection delegation.

## [0.26.0] - 2026-07-17

### Added
- **`mloop compare --metrics-file <path> --json`** — a provided-state comparison mode that ranks candidates and selects the direction-aware best from a supplied JSON array of metrics, **without a local `.mloop/` project**. Input `[{"id":"exp-a","metrics":{"r_squared":0.87}}, ...]` (use `-` for stdin); output `{"metric","direction","best","ranking":[...]}`. Distributed / external-SoR consumers — whose experiments live in their own database rather than a local `.mloop/` — can now delegate mloop's direction-aware best-selection (reusing the same `MetricDirection` authority the local path uses) instead of re-implementing it and risking semantic drift. `--metric` picks the ranking metric (inferred when candidates share exactly one). Read/decide only; no project state is touched. Addresses the empirical finding that `promote --decide-only` (0.24.0) can't serve a distributed consumer whose ephemeral checkout holds only a single experiment (no champion present locally to compare against).

## [0.25.0] - 2026-07-17

### Changed
- **The `init` → `train` project workflow now defaults to auto-time** (data-size-based estimation clamped to `[30, 1800]s`), exactly as `train --help` has always advertised. Two changes together deliver this: (1) `mloop init` no longer hardcodes `time_limit_seconds: 300` in the generated `mloop.yaml` — the key ships as a commented example; (2) the config merge now treats an **unspecified `time_limit_seconds` as auto-time (null)** rather than seeding a fixed 300s, so the auto-time gate is reachable through a project config. Previously the hardcoded 300s *and* the merge's 300s seed both made auto-time reachable only in the yaml-less single-file path; 0.24.0's `--auto-time` opt-in remains the way to force auto-time when a project *does* set an explicit `time_limit_seconds`. **Impact:** new projects get training time scaled to their data instead of a fixed 300s — set an explicit `time_limit_seconds` in `mloop.yaml` (or pass `--time`) to opt back into a fixed budget. Existing projects that already set `time_limit_seconds` are unaffected. Resolves the contract inconsistency where `--help` promised auto-time by default but `init` + the config merge silently defeated it. The fixed 300s remains the fallback only when auto-time is explicitly disabled (`--no-auto-time`).

## [0.24.0] - 2026-07-17

### Added
- **`mloop train --auto-time`** — an opt-in flag (the symmetric twin of `--no-auto-time`) that forces data-size-based automatic time estimation **even when `mloop.yaml` sets `time_limit_seconds`**. Previously auto-time was reachable only in the yaml-less single-file training path: the `mloop init` template hardcodes `time_limit_seconds: 300`, so any project-workflow train (`init` → `train`) silently used the fixed 300s and never the `TimeEstimator`, despite `--help` advertising auto-time as the default. A consumer (fire-and-forget distributed MLOps) that wanted "training time scaled to data size" through the project workflow had no way to request it without editing the yaml schema. `--auto-time` gives that opt-in without coupling the consumer to the yaml format. The training-config summary now shows `Time Limit: auto (estimated from data size)` when auto-time is active (previously it printed the yaml/default `300s` even while the engine estimated a different value — an honest record-vs-actual fix that also covers the pre-existing yaml-less auto-time path). Contradictory combinations (`--auto-time` with `--no-auto-time`, with an explicit `--time`, or on image/directory tasks) are rejected with an actionable error. **Note:** whether `mloop init` should stop hardcoding `time_limit_seconds` — making auto-time the project-workflow *default* and the `--help` statement literally true — is a separate product decision (it changes every new project's behavior from a predictable 300s to an adaptive 30–1800s) and is intentionally **not** changed here.
- **`mloop promote --decide-only`** — reports which experiment the direction-aware `--best`/`--latest` selection *would* promote **without moving the production pointer** (`applied: false` in `--json`; `wouldReplace` flags whether it would displace the current production). Lets a consumer that owns its own system-of-record delegate mloop's best-selection logic (rather than re-implementing direction-aware metric comparison, risking semantic drift) while applying the promotion in its own store. The decision reuses the existing `SelectExperiment` primitive; only the write side (backup / pointer move / history) is skipped.

## [0.23.2] - 2026-07-15

### Security
- **`Microsoft.Bcl.Memory` promoted to a direct `PackageReference` on `MLoop.Core` (pinned 9.0.14, clears CVE-2026-26127 / GHSA-73j8-2gch-69rq / NU1903).** The vulnerable 9.0.4 enters the graph *transitively* via the ML.NET stack; MLoop's repo build already overrode it to 9.0.14 through `CentralPackageTransitivePinningEnabled`, but **a CPM transitive pin is repo-build-scoped and does not cross a `ProjectReference` boundary** — a submodule / in-process consumer of `MLoop.Core` silently re-inherited the vulnerable 9.0.4 (confirmed by an external consumer whose NU1903 reappeared when it dropped its own workaround pin). Declaring the (CPM-versioned) reference *directly* makes 9.0.14 a resolved direct dependency so the fixed floor now propagates to every `ProjectReference` consumer automatically. NuGet consumers were already covered via the transitive-pin in the resolved dependency set; this closes the `ProjectReference` gap. Verified `dotnet list package --vulnerable --include-transitive` is clean across all projects. Consumers on ≥ 0.23.2 can drop any manual `Microsoft.Bcl.Memory 9.0.14` workaround pin.

### Documentation
- **New [`docs/EMBEDDING.md`](docs/EMBEDDING.md) — a durable guide for in-process consumers of `MLoop.Core`.** The tabular-slim DL-pruning `ExcludeAssets` recipe (previously only in the 0.23.0 changelog) now has a stable home, with two clarifications surfaced by a consumer: (1) the recipe must live in the **publishing** project (the executable), not an intermediate library — `ExcludeAssets` does **not** flow transitively through a `ProjectReference`, so a shared `.props` `<Import>`ed by each publishable entry point is the working pattern; (2) **non-CPM consumers** must specify versions explicitly (the versionless form fails NU1015), and should **derive** them via `dotnet list package --include-transitive` rather than freezing a hardcoded trio that drifts on every `Microsoft.ML.AutoML` bump. Linked from the README documentation index.

## [0.23.1] - 2026-07-15

### Security
- **Microsoft.OpenApi 2.4.1 → 2.10.0** clears the high-severity advisory **GHSA-v5pm-xwqc-g5wc (NU1903)** that `MLoop.API`/`MLoop.API.Tests` inherited transitively through Swashbuckle. The advisory is patched at 2.7.5 within the 2.x line, so the fix stays on the Swashbuckle-compatible 2.x major (3.x breaks Swagger with 500s on `/swagger/v1/swagger.json`) — no code change and no migration off Swashbuckle. Verified `dotnet list package --vulnerable` is clean across all projects and the existing Swagger endpoint tests (`SwaggerEndpoint_*`) still pass, confirming `Swashbuckle.AspNetCore 10.2.3` + `Microsoft.OpenApi 2.10.0` serve the document without regression.

### Changed
- **Dependency servicing (low-risk, .NET 10 train alignment):** `Microsoft.AspNetCore.Authentication.JwtBearer` 10.0.5 → 10.0.10, `Microsoft.AspNetCore.Mvc.Testing` 10.0.5 → 10.0.10, `Microsoft.Extensions.*` (Caching.Memory/DependencyInjection/Logging/Logging.Console) 10.0.5 → 10.0.10, `System.CommandLine` 2.0.5 → 2.0.10, `System.IdentityModel.Tokens.Jwt` 8.16.0 → 8.19.2, `Swashbuckle.AspNetCore` 10.1.7 → 10.2.3, `Microsoft.NET.Test.Sdk` 18.4.0 → 18.8.1, `Microsoft.SourceLink.GitHub` 10.0.201 → 10.0.301.
- **Ecosystem package bumps:** `DataLens` 0.13.1 → 0.13.3, `FilePrepper` 0.7.1 → 0.7.2 (both consumed transitively — DataLens 0.13.3 requires FilePrepper 0.7.2).
- **Deliberately frozen (not bumped):** ML.NET stack (`Microsoft.ML`/`AutoML`/`TorchSharp`/`Vision`, native-coupled), `Serilog 4.3.0` (D11 bundle skew), `Microsoft.Bcl.Memory 9.0.14` (CVE override), and the OpenApi 2.x / Swashbuckle compatibility pairing. Majors deferred to individual evaluation: YamlDotNet 17→18, coverlet 8→10, Spectre 0.55→0.57, Roslyn 5.3→5.6 (removal candidate via the tabular-slim scripting-split follow-up).

## [0.23.0] - 2026-07-15

### Added
- **New `MLoop.Core.DeepLearning` assembly** isolates all deep-learning task support (the six DL handlers — image-classification, text-classification, sentence-similarity, ner, object-detection, question-answering — plus `ObjectDetectionEvaluator`, `ObjectDetectionPredictor`, `ImageDirectoryLoader`, `ObjectDetectionDataLoader`) behind an `IDeepLearningModule` seam registered once at startup via `DeepLearningRegistry`. `MLoop.CLI`/`MLoop.API` reference the new assembly and register the module, so their DL behavior is unchanged. A tabular-only in-process consumer that references `MLoop.Core` alone gets an actionable "requires the MLoop.Core.DeepLearning package" error for DL tasks instead of a crash or silent success.

### Changed
- **`MLoop.Core` no longer carries a compile-time reference to `Microsoft.ML.TorchSharp` / `Microsoft.ML.Vision`.** The DL task handlers were extracted (see Added), the `AutoMLRunner.RunAsync` task switch and `DataLoaderFactory.Create` now dispatch DL tasks through `DeepLearningRegistry`, and the two `PackageReference`s were removed from `MLoop.Core.csproj`. `MLoop.Core.dll` is now **Torch/Vision-free at the IL level** — pinned by two boundary-guard regression tests (a referenced-assembly assert and a `MLoop.Core.csproj`-text assert). No behavior change for tabular tasks (binary/multiclass/regression, clustering, ranking, forecasting, time-series-anomaly, anomaly-detection, recommendation).
  - **Consumer note — pruning the transitive DL weight:** this split does **not** by itself shrink a tabular-only consumer's `dotnet publish --self-contained` output, because `Microsoft.ML.AutoML` (MLoop.Core's core engine, irremovable) transitively depends on `Microsoft.ML.TorchSharp` + `Microsoft.ML.Vision`. What the split delivers is that excluding those transitive DLLs is now **safe** (pre-split, `MLoop.Core.dll` referenced Torch types, so pruning the DLLs would crash Core on load). A tabular-only consumer prunes them by adding, to its own project:
    ```xml
    <PackageReference Include="Microsoft.ML.TorchSharp" ExcludeAssets="all" />
    <PackageReference Include="Microsoft.ML.Vision" ExcludeAssets="all" />
    <PackageReference Include="TorchSharp" ExcludeAssets="all" />
    ```
    Verified: the tabular self-contained publish then contains no TorchSharp/Vision/LibTorchSharp assemblies and tabular AutoML still runs.
- **A directory-based task (`image-classification`/`object-detection`) whose DL data loader is unavailable now throws an actionable error instead of silently falling back to the CSV loader** (`DataLoaderFactory.Create`) — closing a latent silent-wrong-loader trap surfaced during the split's final review.

## [0.22.1] - 2026-07-12

### Fixed
- **Degenerate all-NaN-scoring regression models no longer pass the quality gate or pollute predict output** (live repro: an 8-row perfectly-linear regression trained a FastForest that scored `NaN` for *every* input). Three compounding silent failures fixed at their roots:
  - **Train**: ML.NET's `Regression.Evaluate` silently filters non-finite-scored rows, so an all-NaN holdout yielded fabricated all-zero metrics — `r²=0` **and** `rmse=0`, a mathematically contradictory pair that sailed through the quality gate and got promoted. `AutoMLRunner` now detects the all-non-finite holdout directly (`AllScoresNonFinite`) and marks the core metrics `NaN`, so the existing `MetricSanitizer` funnel records them as direction-aware worst-sentinels (with its "too few samples" warning), and conformal calibration over meaningless residuals is skipped.
  - **Promotion gate**: `MetricSanitizer`'s contract says a worst-sentinel metric "must never pass the promotion gate", but the gate only enforced floor thresholds for higher-is-better metrics — a regression optimized on `rmse`/`mae`/`mse` had **no floor at all**, so even the honest `rmse = double.MaxValue` sentinel still auto-promoted as the first production model (verified live). `ShouldPromoteAsync` now rejects any experiment whose gate metric carries the direction-aware undefined-metric sentinel (`MetricPolicy.IsUndefinedMetricSentinel`), closing the gate for both metric directions.
  - **Predict (CSV)**: `SaveAsText` serialized the `NaN` scores as literal `?` into the user's output CSV with exit 0 — silent data pollution indistinguishable from real predictions downstream. The CSV path now runs the same all-degenerate guard as the `--json`/serve path (`PredictionService.RequireNonDegenerateScoredView`) before writing, failing fast with an actionable degenerate-model error (new `DegeneratePredictionException`, a `InvalidOperationException` subtype).
  - **Predict (`--json`/serve)**: non-finite signal values (`Score`, band bounds, `Probability`, per-class probabilities, cluster distances, anomaly scores) flowed into `PredictionRow` and crashed `System.Text.Json` ("infinity cannot be written as valid JSON", exit 1 with empty stdout). They are now scrubbed to `null` at the single row-materialization authority — the all-NaN case is caught honestly by the degeneracy guard, and partially-degenerate rows serialize as honest `null`s (their `Confidence` is `null` instead of a fabricated `0`).

## [0.22.0] - 2026-07-09

### Added
- **`mloop predict --output <file>` now carries a normalized `confidence` column**: the per-prediction confidence that `predict --json`/`serve` already expose (∈ [0,1], from the single `ConfidencePolicy` authority — classification max-prob, anomaly boundary-distance, regression band-width vs `residual_std`) was absent from the CSV output path, so a consumer wanting *both* a CSV file and confidence had to run `mloop predict` twice (once for `--output`, once for `--json`). Root cause was the two predict paths being separate engines: the structured path materialized `PredictionRow`s and filled `Confidence` via `ConfidencePolicy`, while the CSV path took the raw ML.NET `IDataView` straight to `SaveAsText` and never called the policy. Rather than re-implement the confidence rules a second time in ML.NET column space (which is exactly the drift `ConfidencePolicy`'s single-authority design exists to prevent), a new guard-free `PredictionService.ComputeRowConfidences(IDataView, taskType, interval)` seam reuses `ExtractRows` + `ConfidencePolicy`, and the CSV path appends an **additive trailing `Confidence` column** after `SaveAsText` (existing columns unchanged). For banded regression the confidence is derived from the actually-written `ScoreLowerBound`/`ScoreUpperBound` band, so the CSV `Confidence` matches the structured path row-for-row (heteroscedastic included). Tasks with no confidence signal (bare-point regression, clustering, ranking, recommendation, time-series-anomaly) omit the column entirely, and enrichment failure never breaks the core CSV (graceful degradation). Pinned by `ComputeRowConfidences` unit tests and `PredictConfidenceColumnTests` (CSV↔service parity).

### Fixed
- **`mloop predict --log` no longer logs a trailing enrichment column as the prediction**: `LogPredictionsAsync` selected the prediction from each output row by *position* (`dict.Values.LastOrDefault()`, "last column = prediction"). That was already latently wrong for banded regression (it logged `ScoreUpperBound`), and adding the additive `Confidence` column above made it log the *confidence* (e.g. `0.130759`) instead of the actual `Score` (e.g. `26.47`) for every confidence-bearing task — corrupting the prediction logs that feed feedback/trigger/drift. Fixed at the root by selecting the prediction column **by name** (`PredictCommand.SelectPredictionValue`): `PredictedLabel` (classification/clustering/anomaly-detection) → scalar `Score` (regression/ranking); when neither exists (time-series-anomaly emits a `Prediction` vector) the last-column fallback now **skips** the known enrichment columns (`Confidence`/`ScoreLowerBound`/`ScoreUpperBound`) so an appended enrichment is never mistaken for the prediction. As a deliberate consequence, classification now logs `PredictedLabel` (was `Probability`) and anomaly-detection logs the `PredictedLabel` verdict (was the continuous `Score`), both better matched to the "predicted-vs-actual" comparison the log feeds. Pinned by `PredictLogColumnSelectionTests`; verified end-to-end (`--log` now records `26.46964` not `0.130759`).

## [0.21.1] - 2026-07-08

### Fixed

- **`mloop train` (regression) no longer aborts with "Infinity cannot be written as valid JSON" on small datasets**: training a regression model whose test split is (near-)degenerate — a handful of rows, near-zero label variance — could produce a model that scored `NaN` on the holdout, which made the split-conformal *descriptive* stats (`residual_std`, `interval_half_width_{80,90,95}`) non-finite. `System.Text.Json` rejects `NaN`/`±Infinity`, so serializing `metrics.json` threw and the whole pipeline failed at save time (reported by an All.Models 0.21 subprocess smoke test: 8-row `y=2x` + `--test-split 0.2` → exit 1). Root cause was the conformal *uncertainty* metrics, not the ML.NET point metrics (ML.NET already clamps a degenerate R² to a finite 0). Two-part fix: (1) the conformal interval computations (`ComputeConformalIntervals` / `ComputeNormalizedConformal`) now **omit** any non-finite descriptive stat — absence is the honest "interval unavailable" signal, and predict/serve already fall back when the keys are missing; (2) a new single-authority `MLoop.Core.Evaluation.MetricSanitizer` normalizes any non-finite *optimization* metric (e.g. classification `log_loss` → +∞ when a class probability underflows to 0) to the **direction-aware worst** value (`double.MaxValue` for lower-is-better, `double.MinValue` for higher-is-better) resolved through the `MetricDirection` authority, so a broken measurement can never crash serialization, win best-model selection, nor pass the promotion gate. Both the training path (`AutoMLRunner.RunAsync`) and the evaluate path (`EvaluationEngine`) funnel their metric dicts through it, and the old inline `NaN → 0` guards across every path — regression, recommendation, and the deep-learning classification/regression heads (`NanSafe`) — which wrongly laundered a broken error metric (e.g. classification `log_loss`) into a perfect-looking 0, are removed in favor of the central authority. When a metric is sanitized, training now warns that the test split is likely too small. Pinned by `MetricSanitizerTests`; the reported repro now exits 0 with a valid `metrics.json`.

### Changed

- **`serve` now returns an actionable 401 body, and the JWT/machine-output contract is documented where callers look**: a programmatic caller embedding `mloop serve` (0.21 makes every endpoint except `/health` require a JWT) previously got a bare, empty 401 with no hint that a token was required nor how to mint one — the token-issuance surface is the CLI (`mloop token`), not the server, so it wasn't discoverable. The JwtBearer pipeline now has an `OnChallenge` handler that replaces the empty challenge with `{"error":"Unauthorized","hint":"… mloop token … Authorization: Bearer <token> …"}`. Alongside, the discoverability gaps are closed: `serve --help` now states the JWT requirement and points to `mloop token`; `predict --help` steers machine/automation consumers to `--json`/`--output` (the default stdout mixes a human summary); `docs/GUIDE.md` serve examples now include the `Authorization: Bearer` header and an `mloop token` step (they previously showed unauthenticated curls that would 401), and its predict options list documents `--json`; README's serve snippet shows the token flow. Pinned by `UnauthorizedChallengeTests` (real JwtBearer pipeline → 401 + hint). A `serve --anonymous` dev-bypass flag was considered and **declined** — it would reintroduce the auth regression 0.21 deliberately closed; the dev-key + Development-environment path already serves local use out of the box.

- **Dropped consumer-side `System.Security.Cryptography.Xml` transitive pin** — FilePrepper 0.7.1 raises its EPPlus floor to 8.5.4, which pulls the patched `Cryptography.Xml` 10.0.7. The upstream remediation makes MLoop's temporary pin (10.0.9, for GHSA-37gx-xxp4-5rgx / GHSA-w3x6-4m5h-cxqf) unnecessary; removed. Verified: `dotnet list package --vulnerable --include-transitive` reports no vulnerable packages and build emits 0 NU1903 after removal. **Requires FilePrepper 0.7.1 on nuget.org before restore.**

## [0.21.0] - 2026-07-07

### Added
- **`mloop detect` — one-shot time-series anomaly detection with SPC-chart bounds (no training required)**: SR-CNN whole-series detection is inherently a one-shot operation, but MLoop only exposed it through the train → save → predict pipeline — every fresh analysis paid a full training pass (10s+), unusable for near-real-time SPC monitoring. `mloop detect <data.csv> [--column <col>] [--threshold] [--sensitivity] [--period] [--output out.csv] [--json]` now runs `DetectEntireAnomalyBySrCnn` directly over the series — no model artifact, no MLoop project required, seasonality auto-detected via `DetectSeasonality` (override with `--period`, 0 = non-seasonal). Because the one-shot API is also the only SR-CNN surface with `AnomalyAndMargin` mode, every point comes back with the full SPC triple — `ExpectedValue`/`UpperBound`/`LowerBound` — plus the anomaly verdict and score, closing the long-standing bounds gap as well: the streaming `DetectAnomalyBySrCnn` transformer that trained time-series-anomaly models use emits only [alert, score, mag] and has no margin mode upstream, so bounds structurally cannot come from the trained-model predict path. The 7-slot output contract lives in one place (`MLoop.Core.Detection.SrCnnOneShotDetector` slot constants). Column selection is forgiving (case-insensitive, auto-selects a single-column CSV) and gaps fail fast with the offending row number instead of silently skipping. Pinned by `SrCnnOneShotDetectorTests` (spike detection, band coherence, index/value round-trip, option validation — MKL-guarded) and `DetectCommandTests` (column resolution, non-numeric rejection, parse surface).
- **`mloop list --json` and `mloop promote --json` — machine-readable output for the automation surface**: both commands only emitted Spectre.Console rich tables, so subprocess consumers (mloop-mcp foremost) had to scrape formatted text. `list --json` emits the experiments array (model, id, ISO timestamp, status, trainer, metric name/value, isProduction) plus the production map; `promote --json` emits the promotion result (model, experimentId, selectedBy, previousProduction, backupPath) and is non-interactive — it never prompts, matching the redirected-stdin behavior automation already relies on. Errors in JSON mode are emitted as `{"error": …}` with exit 1, keeping stdout parseable. (The OD `predict --json` gap noted above stays gated on D27 — an OD model can't be live-verified until real-data OD training stops crashing.)
- **`mloop promote --latest` / `--best` — auto-selecting the experiment to promote**: automation pipelines had to scrape `mloop list` output to find the newest experiment ID before promoting (post-`--no-promote` workflows). `--latest` picks the most recent completed experiment; `--best` picks the best-metric completed experiment, comparing through the `MetricDirection` authority so error metrics (rmse, mae, log-loss) select the *minimum* instead of naively maximizing, refusing (actionably) when experiments were optimized for different metric names — a cross-metric comparison would be meaningless — and breaking exact ties toward the newer experiment. Exactly one selector (explicit ID, `--latest`, `--best`) must be given; anything else fails with an actionable error. Pinned by seven new `PromoteCommandTests` cases covering direction-awareness, mixed-metric refusal, no-metric refusal, and tie-breaking.

## [0.20.0] - 2026-07-07

### Fixed
- **Normalized-conformal σ-model can no longer silently diverge into a constant-width band (also un-reds macOS CI)**: the heteroscedastic aux σ(x) model was a linear SDCA fit, and SDCA's async optimization trajectory is platform-/thread-sensitive on this small aux problem — it can outright *diverge* (observed: σ(x) ≈ −8.7×10⁷ on win-x64 single-threaded, and the same degeneracy on macos-arm64 CI's default fit, while win/linux multi-threaded happened to converge). Because the band formula floors σ at `max(σ,0)+β`, a diverged σ-model doesn't crash — every row silently gets the identical constant width while the experiment metadata still claims a heteroscedastic band, which is exactly why macos-latest CI had been red since 0.19.0 on the two per-row-band tests without anyone seeing a *wrong prediction*. The σ-model is now a FastForest (managed, no native deps): a forest predicts leaf averages of the |residual| target, so σ is structurally bounded to the training residual range — it cannot go negative or diverge — and it trains with `NumberOfThreads = 1` so conformal calibration is reproducible for the same data (as a bonus the aux fit dropped from ~90 s of grinding divergence to ~1 s in the affected tests). Alongside: `PredictionService`'s σ-scoring fallback now surfaces a warning naming the constant-width fallback instead of degrading silently (P-svc1's lesson), both band tests read the aux σ directly in their failure messages so a future regression reports whether the *fit* or the *serving path* degenerated, and `Predict_Multiclass_NumericLabel_MatchesTrainedSingleSchema` swaps its incidental LightGBM trainer for SdcaMaximumEntropy — GitHub's macos-latest runners are Apple silicon and lib_lightgbm ships no osx-arm64 native, which was the third red test (`DllNotFoundException`), and the pinned regression (serve-path label loading) is trainer-agnostic.

- **API integration-test factory no longer mutates the process-global working directory**: `TestWebApplicationFactory` pinned the API's project discovery to its temp root by assigning `Environment.CurrentDirectory` — process-global state. That was latent while a single test class consumed the factory, but xUnit runs test classes in parallel, and D21-A's `ForecastingApiTests` added a second factory instance: one factory's dispose could delete the directory another factory's `Program` startup was using as its working directory, killing the entry point before it built an `IHost` ("The entry point exited without ever building an IHost", 37 API tests failing on every OS). The factory now injects a fixed-root `IProjectDiscovery` stub instead of touching the CWD, so parallel factories stay isolated.

### Changed
- **Object-detection training now forces libtorch to a single thread (D27 defensive mitigation, unverified)**: real-data OD training intermittently dies with a native access violation (0xC0000005) inside libtorch, with the crash location varying between runs (`THSNN_Linear_forward`, `Tensor.backward`) — a signature upstream (dotnet/TorchSharp#1292) traces to native heap corruption whose failure point shifts with memory/thread pressure. `RunObjectDetectionAsync` now calls `TorchSharp.torch.set_num_threads(1)` before fitting, removing that pressure source. This is a defensive, no-downside change (the only cost is slower CPU training) applied unconditionally rather than gated on a confirmed fix — this investigation's development environment had ambient resource contention from concurrent unrelated workloads that prevented reliably re-triggering or clearing the crash to verify the mitigation directly (see the D27 issue for the full investigation). Full elastic thread-count tuning or an upstream TorchSharp fix remain open; this is a stopgap.

### Added
- **`POST /predict` now serves forecasting models with a horizon-based contract instead of rejecting every request (D21-A)**: forecasting was the only task where `POST /predict` always failed — SSA forecasting is stateful (it forecasts a fixed horizon ahead of its training series), so the row-based contract every other task uses cannot apply, and D21 made that rejection actionable rather than a silent all-null 200. `POST /predict` for a forecasting model now accepts an optional JSON body `{"horizon": N}` (an empty object `{}` uses the model's trained horizon — a body is still required, matching every other task's contract, but the `horizon` field itself is optional) and returns the forecast on the exact same `PredictionRow` schema `mloop predict --json` already used (`score`/`scoreLowerBound`/`scoreUpperBound`/`intervalConfidence`), so structured consumers (mloop-mcp) get one shape regardless of which surface they call. Because the saved SSA model's horizon is fixed at train time (`variableHorizon` is not enabled), a `horizon` that doesn't match the trained value fails fast with an actionable 400 naming the trained horizon, instead of silently ignoring the request or truncating/padding the result — full elastic (train-time-variable) horizon support is tracked separately as a proposal, since it would change the training contract for all future forecasting models. The replay computation (load model → replay training series from the experiment config → extract native forecast/confidence-band columns) was extracted from CLI-only `PredictCommand` into a new shared `MLoop.Core.Prediction.ForecastReplayService`, so the CLI (`mloop predict`/`--json`) and the API now share one computation and cannot drift on how a forecast is produced (the same lesson as the predict-path convergence). Pinned by new `PredictForecastingTests` (horizon-match/-mismatch) and a new `ForecastingApiTests` integration suite (empty body, matching horizon, mismatched horizon) that trains a real SSA model and seeds it as a production model end-to-end.

### Fixed
- **Label-less clustering's structured predict surfaces no longer hard-fail with a Features dimension mismatch (D24)**: `serve POST /predict` and `mloop predict --json` threw `"Schema mismatch for feature column 'Features': expected Vector<Single, 3>, got Vector<Single, 4>"` for every clustering model trained without a declared label (i.e. nearly all of them) — the CSV predict path worked by incidental shape-matching. Root cause: `CsvDataLoader` picks the first CSV column as a placeholder label so `InferColumns` can run on label-less data, excluding it from the merged `Features` vector at train time; the old `RunClusteringAsync` embedded a `Concatenate("Features", [placeholder, Features])` *inside* the fitted+saved pipeline, baking in that incidental shape. At predict time `PredictionService.EnsureFeaturesColumn` (D8) independently builds its own `Features` from every named non-excluded column (the placeholder included, since predict-time has no concept of "placeholder") — so the model's embedded `Concatenate` re-consumed that already-built 3-dim `Features` as one of *its own* two inputs, double-counting a dimension. `AutoMLRunner.RunClusteringAsync` now pre-featurizes before the K-search loop and fits only the trainer on the result, so the saved model's sole input contract is `Features` with no re-concatenation step left to collide with `EnsureFeaturesColumn`. The CLI CSV path (`PredictionEngine`) needed a matching, narrower fix: it now accepts the task type and, for label-less clustering specifically, re-concatenates `InferColumns`' placeholder-label column back into `Features` to match the new bare-trainer model's expectation. Non-destructive: pre-existing model.zip files trained under the old shape are unaffected (they keep their old embedded-concat behavior; only newly-trained clustering models get the corrected contract). Pinned by a new `ClusteringFeaturesContractTests` (RED confirmed against the pre-fix code — reproduces the exact reported error — then GREEN against the fix, covering both the row-based and CSV predict surfaces).
- **`PredictionService` now rejects a task/model mismatch instead of silently returning an all-null result (P-svc1)**: D20 (zero column overlap), D21 (stateless forecast), and D22 (unread TS-anomaly vector) were each fixed as one-off guards, but they share a single failure shape — a schema/taskType mismatch (a renamed output column, a model trained for a different task than the caller declared, an unhandled model shape) leaves every row's task-defining field null while `ExtractResults` still returns a row per input and the caller still gets a 200/success. `RequireNonDegenerateOutput` generalizes the backstop: after extraction, if there is at least one row and the task's defining output field (`PredictedLabel` for classification/clustering, `Score` for regression/ranking/recommendation, `IsAnomaly`+`AnomalyScore` for anomaly/TS-anomaly) is null on *every* row, it throws an actionable `InvalidOperationException` naming the declared task instead of fabricating an empty-looking "nothing to report" answer. A genuine all-boring-but-real result (e.g. "no anomalies detected", "cluster 0 for everyone") has non-null values and passes through untouched — only total column absence trips the guard. Pinned by four new `PredictionServiceTests` forcing real task/model mismatches (regression model declared as multiclass, ranking model declared as clustering, a bare featurizer declared as anomaly-detection/time-series-anomaly).

## [0.19.3] - 2026-07-05

### Fixed
- **Structured ranking prediction no longer fails with "Could not find input column 'Features'" (D26)**: `ranking` was missing from `PredictionService.RequiresFeaturesVectorInput`, so the structured surfaces (serve `POST /predict`, `mloop predict --json`) loaded the schema's named columns but never concatenated them into the `Features` vector the trained ranking model (embedded `MapValueToKey(GroupId)` + `Concatenate` + LightGbm/FastTree ranker) expects — every structured ranking predict hard-failed while the CSV path worked (found live on a synthetic LTR set; earlier checks had only exercised the CSV path). Adding ranking to the Features-vector task set fixes both surfaces; pinned by an end-to-end `PredictionServiceTests` FastTree-ranking round-trip mirroring the trainer's exact model shape.
- **Ranking/recommendation rows no longer fabricate a `confidence` (D25)**: `ConfidencePolicy` mapped "other scalar-score tasks" to `clamp(score)` — but a recommendation `Score` is a predicted **rating on the label's own scale** (a 1–5 rating clamps to a constant `1.0`, observed live: every MatrixFactorization row returned `confidence: 1`), and a ranking `Score` is an unbounded relative ranking score (clamping yields an arbitrary value). Both presented a fabricated "fully confident" signal to trust-loop consumers — the exact overclaim the confidence contract exists to prevent. Ranking/recommendation now honestly return `confidence: null` (per the policy's own documented semantics: "null when the model exposes no usable confidence signal") until a real order-uncertainty signal (top-k score margin) is designed and measured. The raw `score` stays on the row, so consumers deriving their own heuristics are unaffected.
- **Time-series-anomaly predictions are no longer dropped by the structured surfaces (D22)**: TS-anomaly models emit a single vector column (`Prediction` = [alert, raw score, detector-specific third slot]) instead of the `PredictedLabel`/`Score` columns every other task family produces — and `PredictionService`'s generic extraction read none of it, so `POST /predict` and `mloop predict --json` returned rows with **every field null** and success status (observed live: 5000 × `{}` with exit 0 on a KAMP sensor series — silent no-op, same family as D20/D21; the CSV predict path was unaffected since it dumps raw output columns). A dedicated extractor now surfaces the authoritative slots as `isAnomaly`/`anomalyScore`. The output contract (column name + slot meanings) is single-sourced in a new `TimeSeriesAnomalyOutput` authority consumed by the trainer chain and the extractor. Deliberately **no confidence** is derived: the raw score is a detector score (SrCnn spectral residual / SSA raw score), not a [0,1] probability with a 0.5 boundary, so `ConfidencePolicy`'s anomaly boundary-distance rule is now gated to `anomaly-detection` — running the TS raw score through it would fabricate a confidence (the TS-anomaly signal mapping is a separate, measured design). Pinned by an end-to-end `PredictionServiceTests` SrCnn round-trip (alert + raw score surfaced, spike alerts, confidence stays null) and a `ConfidencePolicyTests` fabrication guard.
- **`mloop predict --json` for time-series-anomaly no longer strips the series value column (D23)**: the structured CLI path excludes the configured label column from the input rows — correct for tabular tasks (the label is the unknown-to-predict, and stale CSV values must not leak), but for time-series tasks the "label" **is** the monitored series value column. Stripping it defaulted the entire series to zero, and the detector silently returned an all-normal fabrication (`isAnomaly: false, anomalyScore: 0` on all 5000 rows, exit 0) — a *plausible-looking wrong answer*, the worst member of the D20 silent-GIGO family. The exclusion is now task-aware (`AutoMLRunner.IsTimeSeriesTask` single source): time-series tasks keep the value column. Live-verified on the KAMP series: `--json` now reports the same 12 anomalies at the same indices with the same raw scores as the CSV path.
- **Forecasting predictions no longer fail silently on the structured surfaces (D21)**: posting rows to `POST /predict` for a forecasting model returned **200 with every field null** (`predictedLabel`/`score`/bounds/`confidence`) and `warnings: null` — the SSA forecaster is stateful (it forecasts a fixed horizon ahead of its training series), so the shared `PredictionService`'s stateless per-row `Transform` extracted nothing while the response still claimed success (silent no-op, same family as D20; observed live against a KAMP sensor time series). And `mloop predict --json` was **silently ignored** for forecasting: the horizon-based CLI branch returned before the `--json` handler, emitting zero bytes of JSON to stdout (with human output rerouted to stderr, structured consumers like mloop-mcp got nothing). Both surfaces now behave honestly: (1) `PredictionService` rejects row-based prediction for forecasting with an actionable `ArgumentException` naming the working path (`mloop predict`), which the API maps to **400** via the D20 caller-error handling — the horizon-based serve contract itself is a separate design decision and remains open; (2) `mloop predict --json` emits the horizon forecast as structured JSON on the **same payload/row schema as every other task** (`score` = forecasted value, `scoreLowerBound`/`scoreUpperBound` = the SSA native 95% band, `intervalConfidence` = its coverage level; row order = step order), sharing one computation with the CSV path; (3) the CLI forecasting path now returns **exit code 1 on failure** (missing training data, no forecast column, 0-value forecast) instead of printing an error and exiting 0. The SSA output contract (column names + confidence level) is single-sourced in `ForecastOutput` consts consumed by the trainer pipeline and the CLI predict path (drift guard). Pinned by two `PredictionServiceTests` (both overloads reject forecasting) and four `PredictForecastingTests` (real SSA model round-trip: horizon forecast with ordered bounds, actionable missing-data error, PredictionRow mapping).
- **The lazy update notice no longer pollutes machine-readable stdout (D19)**: every CLI invocation runs a lazy update check after the command completes, and the "A new version (vX.Y.Z) is available" notice was written to **stdout** via `AnsiConsole`. Any stdout-captured surface was corrupted when the notice fired: `mloop token -q` — whose own help text documents `export MLOOP_TOKEN=$(mloop token -q)` — returned `<jwt>\n\nA new version…`, producing a 401 on the very serve instance the token was minted for (observed live: a 272-char "token" = 204-char JWT + appended notice), and `mloop predict --json` risked trailing non-JSON garbage. The notice is diagnostics, not command output, so it now writes to **stderr** (a dedicated `AnsiConsoleOutput(Console.Error)` console) — fixing every current and future machine-readable surface at once rather than suppressing per command.
- **`/predict` silent garbage-in-garbage-out: input rows sharing no column with the trained schema are now rejected with an actionable error (D20)**: `PredictionService` filled *every* schema column with its default when the caller's rows contained none of them (e.g. posting the envelope `{"rows":[...]}` instead of the documented bare JSON array — the whole payload became one row with a single unknown "rows" column). The model then emitted an all-zero score vector whose argmax fabricated a plausible-looking label, returned as **200 with `warnings: null`, `confidence: 0`** — indistinguishable from a genuine low-confidence prediction and actively misleading for trust-loop consumers (observed live against an image-classification model: alphabetically-first class returned for arbitrary garbage input). `PredictionService.Predict` now throws an `ArgumentException` naming the expected vs. provided columns when the overlap is empty (partial overlap keeps the existing default-fill behavior — that is legitimate), and the API maps caller-error `ArgumentException` to **400** instead of the blanket 500. Pinned by two `PredictionServiceTests` (zero-overlap throws with actionable message; partial overlap still predicts).

## [0.19.2] - 2026-07-05

### Changed
- **Production pointer is now single-sourced in `production/metadata.json`; the duplicate `registry.json` write path was removed (drift guard, no behavior change)**: "which experiment is in production" (plus its promotion-time snapshot — metrics/task/trainer/label) was written to *two* artifacts by every promote — `production/metadata.json` and the `production` entry of a sibling `registry.json` — carrying the same fields. Three readers then split across them: the Ops `FilePromotionManager` read `metadata.json` while the CLI `ModelRegistry.GetProductionAsync` and `ModelNameResolver` (mloop list) read `registry.json`. Two artifacts holding one fact could silently diverge — the same cross-artifact drift class (promote silently skipping the production backup) came from. A new Core authority `MLoop.Core.Storage.ProductionMetadata` (a typed view of `production/metadata.json`, co-located with `model.zip`) is now the sole source: all three readers consume it, `promote` no longer writes `registry.json`, and `ExperimentLayout.RegistryFileName` is removed. `GetProductionAsync`'s `ModelInfo` contract is unchanged (every field it returned lives in `metadata.json`), so API/cache/CLI consumers are unaffected. Non-destructive: pre-existing on-disk `registry.json` files are left as harmless stale sidecars (no reader consults them). A consistency guard test pins that after a promote the CLI readers agree on the pointer and `registry.json` is not recreated; `ProductionMetadata` round-trip/casing is covered directly. (Follow-up to the predict-path convergence; supersedes the earlier registry-name single-sourcing.)
- **Heteroscedastic regression band width is now single-sourced across both predict paths (drift guard, no behavior change)**: the per-row conformal half-width `q·(max(σ(x),0)+β)` was implemented twice — `PredictionService.ExtractRegressionRows` (serve `/predict` + CLI `--json`) resolved it through the `RegressionInterval.WidthFor` primitive, while `PredictionEngine.ApplyConformalBand` (CLI `mloop predict` CSV path) re-inlined the same formula in its `CustomMapping`. The two produced identical bands today, but the inline copy could silently drift from `WidthFor` and make the CSV and the serve JSON disagree on the band while every other test still passed — exactly the cross-path drift class the Single-Source Authorities doctrine exists to prevent (the constant-width band was already single-sourced via `interval.HalfWidth`; only the heteroscedastic branch was duplicated). `ApplyConformalBand` now calls `interval.WidthFor(σ)` instead of re-deriving `q`/`β`, so `RegressionInterval.WidthFor` is the sole authority for the band width. A new cross-path equivalence test (`CrossPathConformalBandTests`) builds a real heteroscedastic model and pins that `mloop predict` (CSV) and `PredictionService` (rows) emit identical per-row bands for the same model+input, so the two can no longer drift undetected. The broader `PredictionEngine`↔`PredictionService` divergence (CSV-batch vs. structured output, `InferColumns` vs. schema-driven loading, ranking/recommendation preserve-columns) is a legitimate divergence and stays as-is (no concrete driver — YAGNI).

## [0.19.1] - 2026-07-05

### Fixed
- **In-process capture→predict round-trip: `RunAsync`'s captured schema is now directly consumable by `PredictionService` (upstream-008)**: the natural in-process API path — `AutoMLRunner.RunAsync` → `result.Schema` → `PredictionService.Predict(rows, schema, model, task, label)` — failed for every tabular task, reported by an embedded `MLoop.Core` consumer (Forge). Two producer-side defects: (1) **vocabulary drift** — `CaptureInputSchema` wrote raw .NET type names (`"Single"`, `"Double"`, `"Int32"`, …) into `ColumnSchema.DataType`, while the consumers (`PredictionService.MapDataTypeToDataKind`, `CsvDataLoader.MapTrainedTypeToDataKind`, `CategoricalMapper`, `EnsureFeaturesColumn`) only understand the semantic vocabulary the CLI schema writer produces (`"Numeric"`/`"Categorical"`/`"Text"`/`"Boolean"`), so every numeric feature fell through the `_ => String` fallback and `Transform` threw (`Type 'String' is not supported by NAReplaceTransform`). The same drift class; the predict-side `"String"` tolerance was a symptom patch of this root cause. The vocabulary is now single-sourced in **`MLoop.Core.Models.SchemaDataTypes`** (constants + producer-side `FromRawType` + consumer-side `ToDataKind` with legacy raw-name tolerance) and both producers (Core `CaptureInputSchema`, CLI `TrainingEngine`) and all consumers reference it. (2) **merged-vector capture** — `InferColumns` merges adjacent same-kind CSV columns into one ranged vector (`Features`), and the loaded IDataView carries *no SlotNames annotation* for it (verified empirically), so the captured schema was `{Features: VBuffer, label}` with the named columns row-based predict needs irrecoverably lost. A new `IDataProvider.GetMergedColumnGroups()` (default-interface-method, null by default) lets `CsvDataLoader` expose the merged-range→CSV-header mapping it alone knows, and `CaptureInputSchema` expands the merged vector back into named `Numeric` columns; the existing `EnsureFeaturesColumn` (D8/D12) reassembles the vector at predict time, so no schema format change was needed. Side fix: the in-process categorical-values capture was dead code (it keyed on `RawType.Name == "String"`, but text columns' RawType is `ReadOnlyMemory<char>`) — string feature columns now correctly capture distinct values and upgrade to `Categorical`. Pinned by `SchemaDataTypesTests` (18) and `SchemaRoundTripTests` (4 — regression and anomaly-detection round-trips reproduce the reporter's exact failure RED→GREEN). CLI/serve behavior unchanged (the CLI persists its own `CaptureInputSchemaEnhanced` schema; `autoMLResult.Schema` is the fallback).

### Changed
- **DL task handlers isolated into `AutoMLRunner.DeepLearning.cs` (upstream-007 stage 1, no behavior change)**: the six TorchSharp/TensorFlow-backed handlers (image-classification, text-classification, sentence-similarity, ner, object-detection, question-answering) moved into a dedicated partial so the `Microsoft.ML.TorchSharp`/`Microsoft.ML.Vision` usings are confined to that file plus `ObjectDetectionEvaluator`. This is preparation for an optional `MLoop.Core.DeepLearning` package (requested by embedded tabular-only consumers whose self-contained publishes carry the unused managed DL assemblies): the future extraction is now a mechanical file move. Native DL runtimes were already task-gated at run time by `RuntimeManager.EnsureRuntimeForTask`; the managed-layer split is tracked in the project's roadmap.
- **`ConfidencePolicy` documents the regression-confidence usage boundary (doc-only, no behavior change)**: the normalized regression confidence (and the band width it derives from) measures a prediction's **magnitude uncertainty**, not its **decision-boundary risk** — and the two are anti-correlated. On a live KAMP anodizing-thickness regression set the Pearson correlation between band width and the prediction's distance from a fixed pass/fail threshold T was ≈ +0.88 (heteroscedastic variance scales with magnitude, so wide-band rows sit *far* from a mid-range T, not near it). Ranking rows by lowest confidence / widest band to escalate a threshold (pass/fail) decision therefore routes confidently-far-from-line rows and misses the near-boundary rows where decision errors actually live (measured escalation precision 0.00 for width/confidence ranking vs. recall 0.91 for a band-straddles-T predicate). A new `<remarks>` block on `ConfidencePolicy` states this boundary and directs decision-escalation consumers to the band-straddles-T signal (`lower &lt; T &lt; upper`, which needs the domain threshold T as input) instead of the scalar confidence. Prevents consumers (HoneAI, U-Vision, mloop-mcp) from misapplying the confidence contract to threshold decisions. (Measured on a live reliability-loop run.)

## [0.19.0] - 2026-07-04

### Fixed
- **`mloop serve` now works out-of-the-box from the installed `dotnet tool` — the `mloop` package bundles `MLoop.API` (D11)**: `mloop serve` launches the REST API as a `dotnet MLoop.API.dll` subprocess, resolving the assembly via `ServeCommand.FindApiAssembly` (search path #2 = `AppContext.BaseDirectory/MLoop.API.dll`, "for bundled deployments"). But the `mloop` tool package never contained `MLoop.API.dll`, so a plain `dotnet tool install -g mloop` produced a `serve` that could not start without a manual `MLOOP_API_PATH` override or a from-source build (friction ①, re-confirmed by a consumer on 0.18.3). The fix is pure packaging: a new `BundleApiIntoTool` target in `MLoop.CLI.csproj` publishes `MLoop.API` and copies **only the files it adds over the CLI's own output** (a build-time set difference by relative path) into the tool's `tools/<tfm>/any/` directory — 22 files, ~2.3 MB (`MLoop.API.dll`/`.deps.json`/`.runtimeconfig.json`, `appsettings.json`, and the ASP.NET-specific managed DLLs: JwtBearer, Swashbuckle, Serilog.AspNetCore, OpenApi, IdentityModel.Protocols). The heavy `Microsoft.ML` native runtimes (~400 MB across all RIDs) and the shared `MLoop.Core`/`DataStore`/`Ops` managed DLLs are already in the tool output, so they are **not** duplicated (a full-publish subdir bundle would have doubled them and blown NuGet's package-size limit). The set difference compares against `$(OutputPath)` (never mutated by the copy) so repeated local `dotnet pack`s are deterministic, and the separate `dotnet publish` invocation breaks the `MLoop.API → MLoop.CLI` ProjectReference cycle. This surfaced a latent transitive-dependency skew — the CLI graph resolved `Serilog 4.0.0` (via `Serilog.Sinks.Console`) while the API graph resolved `Serilog 4.3.0` (via `Serilog.AspNetCore`); with `Serilog.dll` shared by name the bundled API child failed to bind `Serilog 4.3.0.0`. A central `Serilog 4.3.0` pin in `Directory.Packages.props` (with `CentralPackageTransitivePinningEnabled`) aligns both graphs, eliminating the skew at the source. `serve` runs the API framework-dependent, so the **ASP.NET Core 10 runtime must be installed on the host** (self-containing it would re-duplicate the ML natives). Verified end-to-end: a `--tool-path`-isolated install of the packed tool, run in a fresh project outside the source tree with no `MLOOP_API_PATH`, served `GET /health` → `200 {"status":"healthy"}` with the bundled API child starting cleanly.

### Added
- **Regression prediction intervals (split-conformal)**: regression models now carry a distribution-free uncertainty band, giving the regression trust-loop the signal that classification gets from its confidence probability. At train time, `AutoMLRunner.ComputeConformalIntervals` derives split-conformal half-widths from the absolute residuals of the held-out test predictions (the same holdout used for metrics) — the finite-sample empirical quantile of `|Label - Score|` for the 80/90/95% levels, giving a marginal-coverage guarantee `P(|y - ŷ| ≤ q) ≥ level`. It is **model-agnostic** (works with whatever trainer AutoML selects, unlike ML.NET's trainer-locked quantile-regression forests, which would break the free trainer sweep the quality gate depends on) and persists as flat scalars in `metrics.json` (`interval_half_width_{80,90,95}`, plus `residual_std`) with no schema change. At predict time both paths surface the primary 90% band as `[Score - q, Score + q]`: `POST /predict` adds `scoreLowerBound`/`scoreUpperBound`/`intervalConfidence` to each regression row, and `mloop predict` adds `ScoreLowerBound`/`ScoreUpperBound` columns to the CSV. A single source (`RegressionInterval.FromMetrics`) decides which band both paths expose, so serve and CLI never drift. Backward-compatible: models without a stored band emit point estimates only.
- **Heteroscedastic (per-row) regression intervals (normalized split-conformal)**: the constant-width band above cannot triage — every row gets the same half-width, so band width carries no per-row signal (measured: recall@10% of the largest-error rows = 0.10, i.e. random). MLoop now fits an auxiliary residual model σ(x) and calibrates a normalized quantile so each prediction's band is `ŷ ± q·(max(σ(x),0)+β)` — wide where the model is locally uncertain, narrow where it is confident (Papadopoulos/Lei normalized conformal). `AutoMLRunner.ComputeNormalizedConformal` splits the held-out test set into two folds (σ fit on C1, q calibrated on C2) so marginal coverage is preserved, uses an SDCA aux model over the pre-featurized `Features` vector (no new trainer dependency, captures linear heteroscedasticity), and a β floor (p10 of C1 residuals) prevents zero-width bands. It persists as `residual-model.zip` beside `model.zip` plus flat scalars `norm_interval_q_{80,90,95}` and `interval_beta` in `metrics.json`; `promote` copies the σ-model into production. `RegressionInterval.FromMetrics` auto-detects the normalized keys and switches both predict paths to per-row widths, falling back to the constant band when the σ-model or its metrics are absent (fully backward-compatible). Validated on a real KAMP plating dataset (n=431): 426/431 distinct per-row widths, **recall@10% of large-error rows 0.10 → 0.23 (2.3×)** while empirical 90% coverage stayed valid (93.3%). HoneAI's regression trust-loop consumes the band width as its escalation confidence (`1 − min(halfWidth/|Score|, 1)`) — narrow band = trust, wide band = escalate.
- **Normalized confidence contract + `mloop predict --json`**: which output field carries a prediction's uncertainty — and how it maps to a `[0,1]` confidence — is MLoop task-family knowledge, but two consumers (U-Vision, HoneAI) were independently re-deriving it from raw `probabilities`/`score` and drifting apart (e.g. HoneAI mapped the anomaly boundary distance, U-Vision did not). MLoop now owns it: `MLoop.Core.Prediction.ConfidencePolicy` is the single authority (classification → max class probability, anomaly → decision-boundary distance `|score−0.5|·2`, regression → conformal band width normalized by `residual_std`, ranking/recommendation → the score), and `PredictionService` fills a new nullable `confidence` field on every prediction row. `POST /predict` exposes it automatically (rows are serialized directly), so consumers stop re-deriving it. Normalizing regression by `residual_std` (which MLoop has and consumers don't) is more principled and dataset-scale-stable than a `|Score|` heuristic. Separately, `mloop predict --json` emits the full structured prediction rows (same schema as the API, including `confidence`) to stdout — progress goes to stderr — routed through the same `PredictionService` as the API so the two never drift. This unblocks `mloop-mcp` from propagating structured predictions (it parsed scraped console text before) and lets middleware consume the confidence contract instead of re-implementing it. Additive and backward-compatible: the `confidence` field is nullable, and models/servers without it are handled by consumer fallbacks.

## [0.18.4] - 2026-07-03

### Fixed
- **serve `/predict` for binary-classification (and multiclass/regression)**: two cascading defects made `mloop serve` `/predict` fail for every AutoML tabular classification/regression model (the CLI `predict` path was unaffected). (1) The saved model expects a single `Features` vector input (how AutoML's `InferColumns` loads features), but `PredictionService` only built one for anomaly/clustering/time-series-anomaly → `500 "Could not find input column 'Features'"`. `RequiresFeaturesVectorInput` now also covers classification/regression/forecasting (guarded: no-op when a `Features` column already exists or no numeric features). (2) Binary models output `PredictedLabel` as a raw `Boolean`, but `ExtractClassificationRows` always read it with a String getter → `500 "Invalid TValue: ReadOnlyMemory<Char>, expected Boolean"`. The getter is now chosen by column type, and binary predictions expose both class probabilities (`{True, False}`) so max-probability confidence is correct for either class. Regression test added.
- **serve `/predict` for multiclass models with a numeric-looking class label (D14)**: found with a three-class set (class ids `"0"`/`"1"`/`"2"`). Two cascading type mismatches, the same shape as D12/D13. (1) `ResolveDataKind` unconditionally loaded a multiclass label as `String`, but when the label happens to be numeric-looking, train-time schema inference records `DataType=Numeric` and AutoML's fitted pipeline embeds `MapValueToKey` over a `Single`-typed column — the mismatch made `model.Transform` throw `500 "Could not apply a map over type 'Single' to column '<label>' since it has type 'String'"`. The multiclass/text/image-classification branch now respects the trained `col.DataType` (via the existing `MapDataTypeToDataKind`) instead of forcing `String`. (2) With that fixed, the model's own `PredictedLabel` key restores (via `MapKeyToValue`) to the same `Single` type it was trained on, but `ExtractClassificationRows` only had getters for `Boolean` and `String` → `500 "Invalid TValue: ReadOnlyMemory<Char>, expected Single"`. A `Single`-typed getter branch was added, rendering the numeric label with invariant-culture `ToString()` (e.g. `0f` → `"0"`, matching the original CSV value). Two `BuildTextLoaderOptions` unit tests plus an end-to-end `Predict_Multiclass_NumericLabel_MatchesTrainedSingleSchema` regression test (trains a real `MapValueToKey` → `LightGbmMulti` pipeline on a `Single` label) pin both halves. Live-reverified against that set (30/30 `/predict` 200, correct labels).
- **`mloop train` could silently save an uncalibrated binary model, permanently zeroing its confidence signal (D15)**: found on a binary set — AutoML chose `FastForestBinary` as the best trainer, which (unlike SDCA/LightGBM) emits a `Score` column but no `Probability`. Training already falls back to `EvaluateNonCalibrated` so metrics still compute, but the *saved* model — the exact one `mloop predict` and `mloop serve` load — stayed uncalibrated. Every downstream consumer reading `probabilities`/`Probability` for a confidence signal (HoneAI's `ConfidenceOf`, any future consumer of the API's `probabilities` field) got nothing, and confidence permanently collapsed to 0. Live impact: **100% of a 30-row review-escalation loop in a consuming service was escalated to the LLM** (every row) instead of the expected handful — the entire confidence-gated dual-check design silently degraded to "always ask the LLM" for this model. `AutoMLRunner` now appends a Platt calibrator (fit on the already-computed test-set predictions, which carry the `Score` the calibrator needs) to the returned model whenever its output lacks a `Probability` column, for both the AutoML and manual-SDCA-fallback binary paths. Two tests pin the behavior (`EnsureCalibratedModel_UncalibratedTrainer_AppendsPlattCalibratorRestoringProbability` reproduces the exact FastForestBinary shape and asserts the composed model's `Probability` is in `[0, 1]`; a passthrough test confirms an already-calibrated model is returned unchanged). Live-reverified on that set: CLI predict now exposes `Probability` (e.g. `0.9396` for a `Score` of `742.7`), and the share of predictions a consuming service sent on for review dropped from 30/30 to 0/30 with realistic confidence values (0.72–0.99).
- **Quality gate's degenerate-model detector missed the "always predicts the majority class" pattern when that majority is the positive label (D16)**: found on a binary set — a 2.9:1-imbalanced ("Slightly Imbalanced" — the analyzer's own guidance for this tier says "should handle this naturally") binary set trained (with the default `accuracy` optimization metric) to a model that predicted the majority class (`OK`) for **100% of 12,069 holdout rows**, i.e. zero recall on the minority (`NG`) class — yet `accuracy=0.747`, `f1_score=0.855`, `auc=0.772` all looked healthy and the model auto-promoted to production. `MetricPolicy.IsClassificationDegenerateModel`'s existing check (`accuracy > 0.5 && f1_score ≈ 0`) only catches the mirror case (always predicts the *negative* class) — F1 is computed on the positive class, so an always-positive model's F1 stays high (`≈ 2·prevalence/(1+prevalence)`) precisely when the majority is the positive label, which is the common real-world convention for pass/fail, OK/NG, healthy/anomaly QC data. The gate now also blocks when `accuracy > 0.5 && negative_recall ≈ 0` (`BinaryClassificationMetrics.NegativeRecall`, newly threaded into `AutoMLRunner`'s metrics dict alongside the existing `precision`/`recall`, for both the calibrated and `EvaluateNonCalibrated` paths, and both the AutoML and manual-SDCA-fallback binary trainers). Three tests pin the new branch (the exact metric shape, a healthy control, and a genuinely-bad-but-not-degenerate control). Live-reverified: retraining that set with the fix reproduced the identical degenerate metrics (`negative_recall=0.0000`, now visible in `mloop train`'s own metrics table) but auto-promotion was correctly blocked (`Model saved to staging`, no "promoted to production" — contrast with the pre-fix run, which promoted).

## [0.18.3] - 2026-07-01

### Added
- **`mloop token` — issue a JWT bearer token for the local serve API (D7)**: `mloop serve` requires authentication on every endpoint except `/health`, but there was no way anywhere (API, CLI, docs, tests) to *obtain* a token — so a freshly served model was unreachable and any legitimate client (e.g. an in-house consumer calling `/predict`) was blocked. The new `mloop token` command issues an HS256 JWT (`--role` for admin/write endpoints, `--subject`, `--expires-hours`, `--key` override, `--quiet` for scripting). It signs with a new single-authority `MLoop.Core.Security.DevJwtDefaults` (key/issuer/audience) that the API now also validates against, so issued tokens validate out of the box and the issuer↔validator can't drift. The API reads claims verbatim (`MapInboundClaims=false`, `NameClaimType="sub"`, `RoleClaimType="role"`) so the short `sub`/`role` claims authorize correctly (admin role included). Not for production auth — a local-serving convenience; production sets its own `Jwt:Key`. (A consuming service could not call the served model at all.)

### Fixed
- **`mloop train` misdetected a sorted binary/multiclass label as single-class when the minority class first appeared past row 1000 (D2)**: two label-inference sites in `TrainingEngine` sampled only the first 1000 data rows. `CollectDataStats` counted label classes inside a `rowCount <= 1000` guard, and `CollectCompleteCategoricalValues` — the full-file scan added for ordered data (BUG-R2-06) — filtered to `Purpose == "Feature"`, so it *skipped the label column*. On manufacturing data sorted OK-first with the NG tail beyond row 1000, the schema recorded `label: CategoricalValues=["OK"], UniqueValueCount=1` and `classCount=1`; the promotion quality gate derives its 1/N accuracy threshold from the label's `UniqueValueCount`, so a single-class reading corrupts the gate (anomaly is unaffected — unsupervised). Class counting now scans the full file, and the complete-values scan now includes the label column (regression labels are `Numeric`, not `Categorical`, so they remain excluded); text-feature detection stays sampled (a heuristic). Three regression tests pin full-file class counting and label/feature completion past the 1000-row boundary. (Contrast with DataLens's full-column scan.)
- **`mloop promote` wrote `bestTrainer: null` into `production/metadata.json` and `registry.json`, losing the trainer identity (D5)**: `ExperimentStore.SaveAsync` persists a `result` block (`bestTrainer` + `trainingTimeSeconds`) into the staging `metadata.json`, but `ExperimentStore.LoadAsync` never reconstructed it — it built `ExperimentData` with `Result` left null. So `ModelRegistry.PromoteAsync`, which reads `experiment.Result?.BestTrainer` for both the production metadata and the registry entry, always recorded null (observed for anomaly's `RandomizedPca (rank=2)`), even though the staging metadata on disk had the correct value — a writer↔reader round-trip drift. `LoadAsync` now reconstructs the result via a `ParseResult` helper (an in-progress/failed experiment with no result block, or `"result": null`, keeps `Result` null — no fabricated trainer). Two round-trip tests pin trainer preservation and the null-result case; a live promote confirmed production/registry now carry the real trainer.
- **`mloop serve` loaded a stale Debug build of the API in development, hiding fixes built into Release (D9)**: `ServeCommand.FindApiAssembly` collected the dev-build `MLoop.API.dll` candidates in `{ "Debug", "Release" }` order and returned the *first existing* one, so whenever both configurations were built, the Debug assembly always won — even a freshly rebuilt Release was ignored. During live testing this loaded a stale Debug API that lacked the D8 fix, forcing a manual `MLOOP_API_PATH=Release` workaround. Resolution now prefers the *most recently built* dev assembly (newest last-write-time wins), so a fresh Release build is picked automatically; the `MLOOP_API_PATH` override and the fixed deployment-location paths keep their existing precedence (only the ambiguous dev-build search changed). The selection policy is extracted to a pure, unit-tested `ServeCommand.NewestExisting` (filesystem access injected) with four tests pinning newest-wins in both directions, missing-candidate skipping, and the none-exist case.
- **REST `/predict` returned 500 for anomaly-detection (and clustering / time-series-anomaly) models — "Could not find input column 'Features'" (D8)**: these tasks train via a manual `Concatenate("Features", featureCols) → trainer(featureColumnName: "Features")` pipeline, so the saved model expects a single `Features` vector input. The CLI predict path gets this for free (its TextLoader/InferColumns produces a `Features` vector), but the shared `PredictionService` (used by `mloop serve`) loads the schema's individual named columns and so the model could not find `Features`, failing every served anomaly/clustering prediction with a 500. `PredictionService` now concatenates the numeric (non-label) feature columns into `Features` before `Transform` for exactly these manual-pipeline tasks (`RequiresFeaturesVectorInput`), matching the single-vector shape they were trained on; AutoML tasks (binary/multiclass/regression), whose models featurize named columns internally, are untouched. The selection is driven by `DataType`/label (not `Purpose`, which can be absent when the schema is deserialized from `config.json`), mirroring how anomaly/clustering pick feature columns at train time. A regression test trains a real RandomizedPca model expecting a `Features` input and asserts `PredictionService.Predict` succeeds from individual columns. Seen when serving an anomaly model (`/predict` answered 500). This is a concrete instance of the PredictionEngine↔PredictionService divergence.
- **`mloop serve` exited immediately out of the box — default Production environment rejected the default JWT key (D6)**: `ServeCommand` launched the child `MLoop.API` process with no environment set, so it defaulted to Production, where the startup guard (correctly) refuses the default JWT key and throws `Cannot start in Production with default JWT key` — making `mloop serve` unusable without manual `ASPNETCORE_ENVIRONMENT`/`Jwt:Key` configuration. `serve` (a localhost serving convenience) now defaults the child API to the Development environment when `ASPNETCORE_ENVIRONMENT` is unset, where the default dev key is allowed with a warning; production deployments still set the environment and a real key themselves. Seen when `serve` died on launch. The default key/issuer/audience are now single-sourced in `MLoop.Core.Security.DevJwtDefaults` (shared by the API validator and `mloop token`).
- **`mloop promote` silently skipped backing up the current production model, risking irrecoverable loss of the previous model**: the CLI `promote` command (and the REST `/promote` endpoint), when not given `--no-backup`, called `FilePromotionManager.BackupProductionAsync` to preserve the outgoing production model before overwriting it. But that method read the production experiment id from `model-registry.json` — a file written only by `FilePromotionManager.UpdateRegistryAsync`, which was unwired dead code (CLI/REST promote actually persist production state through `ModelRegistry` → `registry.json`). So `model-registry.json` never existed on a real project, the id lookup returned null, and the backup was silently skipped; `ModelRegistry.PromoteAsync` then overwrote `production/` with no backup, permanently losing the prior model. The backup now triggers off the production *directory's* presence (not a registry file) and reads the experiment id from the production model's own `production/metadata.json` (the artifact `ModelRegistry` actually writes on promote), falling back to a timestamp-labelled backup when metadata is absent — so an existing production model is always preserved. The dead promotion/comparison methods that hardcoded `model-registry.json` (`IModelComparer.CompareWithProductionAsync`/`FindBestExperimentAsync`, `IPromotionManager.EvaluatePromotionAsync`/`PromoteAsync`/`RollbackAsync`/`GetHistoryAsync`) are removed; `model-registry.json` no longer appears anywhere in code. Regression tests pin that backup triggers off the production directory and is labelled with the metadata experiment id. Found via the registry-name reconciliation.

### Changed
- **Extracted the CLI promotion-gate metric policy out of `ModelRegistry` into a dedicated `MetricPolicy` (pure refactoring, no public API change)**: `ModelRegistry` — whose responsibility is filesystem production deployment (`PromoteAsync`, `GetProductionAsync`, `registry.json` I/O) — also hosted four `public static` metric-policy helpers (`ResolveMetricKey`, `ResolveCanonicalMetricKey`, `GetMinimumMetricThreshold`, `IsClassificationDegenerateModel`), and `CompareCommand`/`TrainCommand` reached into the deployment registry purely to resolve metric keys and thresholds — an SRP leak (a command sorting experiments by metric had to depend on the production registry). The helpers move verbatim to a new `MLoop.CLI.Infrastructure.ML.MetricPolicy`, beside the other ML infrastructure; `ModelRegistry.ShouldPromoteAsync` and the two commands now consume `MetricPolicy`. The split keeps this knowledge CLI-side by design (it is promotion-gate *policy* — alias spellings, 1/N-baseline floors, the degenerate-classifier heuristic — not the shared domain facts in `TaskMetadata`/`MetricDirection`, which it composes). A dead private `ResolveMetricKey(string, Dictionary)` overload with no caller is removed. Behavior is byte-for-byte identical; the metric-policy unit tests (alias resolution, thresholds, degenerate detection) relocate from `ModelRegistryTests` to a new `MetricPolicyTests` mirroring the SUT, while the `ShouldPromoteAsync` promotion tests stay (they exercise the policy through the gate). Closes the CLI metric-policy-extraction proposal (roadmap §4.1).
- **Single-sourced the production registry filename so the writer and reader cannot drift (no behavior change)**: `ModelRegistry` (writes `registry.json` on promote) and `ModelNameResolver` (reads it back when listing models) each hardcoded `RegistryFileName = "registry.json"` independently — the same writer↔reader filename-drift shape that produced the dead path. Both now delegate to a new `ExperimentLayout.RegistryFileName`, and the stray `model.zip`/`config.json` literals in `PredictCommand` are likewise pointed at the `ExperimentLayout` authority. Closes the registry-name divergence noted in the 0.18.2 layout-consolidation entry.

## [0.18.2] - 2026-06-29

### Changed
- **Converged ML-metric knowledge and the experiment filesystem layout into single `MLoop.Core` authorities (pure refactoring, no public API change)**: two classes of cross-cutting constants had each been hardcoded across the CLI/Ops/API assembly boundary, the exact drift shape behind the metric-direction bug and the `experiments/` vs `staging/` mistake. (1) `TaskMetadata` (task→canonical metric) moved from `MLoop.CLI` to `MLoop.Core.Evaluation` beside `MetricDirection`, so the metric *name* and *direction* halves share one home that both CLI and Ops consume; the promotion-gate threshold/alias helpers stay in CLI (single-consumer policy, not domain facts). (2) A new `MLoop.Core.Storage.ExperimentLayout` is the one authority for the on-disk layout (`models`/`staging`/`production`/`model.zip`/`metadata.json`/`metrics.json`/`config.json`/`experiment-index.json`); the interim `OpsStorage` is removed, and every writer (`ExperimentStore`, `InitCommand`, `ModelRegistry`) and reader (the three Ops services, `predict`/`evaluate`, the API, the model-cache preloader) now references it — the `model.zip` literal alone had been duplicated across 8+ sites. Behavior is identical (all converged values are byte-for-byte the same); the change makes the cross-assembly drift class unrepresentable. New `ExperimentLayoutTests` pin the canonical values. (The Ops `model-registry.json` read path remains a known divergence from the authoritative `registry.json` — unwired dead code, tracked separately.)

### Fixed
- **`mloop evaluate` mis-loaded `String`-typed features and comma-bearing quoted fields — a silent drift from `mloop predict`**: the inference schema reconciliation that runs after `InferColumns` (override each feature's type from the trained schema, enable RFC 4180 quoting, split preserved group/user/item columns out of the merged Features range) was reimplemented independently in `PredictionEngine` and `EvaluationEngine`. Two fixes had landed on the predict path but never reached evaluate: the `"String"` raw-type-name case (so evaluate could not restore a feature training fitted as String) and `AllowQuoting = true` (so a test CSV with `"[1, 2, 3]"`-style quoted fields split mid-row and the feature-vector width drifted from training). This is the same "fix one inference path, the other drifts" shape. Both engines now call one shared `CsvDataLoader.ReconcileInferredSchemaForInference`, with the trained-type→`DataKind` mapping further extracted to `CsvDataLoader.MapTrainedTypeToDataKind` so predict's label-type alignment (which evaluate deliberately omits) shares the same table. The label stays each engine's own responsibility — predict aligns it to the schema and converts Boolean→String, evaluate applies task-specific (multiclass→String / regression→Single / binary string→bool) handling — a legitimate divergence the shared helper leaves untouched. New unit tests pin the helper contract (String override, quoting, label-untouched, preserved-column split); the existing evaluate/predict task-family integration tests guard against regression. Found by the structural review of the drift.
- **`mloop docker` generated a Dockerfile that failed `docker build`**: the generated Dockerfile ran `dotnet tool install --global mloop --version *`, but in a Dockerfile `RUN` the shell glob-expands the unquoted `*` to the working-directory file list (`mloop.yaml .mloop …`), so the build aborted with an invalid `--version` argument. The generator now emits `dotnet tool install --global mloop` (latest stable), which is both correct and the simplest form. A regression test asserts the generated Dockerfile contains no `--version *`.
- **`mloop sample create --model` crashed when any prediction had feedback recorded more than once**: `FileDataSampler` joined predictions to feedback with `feedback.ToDictionary(f => f.PredictionId, …)`, which throws `ArgumentException: An item with the same key has already been added` if two feedback entries share a prediction id. Feedback is append-only (`RecordFeedbackAsync` adds a new entry each call), so a user correcting a label — recording feedback twice for the same prediction — made `sample create --model` (the retraining-dataset builder) abort outright. The join now groups by prediction id and keeps the most recently recorded value (feedback entries now carry their timestamp through the load step). A regression test records duplicate feedback and asserts sampling succeeds and uses the corrected value.
- **The whole MLoop.Ops layer read experiments from a non-existent `experiments/` directory instead of `staging/`**: `FileModelComparer` and `FilePromotionManager` (alongside `TimeBasedTrigger`) each hardcoded `models/{name}/experiments/{expId}/...`, but MLoop stores experiments under `models/{name}/staging/{expId}/` (`ExperimentStore`). The user-facing consequence: the REST `/compare` endpoint (which calls `FileModelComparer.CompareAsync` → reads `experiments/{expId}/metrics.json`) threw `FileNotFoundException` on every real project, since the metrics actually live under `staging/`. `FilePromotionManager`'s experiment-reading methods (`PromoteAsync`/`EvaluatePromotionAsync`/`LoadMetricsAsync`) had the same mistake (currently reachable only via its own contract — the API `/promote` promotes through `IModelRegistry`, which was already correct — but fixed for correctness). The three services' duplicated literals are converged into a shared `OpsStorage` constant (`StagingDirectory`/`MetadataFileName`/`MetricsFileName`) so they cannot drift again. The Ops test fixtures, which had fabricated the non-existent `experiments/` layout (so the tests passed while the code was broken), now write the real `staging/` layout and thus validate against reality.
- **The REST `/trigger` endpoint always reported "no training history → retrain" because `TimeBasedTrigger` read the wrong experiment path**: it looked for training history under `models/{name}/experiments/{expId}/experiment.json`, but MLoop stores experiments under `models/{name}/staging/{expId}/metadata.json` (the documented MLOps layout — `ExperimentStore`). That directory never exists on a real project, so `GetLastTrainingTimeAsync` always returned null and the time-based trigger unconditionally recommended retraining (`"No training history found"`) even moments after a successful train. The unit tests passed only because their fixture fabricated the non-existent `experiments/.../experiment.json` layout. `TimeBasedTrigger` now reads `staging/{expId}/metadata.json`, and the test fixture writes the real layout so it validates against reality (a recent completed experiment now correctly yields *no* retrain). (`FileModelComparer` and `FilePromotionManager` shared the same `experiments/` mistake — fixed together above.)
- **`mloop feedback metrics` reported 0% accuracy (and `feedback list` showed all ✗) for regression and numeric-label models**: a prediction's output is logged as a JSON number when the model emits a number (regression, numeric class labels), but the CLI's `feedback add --actual-value` is always a string. The comparer matched two numbers or two strings but had no cross-type case, so a predicted `1` never matched an actual `"1"` — every numeric-output model's feedback accuracy collapsed to 0 regardless of correctness. `FeedbackValueComparer.ValuesMatch` now compares numerically (with tolerance) whenever both sides represent a number, including the number-vs-numeric-string case, and falls back to a culture-invariant textual comparison (which also lets a boolean prediction match an actual `"true"`). Pure string labels still compare case-insensitively and stay distinct (`"01"` ≠ `"1"`). A new end-to-end `CalculateMetricsAsync` test logs numeric predictions and string actuals and pins the accuracy; `FeedbackValueComparerTests` cover the cross-type cases (a reproduction confirmed accuracy was 0.0 before the fix).
- **`mloop logs`/`mloop feedback list`/`mloop feedback metrics` date filters (`--from`/`--to`) were off by the machine's UTC offset**: each command converted the user's date with `new DateTimeOffset(date, TimeSpan.Zero)` — forcing the typed day to UTC midnight — while timestamps are stored in UTC but *displayed* in local time. So in a non-UTC timezone the filter window was shifted: in KST (UTC+9) `--from 2026-06-29` actually started at 2026-06-29T09:00 local, silently excluding the first nine hours of the requested local day and leaking in the tail of the previous one — contradicting the local-time column the user reads. The three call sites each had their own copy of the conversion (drift risk). They now share one `DateRangeFilter.ToFilterRange` helper that interprets a bare date as the user's *local* calendar day (`--to` covers the whole day), matching the display. New `DateRangeFilterTests` assert the local offset (machine-timezone-independent). The REST `/logs` and `/feedback` endpoints are unaffected — they use `DateTimeOffset.TryParse`, which already honors explicit offsets and treats a bare date as local.
- **`mloop feedback list`'s "Match" column always showed ✗, contradicting `mloop feedback metrics` accuracy**: the predicted-vs-actual comparison existed in two drifted copies — `FileFeedbackCollector.ValuesMatch` (used by accuracy, *correct*: normalizes `JsonElement`, numeric tolerance, case-insensitive labels) and `FeedbackCommand.ValuesMatch` (used by the list table, *naive*: no `JsonElement` handling). Feedback values round-trip through JSONL and System.Text.Json materializes `object` as `JsonElement`, so at runtime the list compared two `JsonElement` structs via `object.Equals` — never equal across documents — and every row rendered a red ✗ even when the metrics command reported 100% accuracy off the very same entries. The unit tests missed it because they only ever passed CLR primitives (`42`, `"OK"`), never the `JsonElement` the code actually receives. Fixed by converging both call sites on one public `FeedbackValueComparer.ValuesMatch` in MLoop.DataStore (the feedback domain owner), mirroring the proven-correct collector logic; the naive CLI copy and the duplicate private collector copy are deleted. New `FeedbackValueComparerTests` exercise the real `JsonElement`-sourced types (a reproduction confirmed the old CLI copy returned `false` for equal `"OK"`/`"OK"` and `42`/`42` elements).
- **`ExperimentSummary.BestMetric` was the insertion-order-first stored metric, not the metric the experiment optimized**: `ExperimentStore` recorded `BestMetric = Metrics.Values.FirstOrDefault()`, so the reported score depended on dictionary insertion order rather than the experiment's canonical primary metric. It happened to be correct only because the primary metric was usually stored first; for any experiment whose metrics dictionary led with a different metric (e.g. a clustering run storing `davies_bouldin_index` before `average_distance`), `BestMetric` disagreed with its own `MetricName` — so `ExperimentRanking`, which trusts `MetricName` for direction but ranks by `BestMetric`'s magnitude, compared apples to oranges. `BestMetric` is now resolved by a shared `TaskMetadata.ResolvePrimaryMetricValue(metrics, configuredMetric, task)` helper — the configured metric if present, else the task's canonical primary (`TaskMetadata` already owned the metric *names*; it now owns *value* resolution too), else the first available — so the stored score always matches `MetricName`. `TrainingEngine.GetPrimaryMetricValue` (the probe-vs-main comparison) now delegates to the same helper, so the two can never diverge. No schema change (value computation only); existing experiment indexes are unaffected (the fix applies on next save). New `TaskMetadataTests` pin the resolver and a new `ExperimentStoreTests` case pins that a non-first canonical metric is selected. Closes the follow-up noted below.
- **`mloop status` and `mloop compare` (and the REST status endpoint) reported the worst experiment as "best" for lower-is-better tasks**: the best-experiment selection sorted `BestMetric` *descending unconditionally* in three places (StatusCommand, CompareCommand, the API `/status` handler), so for clustering (`average_distance`), forecasting (`mae`), and recommendation (`rmse`) — whose canonical metric is lower-is-better — it picked the experiment with the *highest* (worst) metric as best, contradicting the promotion gate. `ExperimentSummary` already carried `MetricName`, so the fix is a shared `ExperimentRanking` helper that orders by quality using that name plus the `MetricDirection` single source of truth; all three callers delegate to it. (The companion `BestMetric` canonical-value fix above closes the remaining gap where `BestMetric` could disagree with `MetricName`.) New `ExperimentRankingTests` pin lower- and higher-is-better selection.
- **`mloop compare` and `mloop evaluate` mis-ranked clustering models — the worse model was reported as "best"**: a metric's optimization direction (lower-is-better for errors/distances, higher-is-better for accuracy/auc/r²/ndcg) was hardcoded in four places — `ModelRegistry.IsErrorMetric` (the only complete copy), `EvaluateCommand.IsLowerBetterMetric`, `CompareCommand` (three inline copies), and `FileModelComparer.IsLowerBetterMetric` — and the latter three had drifted: they recognized only `rmse`/`mae`/`mse`/`loss` and were missing clustering's `average_distance` (clustering's *canonical* metric) and `davies_bouldin_index`, plus `mape`. So `mloop compare` sorting clustering experiments by `average_distance` ranked them ascending-is-worse — picking the model with the *largest* average distance (the worst clustering) as best — and evaluate's train-vs-test diff coloring was inverted for those metrics. The promotion gate (`ModelRegistry`) was already correct, so compare/evaluate even contradicted what would actually be promoted. The direction logic is now a single source of truth, `MLoop.Core.Evaluation.MetricDirection.IsLowerBetter`, that all four sites delegate to (lives in Core so MLoop.CLI and MLoop.Ops share it — the same convergence done for task→metric). A drifted test that asserted `root_mean_squared_error` was *not* lower-better (it contains "error" → it is) was corrected to match. New `MetricDirectionTests` pin every direction.
- **`mloop evaluate` crashed outright on ranking and recommendation models with "Could not find input column"**: a ranking model's pipeline references the group column individually (`MapValueToKey("GroupId", group_column)`) and a recommendation model maps the user/item columns to keys, so those columns must stay individually addressable at evaluate time. But `EvaluationEngine.LoadTestData` ran `InferColumns` *without* the `ApplyColumnPreservation` step that the train (`CsvDataLoader`) and predict (`PredictionEngine`) paths apply — so `InferColumns` merged the standalone group/user/item columns into the Features range, the column vanished, and `model.Transform` threw before evaluation even started. This is the exact evaluate-path twin of the predict-path fix, which added the shared `CsvDataLoader.ApplyColumnPreservation` helper to train and predict but left evaluate behind. `EvaluateAsync` now threads the trained group/user/item columns through `LoadTestData`, which applies the same preservation, so ranking and recommendation evaluation works end-to-end (the CLI and the REST `/evaluate` endpoint both pass all three columns). A new integration test (`EvaluateAsync_Ranking_PreservesGroupColumn_ReturnsNdcg`) builds a ranking model on CsvDataLoader-loaded data (so the column structure matches evaluate's) and pins that NDCG metrics come back.
- **`mloop evaluate`'s overfitting warning never fired for classification models**: `EvaluateCommand.DetectOverfitting` matched the literal task string `"classification"` with the metric key `"accuracy"`, but trained models store the CLI-canonical task string — `"binary-classification"` or `"multiclass-classification"` (saved verbatim from `--task`, no normalization). So the `task == "classification"` branch was never taken for any real model, and the "Model may be overfitting" warning was silently dead for every classification evaluation — doubly so for multiclass, whose primary metric key is `"macro_accuracy"`, not `"accuracy"`. The check now resolves the metric key from the actual stored task (regression→`r_squared`, binary→`accuracy`, multiclass→`macro_accuracy`, with `"classification"` kept as a legacy binary alias) and compares train-vs-test on that key, so the warning works for binary and multiclass as it already did for regression. Three regression tests pin the binary/multiclass cases the prior `"classification"`-only tests had missed.
- **`mloop evaluate` always reported Davies-Bouldin Index 0 for clustering models**: `EvaluationEngine.EvaluateClustering` called `Clustering.Evaluate` with only `scoreColumnName: "Score"` and no `featureColumnName` — and ML.NET only computes the Davies-Bouldin Index when the feature column is supplied. So `mloop evaluate` on any clustering model reported a constant `davies_bouldin_index 0.0000`, giving the user no real cluster-quality signal. This is the exact evaluate-path twin of the predict-path fix, which fixed the identical omission in `AutoMLRunner`'s K-search (where the constant-0 DBI had silently disabled the optimal-K selection). The clustering pipeline concatenates features into `"Features"` (same as training), so passing `featureColumnName: "Features"` restores real DBI computation. Verified on synthetic 3-cluster data: DBI now reports a finite positive value (~0.3) and `cluster_count` is 3; a new regression test (`EvaluateAsync_Clustering_ComputesNonZeroDaviesBouldinIndex`) pins it.

## [0.18.1] - 2026-06-28

### Changed
- **Centralized the task→primary-metric mapping into a single `TaskMetadata` source of truth**: the canonical metric each ML.NET task optimizes was duplicated across three drifting switch statements — `ModelRegistry.DefaultMetricForTask` (promotion gate), `InitCommand`'s yaml template (`metricExample`), and `TrainingEngine.GetPrimaryMetricValue` (probe-vs-main selection) — plus a hand-maintained allowlist in `ValidateCommand.ValidMetrics`. That duplication let two defects through: init wrote `auto` for a task with a canonical metric, so the gate silently skipped, and validate's allowlist fell out of sync with init. They now all read from one `TaskMetadata` table; adding a task in one place keeps every consumer consistent. Converging the table also resolved latent drift in the more-correct direction: `init` now writes `metric: micro_accuracy` for `text-classification` (was `auto` — the same shape), the promotion gate now resolves a canonical metric for `clustering`/`ranking`/`forecasting`/`recommendation`/`time-series-anomaly` so production comparison uses the right metric (blocking is unchanged — those are error/threshold-less metrics), and `validate` accepts every task's canonical metric by deriving the allowlist from `TaskMetadata.AllPrimaryMetrics` so it can never drift again. Pure refactoring (no public API change); 28 new tests pin the converged behavior. (Track 1 tech-debt; roadmap §4 D4.)
- **Upgraded DataLens 0.13.0 → 0.13.1 — restores predictive feature importance for numeric targets**: DataLens 0.13.0's `FeatureAnalyzer` left the target column inside the permutation-importance feature matrix, so a model predicted the target from itself and zeroed every real feature's importance — forcing `mloop analyze importance` onto its honest structural fallback (`method=structural`) for numeric targets. The consumer-side guard kept the output non-misleading, but the true predictive signal was unavailable. DataLens 0.13.1 excludes the target from the matrix (fixed in-session per the local-submodule fast-dev-cycle), so numeric-target importance now reports `method=permutation`. Verified end-to-end on a numeric-target manufacturing set (target `M_X`): `method` flips `structural`→`permutation`, `KWh` correctly ranks ≈0 (matching the finding that dropping it left R² unchanged), 총도금시간/평균온도 rank top. The degenerate-fallback guard is retained as defense-in-depth. (.)
- **`mloop analyze <aspect>` data-file argument is now optional, defaulting to the project's configured train data**: every analyze subcommand required an explicit `<data-file>` positional, while the rest of the CLI (`train`, `info`) defaults to the project's `data.train` (else `datasets/train.csv` discovery) when omitted. The inconsistency was agent-hostile: in a live agent feature-engineering run the agent — instructed to always pass the project path — put the *project directory* in `dataFile` and burned five tool calls recovering before it found `datasets/train.csv`. `analyze` now reuses `TrainDataValidator.ResolveDataFileAsync` (the same resolver `train` uses), so `mloop analyze profile` with no path analyzes the project's train data; an explicit path still overrides. The MCP `mloop_analyze` bridge mirrors this — `dataFile` is no longer a required parameter (pass `projectPath` so the default resolves). Read-only contract unchanged.

### Fixed
- **`mloop predict` failed on ranking and recommendation models with "Could not find input column"**: a ranking model's pipeline starts with `MapValueToKey("GroupId", group_column)` and a recommendation model maps the user/item columns to keys, so those columns must stay individually addressable at predict time. But `InferColumns` (used by the predict loader) merges adjacent numeric columns into one ranged Features vector, so the standalone `query_id` / `user_id` / `item_id` columns vanished and `model.Transform` threw — every ranking and recommendation model's train→predict slice was broken. The training loader already guards against this by splitting preserved columns (group/user/item, preFeaturizer columns) back out of the merged range; that logic is now a shared `CsvDataLoader.ApplyColumnPreservation` helper, and the predict path calls it too — `PredictCommand` reads the trained group/user/item columns from the experiment config and threads them through `PredictionEngine`. The user/item columns are now also persisted in `ExperimentConfig` (the group column already was). Classification/clustering predict are unaffected (they pass no preserve columns). Verified end-to-end: ranking predict now emits per-row scores and recommendation predict emits per (user, item) rating scores.
- **Clustering's optimal-K auto-search was silently inert — it always picked the largest K, and reported Davies-Bouldin Index 0**: the K-search trains KMeans for `k=2..maxK` and is meant to select the K with the lowest Davies-Bouldin Index (which, unlike average-distance, balances cluster separation and cohesion to find the *natural* K). But `Clustering.Evaluate` was called with only `scoreColumnName: "Score"` and no `featureColumnName` — and ML.NET only computes the Davies-Bouldin Index when the feature column is supplied. So DBI came back a constant `0`, the `useDbi = dbi > 0` branch was never taken, and the search fell back to average-distance, which decreases monotonically with K and therefore *always* selected the search ceiling (e.g. k=10 for 3 obvious clusters). Users silently got an over-segmented model and a meaningless `DAVIES BOULDIN INDEX 0.0000` in the metrics. Passing `featureColumnName: "Features"` to `Clustering.Evaluate` restores real DBI computation and thus the intended K-search. Verified on synthetic 3-cluster data: `num_clusters` now selects 3 (was 10) and DBI reports a real value (~0.26); a new integration test (`ClusteringKSearchRegressionTests`) pins both.
- **`mloop predict` crashed on clustering models with "Metadata KeyValues does not exist"**: the predict path restores original class names by applying `MapKeyToValue` to a key-typed `PredictedLabel` — correct for classification (whose key, produced by `MapValueToKey`, carries a `KeyValues` annotation mapping ids back to label strings). But clustering's KMeans also emits `PredictedLabel` as a key (the cluster id 1..k) with **no** `KeyValues` annotation, so `MapKeyToValue` threw and `mloop predict` failed outright for every clustering model — the train→promote→predict slice was broken end-to-end. The guard now applies `MapKeyToValue` only when the column actually has a `KeyValues` annotation (new shared `PredictionService.HasKeyValues`). The exact same restore logic was duplicated in both predict paths — `PredictionService` (Core) and the CLI's `PredictionEngine` (what `mloop predict` runs) — and both had the bug; both now call the one shared guard so they cannot drift. Verified end-to-end: clustering predict now emits cluster ids, multiclass still restores class names (`High`/`Low`), binary is unaffected (its `PredictedLabel` is a boolean, never a key).
- **`mloop validate` did not check task-specific required fields that `train` mandates; a parallel orphan validator that did was dead and drifted**: the live `ValidateCommand` (what `mloop validate` runs) never validated that a `forecasting` model has a `horizon`, a `ranking` model has a `group_column`, or a `recommendation` model has `user_column`/`item_column` — all required at train time (`AutoMLRunner` throws without them), so a project missing them passed `validate` and then failed `train`. A *second* config validator, `ConfigValidator.Validate(MLoopConfig)` ("extracted from ValidateCommand for testability"), did contain these checks — but nothing in production ever called it; only its own tests did. Kept alive solely by tests, it had drifted from the live path: its task allowlist had lost the three NLP tasks (`sentence-similarity`/`ner`/`question-answering`), its unsupervised set had lost `time-series-anomaly`, and it still carried the label bug. The unique required-field checks were ported into the live `ValidateCommand` (with tests), and the orphan `Validate`/`ValidateModel`/`ValidateLabelInCsv`/`ValidTaskTypes` were removed — only the genuinely live `ConfigValidator.ValidatePrepSteps` (used by `PrepRunCommand`) remains. `init` writes a default for each field (`horizon: 10`, `group_column: query_id`, `user_column`/`item_column`), so a freshly-initialized project still validates clean; the new errors only fire when a field is removed. Found via the task-classification duplication audit that the fix motivated.
- **`mloop validate` errored "Label column is required" on valid unsupervised projects, and `mloop train` rejected label-less `time-series-anomaly` — a drifted task-set duplication**: the unsupervised, label-optional task set (`anomaly-detection`, `clustering`, `time-series-anomaly` — `CsvDataLoader` loads a dummy label when none is given so every column becomes a feature) was hardcoded as three separate inline `HashSet`s in `ConfigMerger`, `InitCommand`, and `TrainCommand`, and the sets had **already drifted**: `TrainCommand`'s copy omitted `time-series-anomaly`, so a label-less ts-anomaly project that `init`/merge accept failed `train`'s required-field guard. `ValidateCommand` had no notion of the concept at all and unconditionally required a label, so `mloop validate` reported a hard error on a perfectly valid label-less clustering/anomaly project that `mloop train` runs fine — the same train↔validate contradiction shape. All four now read one source of truth, `AutoMLRunner.RequiresLabel(task)` (sibling to `SupportsPreFeaturizer`/`IsTimeSeriesTask`; unknown tasks conservatively require a label), removing the three inline sets and both drift bugs. Supervised tasks (regression/binary/multiclass/forecasting/ranking/recommendation) still require a label.
- **`mloop validate` did not surface that `test_split` is silently ignored for time-series tasks**: forecasting and `time-series-anomaly` feed the full series to training and hold out the last `horizon` rows internally (a random split would break temporal order — `AutoMLRunner` ignores `config.TestSplit` for these tasks), yet `mloop init` writes a uniform `test_split: 0.2` into every project's yaml and `validate` reported "Training settings ✓" without comment. A user configuring forecasting reasonably expects a 20% holdout but gets `horizon` rows (e.g. 10 of 365) — a silent no-op the project's "no silent failures" principle forbids. `validate` now emits an informational warning for time-series tasks that `test_split` has no effect (they use a temporal `horizon` holdout instead), while still erroring on an out-of-range value (a real typo) and leaving tabular-task behavior unchanged. The `forecasting | time-series-anomaly` classification is centralized as `AutoMLRunner.IsTimeSeriesTask` (sibling to `SupportsPreFeaturizer`) and consumed by both the runner's split branch and `validate`, so the two cannot drift.
- **`mloop validate` flagged its own task-default metrics as "Unknown"**: `ValidateCommand.ValidMetrics` listed only user-facing aliases (`accuracy`, `auc`, `f1`, `r2`, `rmse`, …) and omitted the canonical metrics that `mloop init` writes and that AutoML / the promotion gate actually use — so `mloop validate` warned `Unknown metric 'macro_accuracy'` on a freshly `init`-ed multiclass project (and likewise `micro_accuracy` for image/text, `r_squared` for regression). `ModelRegistry.DefaultMetricForTask` returns exactly these (`multiclass→macro_accuracy`, `image/text→micro_accuracy`, `regression→r_squared`), so validate contradicted init/train — a false warning that, in a live agent feature-engineering run, the agent repeatedly dismissed as "a pre-existing config problem." `ValidMetrics` now also includes the canonical forms (`macro_accuracy`, `micro_accuracy`, `r_squared`, `log_loss`, `f1_score`); a Theory test pins every task-default metric as accepted so the two lists cannot silently diverge again. (The deeper fix — centralizing task→metric knowledge into one `TaskMetadata` source so the allowlist *derives* from it — was since done, see Changed above.)
- **`mloop analyze profile` did not surface ID/index columns that `train` already warns about**: `train` detects strictly-increasing integer columns (`CsvDataLoader.DetectMonotonicColumns`) and warns they are likely ID/index features causing leakage/overfitting — but `analyze profile`, the agent's "eye," was silent: its envelope reported only `uniqueCount`, and `importance` ranks such a column by mutual-info (a *keep* signal), so an LLM FE-loop agent had no basis to apply its "drop ID/index" rule. In a live agent multiclass run the agent received a clean EDA yet judged "no ID/index columns," missing `ncWorkCount` (uniqueCount = row-count, values 1..N). `MapProfile` now accepts the monotonic-column list (reusing the exact detector `train` uses, so the two stay consistent) and emits a `likely-index: <col> (strictly increasing integers; exclude to avoid leakage)` flag plus a `likely-index` count in the summary; the console renders it via the shared `env.Flags` path. Additive — `monotonicColumns` defaults to none, so callers and the JSON shape are unchanged otherwise. Verified E2E on that set: `1 likely-index` / `likely-index: ncWorkCount`.
- **`mloop features select --drop` silently recorded non-existent column names**: `--drop` passed its column list straight to `FeatureSelector.Drop`, which added an `ignore` override for *any* name — including a column that isn't in the data. So a typo, or an LLM agent's hallucinated column name, produced a *valid-looking but wrong* policy: the phantom override is a no-op at train time while the column the user/agent actually meant stays a feature. (`--keep` was already safe — it computes the complement against the real header.) Observed in a live agent binary feature-engineering run: the agent dropped `검사항기`/`검사항목` when the real columns were `검사호기`/`검사모드`, and the tool accepted both. `features select --drop` now validates requested columns against the train-data header (new pure `FeatureSelector.PartitionByHeader`, case-sensitive to match train-time override-key lookup): unknown columns are skipped with a warning (surfaced in the console and in the `--json` envelope's `warnings`), and only real columns are recorded. Best-effort — when no train data is resolvable, the prior unconditional behaviour is kept.
- **Experiment `metadata.json` recorded a hardcoded, stale MLoop version**: `ExperimentStore.SaveAsync` wrote `versions.mLoop = "0.2.0"` for every experiment regardless of the actual tool version (currently 0.18.0), so the experiment record — meant to make a run reproducible — misreported the very tool that produced it. It now records the real version via `UpdateChecker.GetCurrentVersion()` (the assembly informational version with the `+commitHash` suffix stripped, the same source `mloop --version` uses). The `mlNet = "5.0.0"` entry is accurate and unchanged.
- **`mloop analyze distribution` flagged low-cardinality/categorical columns as "highly-skewed"**: skewness is mathematically defined for any numeric column, but for a binary/categorical column stored as numbers (e.g. a 1/2 rectifier id) it merely reflects the class balance — not an actionable distribution shape. The flag (`highly-skewed: 정류기 (skew=-1.14)`) misled a downstream agent into reasoning about a meaningless transform (observed in a live agent run: the agent copied the false skew note into its EDA summary). `MapDistribution` now takes per-column cardinality (from MLoop's own `ComputeColumnStats`, the same scan `profile` uses) and suppresses the highly-skewed flag for columns with fewer than 3 distinct values; the per-column `skewness` value is still reported, only the actionable *flag* is gated. Backward compatible when cardinality is unavailable (flags any |skew|>1).
- **`mloop analyze importance --label <col>` ranked the label against itself and surfaced target-agnostic structural variance instead of predictive importance**: the command requires a label — implying a target-aware ranking — but mapped DataLens's target-agnostic `Features.Importance` (a variance/condition-number structural score computed over the *full* numeric matrix). Two consequences: (1) the label column appeared in its own ranking, usually at rank #1 (`top = M_X` where `M_X` is the target), and (2) features were ranked by structural variance, not predictive relevance — on a manufacturing set, `KWh` ranked #2 structurally yet contributed nothing to the model (dropping it left R² byte-identical at 0.9402), so an agent/user trusting the ranking would mis-prioritize feature engineering. `AnalyzeJson.RankImportance(FeatureReport, label)` now (a) **always excludes the label** from the ranking, (b) **prefers target-aware sources** — permutation (numeric target) → mutual-info → ANOVA-F (categorical target) — over structural, (c) falls back to structural with an honest `method` tag when the target-aware source is **degenerate** (all-zero among non-label features — DataLens currently leaves the target in the feature matrix, letting a model predict the target from itself and zeroing every real feature's permutation importance; tracked as a DataLens issue), and (d) emits a `method` field (`permutation`/`mutual-info`/`anova-f`/`structural`) so consumers know what the ranking means. Structural diagnostics (condition number, low-variance/high-correlation counts) are still surfaced. Both the `--json` envelope and the console table now share `RankImportance`, so they stay consistent (`InfoCommand`'s deep-analyze rendering is unchanged).

## [0.18.0] - 2026-06-26

### Added
- **`mloop prep plan` / `mloop features select` — declarative policy commands (Phase 2)**: record feature-engineering decisions in `mloop.yaml` without touching data. `prep plan --set type[:method] [--columns c1,c2] [--remove type] [--list] [--name model]` upserts prep steps (keyed by type+columns, idempotent) and shows each step's leakage category (✓ fold-safe / ⚠ leak / ℹ csv-stage); `features select --drop <cols> | --keep <cols> | --reset [--name model]` edits column include/exclude via `ColumnOverride{Type:"ignore"}` (`--keep` computes the complement against the train-data header, label always kept). Both commands accept `--json` for a structured envelope (each prep step reports its leakage `category`/`foldSafe`; features reports the resulting `ignored` columns) so AI agents can read the result programmatically. Statistical fit still happens fold-internally at train time; these commands only edit the config. `validate` now surfaces informational model-class context for scaling steps on tasks where AutoML uses the preFeaturizer (informational, not a hard gate). `mloop.yaml` saving now omits null fields for clean round-trips.

## [0.17.0] - 2026-06-25

### Fixed
- **`mloop validate` under-reported prep leakage on tasks that ignore the preFeaturizer (T-B)**: `InspectPrepLeakage` only flagged the always-leaky transforms (`median` fill, `rolling`, `resample`) and was task-agnostic, so it stayed silent on `normalize`/`scale`/`fill-mean` for tasks whose AutoML pipeline ignores the `preFeaturizer` (clustering, anomaly, ranking, recommendation). At train time those steps are CSV-baked globally — leaking test statistics — and `PrepRouter` already warns; `validate` now mirrors that task-aware routing (`AutoMLRunner.SupportsPreFeaturizer`) and emits the same `UnsupportedTaskLeakageWarning`, so `validate` and train agree. PreFeaturizer-category steps on binary/multiclass/regression remain correctly silent (fitted fold-internally). `ValidateModelConfig` passes the model's task through `ValidatePrepSteps`.
- **`mloop analyze` crashed serializing non-finite statistics (e.g. `importance` on multicollinear data)**: `AnalyzeJson.Serialize` used System.Text.Json's default number handling, which throws on `NaN`/`±Infinity`. The `importance` aspect's `conditionNumber` is infinite precisely when features are collinear — the very condition the aspect exists to diagnose — so `mloop analyze importance` (and any aspect emitting a non-finite double) exited with `positive and negative infinity cannot be written as valid JSON` instead of producing output. A new `FiniteDoubleConverter` (registered on the analyze serializer) writes non-finite doubles as JSON `null`, applied centrally to every `double`/`double?` field. Chosen over `JsonNumberHandling.AllowNamedFloatingPointLiterals`, which would emit `Infinity`/`NaN` tokens that are invalid JSON and break strict parsers such as the MCP bridge's `JSON.parse`. The human-readable `flags` entry still reports the infinite condition number, so the multicollinearity signal is preserved while the numeric field stays valid JSON.

### Added
- **`mloop analyze <aspect>` — granular, read-only EDA command group**: decomposes `mloop info --analyze`'s one-shot deep analysis into per-aspect subcommands — `profile` (types/null%/cardinality/constant columns), `correlation` (high-correlation pairs + multicollinearity), `importance` (feature ranking, requires a label), `outliers` (count/rate/isolation-forest threshold), and `distribution` (skewness/kurtosis/normality tests). Each aspect computes only its slice via DataLens `AnalysisOptions`, and `--json` emits a structured `{aspect, available, summary, data, flags}` envelope for agent/LLM consumption. Read-only: never mutates the data file or `mloop.yaml`. (Agent-driven data-enhancement roadmap, Phase 1.)

### Changed
- **Upgraded FilePrepper 0.6.0 → 0.7.0 and DataLens 0.4.0 → 0.13.0** (umbrella version-governance alignment). DataLens's `AnalysisOptions`/`AnalysisResult` surface is backward-compatible (additive); FilePrepper 0.7 adds auto encoding detection, skip-rows, and constant-column removal. Security transitive pins re-verified (0 vulnerable packages).

## [0.16.3] - 2026-06-25

### Fixed
- **`prep:` statistical transforms leaked test statistics into training**: a `mloop.yaml` `prep:` step that fits a statistic over the data (`normalize`, `scale`, `fill-mean`) was applied to the *entire* dataset and baked into the CSV before AutoML's train/validation split — so each cross-validation fold trained on parameters (min/max, mean, std) computed from rows it would later be evaluated against. The raw→AutoML path was always safe (ML.NET's built-in featurizers fit fold-internally), but adding `prep:` reintroduced leakage. Statistical transforms are now routed to ML.NET AutoML's `preFeaturizer` estimator, which the AutoML contract fits *inside* each fold, eliminating the leak. New `PrepStepClassifier` (classifies each step as preFeaturizer / csv-bake / unsupported), `PrepFeaturizerBuilder` (statistical step → ML.NET estimator), and `PrepRouter` (routing + shared leakage warning), wired through `TrainingConfig.PreFeaturizer` → `AutoMLRunner` (three injection sites, task-aware via `SupportsPreFeaturizer`) and `TrainCommand.ApplyPrepAsync`. Transforms ML.NET can't express fold-internally (`median`, `rolling`, `resample`) remain CSV-baked but now emit an explicit leakage warning, and `mloop validate` gained `InspectPrepLeakage` to surface residual-leakage steps ahead of training. Two compounding gaps were caught in review and fixed before release: preFeaturizer-produced columns were dropped by `InferColumns` merging (crashing real training until `preserveColumns` retention was added), and statistical prep was silently dropped on non-classification/regression tasks (now task-aware via `AutoMLRunner.SupportsPreFeaturizer`). Found via deep-research + code verification of the agent-driven feature-engineering design (V2 leakage hypothesis confirmed against MLoop source).
- **Global runtime cache mistaken for a project root, breaking relative paths under `$HOME`**: after `mloop runtime install` creates the global DL-runtime cache at `~/.mloop/runtimes`, the user-profile directory contains a `.mloop` directory. `ProjectDiscovery.FindRoot` identified a project root solely by the presence of a `.mloop` directory, so walking up from *any* directory under `$HOME` matched that cache and treated `$HOME` itself as the project root. Relative data paths were then resolved against `$HOME` — e.g. `mloop info data.csv` run in a folder that actually contains `data.csv` failed with a `File not found` naming a path under the home directory rather than the file the command was pointed at. `ProjectDiscovery` now excludes the runtime-cache root (the user-profile directory) from project detection — a directory impossible to use as a project anyway, since its `.mloop` would collide with the cache. The established "`.mloop` directory marks a project" contract is otherwise preserved.

## [0.16.2] - 2026-06-24

### Fixed
- **Misleading "Data file not found" when a directory is passed to `predict`**: pointing image-classification `predict` at a folder of images reported `Data file not found` even though the directory exists, because the path was only checked with `File.Exists`. It now diagnoses the directory honestly — `Expected a CSV file but got a directory` — and, for image classification, explains that predict consumes a CSV with an `ImagePath` column while a labelled image directory belongs to `evaluate`. (Full directory-input prediction remains a separate, demand-driven feature.)
- **Image-classification quality gate skipped, promoting non-converged models**: a non-converged image model (micro-accuracy below the 1/N random baseline) auto-promoted to production with only a warning, while tabular tasks were correctly blocked. Three compounding gaps: (1) `mloop init --task image-classification` wrote `metric: auto` (the init template's fallthrough), and `ShouldPromoteAsync` computed the threshold from the raw `auto`/alias string — which has no entry in the threshold table — so the gate branch was silently skipped; (2) the directory-based input schema never recorded the label's class count, so even a resolved metric had no 1/N floor; (3) degenerate-model detection keys off `f1_score`, which image classification doesn't emit (only `accuracy`/`micro_accuracy`/`log_loss`). Fixes: `init` now emits `micro_accuracy` for image classification; `ModelRegistry.ResolveCanonicalMetricKey(metric, task, keys)` falls back to the task's canonical metric when the project metric is `auto`, and the gate now derives both the threshold and the error-direction from the *resolved* canonical key (also closing a latent `auto`/alias inconsistency for tabular tasks); `BuildDirectoryInputSchema` populates the label's `UniqueValueCount` from the class-folder count via the new `ImageDirectoryLoader.CountClasses`. A binary image model at micro-accuracy 0.43 (< 0.5) is now blocked; a 6-class model at 0.90 (> 0.167) still promotes.
- **Metric-alias mismatch broke auto-promotion and `compare` sorting**: a non-default optimization metric alias — `--metric f1`, `r2`, `log-loss`, or multiclass `accuracy` — silently failed lookups against the stored metrics dictionary, which uses canonical keys (`f1_score`, `r_squared`, `log_loss`, `micro_accuracy`). Two symptoms shared one root cause:
  - **Auto-promotion**: `ModelRegistry.ShouldPromoteAsync` looked up `experiment.Metrics["f1"]`/`ContainsKey("f1")` and returned `false` at its first branch, so a quality-gate-passing model was left unpromoted with no error (only `mloop list` showing `Production: 0` revealed it).
  - **`mloop compare --sort f1`**: the alias wasn't in the metric-name set, so the sort and the "best experiment" recommendation silently fell back to the first metric (e.g. accuracy) instead of f1.

  A shared `ModelRegistry.ResolveMetricKey(metric, availableKeys)` normalizer now maps aliases to the stored key actually present (exact-match first, then `f1`→`f1_score`, `r2`→`r_squared`, `log-loss`→`log_loss`, `accuracy`→`micro_accuracy`/`macro_accuracy`), returning null when none match. `ShouldPromoteAsync` uses it for the quality gate and production comparison (falling back to promoting a quality-gated model when the key is absent on either side); `CompareCommand` uses it for `--sort`/config-metric resolution.

## [0.16.1] - 2026-06-23

### Fixed
- **`evaluate` encoding gap**: `mloop evaluate` read the test CSV as forced UTF-8 instead of running it through `EncodingDetector` like `train`/`predict`, so CP949/EUC-KR files (most KAMP data) with Korean column names were garbled — schema validation reported spurious "missing columns / not UTF-8", and `EvaluationEngine` could not find the label column even though training accepted the same file. Both `SchemaValidator` and `EvaluationEngine.LoadTestData` now delegate to `EncodingDetector.ConvertToUtf8WithBom`, matching the loaders.
- **`evaluate` feature-dimension mismatch**: `EvaluationEngine.LoadTestData` reimplemented test-data preprocessing and diverged from `train`/`predict` — it did not strip unnamed/index columns or the DateTime/constant/sparse columns the trained schema marks as `Exclude`, so the rebuilt feature vector was wider than the model (e.g. `expected Vector<Single,114> got 123`) and `Transform` threw. It now mirrors prediction: `CsvDataLoader.RemoveIndexColumns` + the new shared `CsvDataLoader.RemoveExcludedColumns`.

### Changed
- **Inference preprocessing unified (root cause of two predict/evaluate divergences)**: `predict` and `evaluate` each hand-rolled their test-data preprocessing and diverged from each other and from `train`. The sequence is now a single shared `InferenceDataPreprocessor.Prepare` (encoding → flatten multiline → remove index → schema-driven exclude / data-dependent fallback), which `PredictionEngine` and `EvaluationEngine` both call. This collapses the divergence that produced both and closes two gaps it had hidden: multiline-quoted-field flattening and constant-column removal were present in `train` but missing from both inference paths, and `predict`'s hand-rolled UTF-8 BOM check read CP949 as UTF-8 (the class survived in `predict` even after `evaluate` was fixed). `predict` now handles CP949 test files, multiline CSVs, and constant columns identically to `train`/`evaluate`.
- **`RemoveExcludedColumns` shared (DRY)**: the schema-driven excluded-column removal that `PredictionEngine` kept private is promoted to `CsvDataLoader.RemoveExcludedColumns(path, excludedNames)`, joining its sibling `Remove*` helpers; both the prediction and evaluation paths now call it.
- **Security: pinned transitive `System.Security.Cryptography.Xml`** to `10.0.9` via central-package transitive pinning, resolving the high-severity NU1903 advisories (GHSA-37gx-xxp4-5rgx, GHSA-w3x6-4m5h-cxqf) pulled in through `FilePrepper → EPPlus 8.5.0`.
- Doc-comment cref fix (`ObjectDetectionEvaluator`, CS1574) and de-flaked `PerformanceTests.ReproducibilityOverhead` (ratio-based threshold instead of an absolute ms-difference that was fragile under CPU contention).

## [0.16.0] - 2026-06-22

### Added
- **Object detection — `predict` (detections output)**: `mloop predict` on an object-detection model reads an image directory (COCO/YOLO, auto-detected) and emits per-image detections — class label, confidence score, and bounding box (`x0 y0 x1 y1`) — as JSON (`predictions/<model>-detections-<ts>.json`), with a per-class count and per-image preview. The new `ObjectDetectionPredictor` (Core) extracts detections from the scored data view, the structural twin of `ObjectDetectionEvaluator`.
- **Object detection — `evaluate` (mAP)**: `mloop evaluate` scores an object-detection model with mean Average Precision via ML.NET's built-in `EvaluateObjectDetection` (`ObjectDetectionEvaluator` → `map_50` / `map_50_95`), rather than a hand-rolled IoU/AP implementation. `EvaluationEngine` loads the image directory, transforms, and scores; `EvaluateCommand` accepts an image directory and skips CSV schema validation.
- **Image classification — `evaluate` over a directory**: `mloop evaluate` now evaluates an image-classification model over a labelled image directory (folder = label), reporting multiclass accuracy — previously evaluation only accepted a CSV. The directory-based evaluate path is generalized across image classification and object detection via `DataLoaderFactory`.

### Changed
- **`DataProviderBase` extraction**: the `SplitData` / `ValidateLabelColumn` / `GetSchema` / `GetRowCount` duplication across the five data loaders (`Csv`, `Image`, `Coco`, `Yolo`, `ObjectDetection`, ~220 lines) is unified into a `DataProviderBase` abstract class; each loader implements only `LoadData`. Directory-loader label matching is now `OrdinalIgnoreCase` (matching `CsvDataLoader`), and row counting falls back to a cursor when `GetRowCount` is null.
- Directory-dataset resolution for evaluate/predict is centralized in `DatasetDiscovery.FindDirectoryDataset` (object detection → `datasets/coco` → `datasets/yolo` → `datasets`; image classification → `datasets/images` → `datasets`), mirroring `train`'s convention.

## [0.15.0] - 2026-06-21

### Added
- **Image classification — end-to-end working**: `ImageDirectoryLoader` consumes a `datasets/images/<class>/` layout (folder name = label) with TensorFlow transfer learning (ResNet v2 50). `train` → `promote` → `predict` now complete on real data. `mloop init --task image-classification` scaffolds the directory layout. Verified e2e on a KAMP folder=label dataset (5/5 correct on the held-out test images).
- **Object detection — input path**: `CocoDataLoader` (COCO JSON, `bbox` → `x0 y0 x1 y1`) and `YoloDataLoader` (YOLO `images/` + `labels/*.txt`, normalized → absolute via image dimensions), dispatched by `ObjectDetectionDataLoader` with COCO/YOLO auto-detection. Wired through `RunObjectDetectionAsync` (LoadImages + MapValueToKey + bounding-box column) and the `train`/`init` CLI (`datasets/coco/`, `datasets/yolo/`). Evaluation honestly reports object-detection metrics as not-yet-supported (mAP pending) rather than mis-scoring.
- **On-demand DL runtime management**: `mloop runtime install/list/remove` for the TensorFlow (image) and libtorch (object detection / NLP) native runtimes, downloaded on demand and loaded transparently when a DL task runs.

### Fixed
- **DL runtime install/load chain**: `mloop runtime install` crashed on every path — a thread-affine `Mutex` released across `await` threw `ApplicationException` (and masked the real error), the torch version coordinate 404'd / mismatched the TorchSharp ABI, and TorchSharp's native wrapper (`LibTorchSharp.dll`) was not on the search path. Replaced the mutex with a cross-process file lock, pinned libtorch to the binding's `2.2.1.1`, and exposed both the runtime cache and the app's `runtimes/<rid>/native` directory to the native loader.
- **Native DL runtime not loaded on inference paths**: image/object-detection `predict`, `evaluate`, and `serve /predict` crashed with `DllNotFoundException` because the native runtime was registered only on the training path. ML.NET resolves the native library while deserializing the model, so a new shared `RuntimeManager.EnsureRuntimeForTask(taskType)` (no-op for tabular tasks) is now invoked before every model load — training, CLI predict, evaluate, serve, and model-cache preload.
- **Directory-schema label type**: the image/OD schema stored its label `dataType` as the raw `"String"`, which fell through `PredictionEngine`'s type-override default and was read as `Single`, breaking the model's `MapValueToKey` at predict time. The directory schema now uses MLoop's canonical vocabulary (`Text` for the image path, `Categorical` for the label), and `PredictionEngine` tolerates `"String"` defensively.
- **Misleading prediction/evaluation diagnostics**: "feature vector dimension mismatch" messages wrongly attributed the cause to text/value distribution drift. Saved transformers embed their fitted featurizers, so the real cause is a prediction-data column-structure/type mismatch; the four affected messages were corrected.

## [0.14.2] - 2026-06-19

### Fixed
- **IPreprocessingScript contract drift**: public preprocessing examples (`examples/preprocessing-scripts/01–07`), the tutorial/docs code blocks, and MLoop's own `PreprocessingScriptGenerator` referenced types that do not exist in the source — `PreprocessingResult`, `PreprocessingContext`, `context.GetTempPath()` — and the dead bare `MLoop.Extensibility` namespace. The authoritative contract is `Task<string> ExecuteAsync(PreprocessContext)` returning the produced CSV path. Root cause: none of these artifacts live in a csproj, so the build never compiled them and they drifted from the contract that `ScriptLoader` compiles standalone at runtime (no implicit/global usings). Generator-produced scripts therefore failed to compile/load, and copy-pasting the examples broke for consumers.
  - `PreprocessingScriptGenerator` now emits the real contract and accumulates `currentPath` so the returned path always exists (also fixes a latent missing-file return on the encoding-only path).
  - `PreprocessContext` gains null-safe `GetMetadata<T>`/`HasMetadata`, matching `HookContext`/`MetricContext`.
  - Examples 01–07 unified to the real contract; `04_data_cleaning` drops the external CsvHelper dependency in favor of the injected `ctx.Csv` (`ICsvHelper`).
- **`mloop preprocess --help`** now documents the two paths: the default path runs `IPreprocessingScript` files in `.mloop/scripts/preprocess/`, while `--incremental` runs the detector-based rule-discovery workflow.

### Added
- Regression guards that compile artifacts through the real `ScriptLoader`: generated scripts (`PreprocessingScriptGeneratorTests`) and the shipped example files (`ExampleScriptsCompilationTests`), making the examples first-class compile targets.

## [0.14.1] - 2026-06-19

### Fixed
- **ScriptLoader missing BCL references**: user scripts (detectors/hooks/metrics) referencing `Regex` failed to compile (`The type name 'Regex' could not be found ... add a reference`) because `System.Text.RegularExpressions` — a separate assembly, not `System.Private.CoreLib` — was absent from the compilation reference set. Added `System.Text.RegularExpressions` and `System.Text.Json` (with single-file-publish DLL fallbacks). Removed the dead `GetScriptOptions()` code path (the live path is `CompileAndCacheAsync`).
- **RuleApplier silent-success**: all 9 rule-application strategies were no-ops yet reported `Success=true`, so `mloop prep` emitted a "cleaned" dataset byte-for-byte identical to the input while claiming the rules were applied. Introduced `RuleApplicationStatus` (`Applied`/`NotImplemented`/`Failed`); unimplemented strategies now surface `NotImplemented` (`Success=false`) with a clear reason, and the incremental workflow warns when no rows were actually modified. (Built-in application strategies and any custom-apply extension remain backlog.)

## [0.14.0] - 2026-06-19

### Added
- **Pluggable pattern detectors**: `IPatternDetector` implementations can now be contributed by consumer apps via `.mloop/scripts/detectors/*.cs` (auto-discovery), mirroring the existing hooks/metrics extension mechanism. Detectors run alongside the built-in 7 in the incremental preprocessing rule-discovery engine — no core modification required.
  - `ScriptDiscovery.DiscoverDetectorsAsync()` + `GetDetectorsDirectory()`; `InitializeDirectories()` now also creates `detectors/`.
  - `RuleDiscoveryEngine` accepts optional injected custom detectors (DI), combined with the built-ins (backward compatible).
  - `ScriptLoader` exposes MLoop.Core + Microsoft.Data.Analysis to compiled scripts so detectors can reference `DataFrameColumn`, `DetectedPattern`, `PatternType`.
  - `mloop preprocess --incremental` wires discovered detectors into the engine (zero-overhead when the directory is absent).

## [0.11.0] - 2026-03-20

### Added
- **Auto-sampling for large datasets**: `mloop train --max-rows N` with task-aware strategy (stratified for classification, random for others)
- `--sampling-strategy` and `--seed` options for explicit sampling control
- `sample` step type in `mloop prep` pipeline (DataPipelineExecutor)
- Shared PredictionService engine (MLoop.Core.Prediction) used by both CLI and API
- API `/predict` endpoint with real prediction (replacing placeholder)
- SSA forecasting predict via model Transform

### Fixed
- EvaluateCommand Boolean/String label mismatch for multiclass
- PredictCommand fails for unsupervised tasks with empty label
- Spectre.Console markup escaping in ServeCommand output

### Changed
- FilePrepper dependency upgraded 0.5.0 → 0.6.0 (StripCarriageReturn support)
- ConfigureAwait(false) applied to all library async code
- CLI test coverage expanded (17/25 commands covered)

### Validated
- 89 KAMP datasets tested, 0 new bugs
- All 15 ML.NET task types implemented and verified
- 1,595 tests passing, 0 warnings, 0 errors

## [0.10.1] - 2026-03-15

### Fixed
- VectorDataViewType feature detection
- GetRowCount null handling with CountRows helper
- Unsupervised label skip in data validation
- InitCommand updated to allow all 15 task types
- DataQualityValidator updated for unsupervised learning

## [0.10.0] - 2026-03-12

### Added
- ML.NET full 15 task types: Anomaly Detection, Clustering, Ranking, Forecasting (SSA), Time-Series Anomaly, Recommendation, Image Classification, Object Detection, Text Classification, Sentence Similarity, NER, Question Answering
- `mloop runtime` command for on-demand DL runtime management (TorchSharp, TensorFlow)

## [0.6.1-alpha] - 2026-03-10

### Added
- Composite text-likeness heuristic (`LooksLikeText`) for column type detection: average token count, string length, and high-cardinality patterns prevent log messages from being misclassified as categorical
- Text-likeness reclassification in `mloop info` output for consistent info/train behavior
- MCP `mloop_predict`: `unknownStrategy` parameter (auto/error/use-most-frequent/use-missing)
- MCP `mloop_train`: `testSplit`, `noPromote`, `dropMissingLabels` parameters
- 7 unit tests for `LooksLikeText` covering log messages, short codes, numeric IDs, etc.

### Fixed
- Text columns with repeated values (low unique ratio) incorrectly classified as Categorical, causing OneHotEncoding instead of FeaturizeText and prediction failures on unseen text
- Training Configuration display showing "Test Split: 20%" when `--balance` pre-split already isolates test data (now shows "Pre-split (filename)")
- Hidden columns in prediction output selection
- Label inference heuristic and unused data scanner scope
- mloop-mcp submodule: DEP0190 warning, evaluate command, init fixes

### Changed
- Column type detection upgraded from simple 50% unique-ratio threshold to composite heuristic
- MCP-CLI parameter parity improved for predict and train tools

## [0.6.0-alpha] - 2026-02-27

### Added
- `mloop prep run` command for standalone YAML pipeline execution
- Prediction preview (first 5 rows) and distribution display in `mloop predict`
- Data summary and production model comparison in `mloop train`
- Trainer name, metric name, and relative timestamps in `mloop list`
- YAML prep pipeline integration in `mloop predict` (automatic preprocessing)
- Prep step validation and CSV label column check in `mloop validate`
- Centralized error suggestions across all CLI commands (ErrorSuggestions)
- 5 new DataPipelineExecutor operations: extract-date, parse-datetime, normalize, scale, fill-missing
- Pipeline tests for all 14 preprocessing step types
- `mloop status` command with config summary and latest prediction display
- Backup and promotion history in `mloop promote`
- FilePromotionManager for filesystem-based model promotion
- Phase 1 hook/metric extensibility in ScriptDiscovery and AutoMLRunner

### Fixed
- ServeCommand TFM discovery for .NET 10
- DataQualityAnalyzer bug with empty datasets
- PerformanceTests scalability test reliability
- TrainCommand nullable warnings
- Datetime columns auto-excluded from training features
- File I/O optimization in PredictCommand and InfoCommand
- HITL workflow session IDs now shared across decisions in a single workflow execution
- Version consistency: removed hard-coded version from MLoop.Extensibility

### Changed
- FilePrepper integrated via submodule with project reference
- Improved error messages with actionable suggestions and context
- Dead code cleanup and standardized Phase 1 TODO comments
- Assembly version used for accurate SemVer display

## [0.5.1-alpha] - 2025-12-20

### Added
- Self-update system (`mloop update`)
- YAML-based preprocessing pipeline configuration

### Fixed
- ARM64 build targets removed (unsupported by ML.NET)
- NuGet CLI tool publishing disabled in favor of GitHub Releases
- RuntimeIdentifiers conflict with dotnet pack

## [0.5.0-alpha] - 2025-12-15

### Added
- FilePrepper CLI integration for data preprocessing
- Class imbalance detection and `--balance` option
- Custom script extensibility (hooks, metrics, preprocessing)
- Roslyn-based script compilation with DLL caching

### Fixed
- Prediction CSV output formatting
- Cache directory creation for compiled DLLs

## [0.4.0] - 2025-11-01

### Added
- REST API server (`mloop serve`)
- Docker file generation (`mloop docker`)
- Feedback collection system (`mloop feedback`)
- Data sampling utilities (`mloop sample`)
- Retraining trigger evaluation (`mloop trigger check`)

## [0.3.0] - 2025-10-01

### Added
- Multi-model support with namespaced directories
- Experiment promotion workflow (`mloop promote`)
- Dataset profiling (`mloop info`)
- Prediction logging (JSONL)

## [0.1.0] - 2025-09-01

### Added
- Initial release
- ML.NET AutoML training (`mloop train`)
- Prediction with production model (`mloop predict`)
- Experiment management (`mloop list`)
- YAML configuration (`mloop.yaml`)
- Filesystem-based state management

[Unreleased]: https://github.com/iyulab/MLoop/compare/v0.43.0...HEAD
[0.43.0]: https://github.com/iyulab/MLoop/compare/v0.42.0...v0.43.0
[0.42.0]: https://github.com/iyulab/MLoop/compare/v0.41.0...v0.42.0
[0.41.0]: https://github.com/iyulab/MLoop/compare/v0.40.0...v0.41.0
[0.40.0]: https://github.com/iyulab/MLoop/compare/v0.39.0...v0.40.0
[0.39.0]: https://github.com/iyulab/MLoop/compare/v0.38.0...v0.39.0
[0.38.0]: https://github.com/iyulab/MLoop/compare/v0.37.1...v0.38.0
[0.37.1]: https://github.com/iyulab/MLoop/compare/v0.37.0...v0.37.1
[0.37.0]: https://github.com/iyulab/MLoop/compare/v0.36.0...v0.37.0
[0.36.0]: https://github.com/iyulab/MLoop/compare/v0.35.0...v0.36.0
[0.35.0]: https://github.com/iyulab/MLoop/compare/v0.34.0...v0.35.0
[0.34.0]: https://github.com/iyulab/MLoop/compare/v0.33.0...v0.34.0
[0.33.0]: https://github.com/iyulab/MLoop/compare/v0.32.0...v0.33.0
[0.32.0]: https://github.com/iyulab/MLoop/compare/v0.31.0...v0.32.0
[0.31.0]: https://github.com/iyulab/MLoop/compare/v0.30.0...v0.31.0
[0.30.0]: https://github.com/iyulab/MLoop/compare/v0.29.1...v0.30.0
[0.29.1]: https://github.com/iyulab/MLoop/compare/v0.29.0...v0.29.1
[0.29.0]: https://github.com/iyulab/MLoop/compare/v0.27.1...v0.29.0
[0.27.1]: https://github.com/iyulab/MLoop/compare/v0.27.0...v0.27.1
[0.27.0]: https://github.com/iyulab/MLoop/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/iyulab/MLoop/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/iyulab/MLoop/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/iyulab/MLoop/compare/v0.23.2...v0.24.0
[0.23.2]: https://github.com/iyulab/MLoop/compare/v0.23.0...v0.23.2
[0.23.0]: https://github.com/iyulab/MLoop/compare/v0.22.1...v0.23.0
[0.22.1]: https://github.com/iyulab/MLoop/compare/v0.22.0...v0.22.1
[0.22.0]: https://github.com/iyulab/MLoop/compare/v0.21.1...v0.22.0
[0.21.1]: https://github.com/iyulab/MLoop/compare/v0.21.0...v0.21.1
[0.21.0]: https://github.com/iyulab/MLoop/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/iyulab/MLoop/compare/v0.19.3...v0.20.0
[0.19.3]: https://github.com/iyulab/MLoop/compare/v0.19.2...v0.19.3
[0.19.2]: https://github.com/iyulab/MLoop/compare/v0.19.1...v0.19.2
[0.19.1]: https://github.com/iyulab/MLoop/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/iyulab/MLoop/compare/v0.18.4...v0.19.0
[0.18.4]: https://github.com/iyulab/MLoop/compare/v0.18.3...v0.18.4
[0.18.3]: https://github.com/iyulab/MLoop/compare/v0.18.2...v0.18.3
[0.18.2]: https://github.com/iyulab/MLoop/compare/v0.18.1...v0.18.2
[0.18.1]: https://github.com/iyulab/MLoop/compare/v0.18.0...v0.18.1
[0.18.0]: https://github.com/iyulab/MLoop/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/iyulab/MLoop/compare/v0.16.3...v0.17.0
[0.16.3]: https://github.com/iyulab/MLoop/compare/v0.16.2...v0.16.3
[0.16.2]: https://github.com/iyulab/MLoop/compare/v0.16.1...v0.16.2
[0.16.1]: https://github.com/iyulab/MLoop/compare/v0.16.0...v0.16.1
[0.16.0]: https://github.com/iyulab/MLoop/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/iyulab/MLoop/compare/v0.14.2...v0.15.0
[0.14.2]: https://github.com/iyulab/MLoop/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/iyulab/MLoop/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/iyulab/MLoop/compare/v0.11.0...v0.14.0
[0.11.0]: https://github.com/iyulab/MLoop/compare/v0.6.1-alpha...v0.11.0
[0.6.1-alpha]: https://github.com/iyulab/MLoop/compare/v0.6.0-alpha...v0.6.1-alpha
[0.6.0-alpha]: https://github.com/iyulab/MLoop/compare/v0.5.1-alpha...v0.6.0-alpha
[0.5.1-alpha]: https://github.com/iyulab/MLoop/compare/v0.5.0-alpha...v0.5.1-alpha
[0.5.0-alpha]: https://github.com/iyulab/MLoop/releases/tag/v0.5.0-alpha
