# MLoop User Guide

Complete guide for using MLoop, the ML.NET CLI tool for building and managing machine learning models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Commands](#core-commands)
3. [Deployment Commands](#deployment-commands)
4. [Data Preprocessing](#data-preprocessing)
5. [Extensibility](#extensibility)
6. [Project Structure](#project-structure)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Exit Codes](#exit-codes)

---

## Quick Start

### Installation

Download from [GitHub Releases](https://github.com/iyulab/MLoop/releases) and place in your PATH.

```bash
# Self-update to latest version
mloop update
```

### Initialize Project

```bash
# Create new ML project
mloop init my-ml-project --task regression
cd my-ml-project

# Project structure created:
# .mloop/               # Project marker
# datasets/             # Your training data goes here
# models/
#   ├── staging/        # All experiments
#   └── production/     # Promoted model
# predictions/          # Prediction outputs
```

### Train Your First Model

```bash
# 1. Add your training data
cp ~/data/train.csv datasets/train.csv

# 2. Train with AutoML (60 seconds)
mloop train datasets/train.csv --label price --time 60

# Output:
# ✅ Training complete: exp-001
# 📊 Best Trainer: FastTreeRegressor
# 📈 R²: 0.9523, RMSE: 12.45

# 3. Make predictions
mloop predict
# Auto-uses production model + datasets/predict.csv
# Output: predictions/default-predictions-20241110-143022.csv
```

---

## Core Commands

### `mloop init`

Initialize a new ML project with filesystem structure.

```bash
# Basic usage
mloop init <project-name> --task <regression|binary-classification|multiclass-classification>

# Examples
mloop init sales-forecast --task regression
mloop init fraud-detection --task binary-classification
mloop init sentiment-analysis --task multiclass-classification
```

**Created Structure**:
```
my-project/
├── .mloop/
│   └── scripts/
│       ├── hooks/       # Pre/post-train hooks
│       └── metrics/     # Custom metrics
├── datasets/
├── models/
│   ├── staging/
│   └── production/
└── predictions/
```

### `mloop train`

Train models with MLoop's trainer search (over ML.NET's trainers) and automatic experiment tracking.

```bash
# Basic training
mloop train <data-file> <label-column> [options]

# What to train:
#   --name, -n <model>          Model name (default: 'default')
#   --label, -l <column>        Label column (default: from mloop.yaml)
#   --task, -t <type>           ML task type (default: from mloop.yaml)
#   --metric, -m <name>         Metric to optimize (default: the task's primary metric). Case and
#                               separators do not matter (F1Score, f1-score and f1_score are one
#                               name); an unknown name, or one the task cannot optimize, is
#                               refused before training starts
#   --test-split <0-1>          Test split fraction (default: 0.2)

# Time budget:
#   --time <seconds>            Fixed training time limit. Without it, the budget is estimated from
#                               the data (auto-time) unless mloop.yaml sets time_limit_seconds
#   --auto-time                 Force the estimate even when mloop.yaml sets time_limit_seconds
#   --no-auto-time              Disable the estimate (falls back to 300s)
#                               Deep-learning tasks (text classification, NER, sentence similarity,
#                               question answering, object detection) run 10 epochs whatever the
#                               budget — a fit cannot stop early and keep a model. Their progress
#                               bar counts epochs and estimates the time left after the first one.

# Data:
#   --data, -d <files...>       Training file(s); several files with the same schema are merged
#   --auto-merge                Merge every same-schema CSV found in datasets/
#   --drop-missing-labels       Drop rows with a missing label (default: on wherever the label is
#                               a target to learn; off for a time series, whose label is the series)
#   --max-rows <n>              Cap on training rows; larger data is sampled (random for
#                               regression/anomaly, stratified for classification)
#   --sampling-strategy <s>     'random' or 'stratified' when --max-rows applies (alias: --sampling)
#   --seed <n>                  Random seed for sampling (default: 42)
#   --balance [strategy]        Class balancing: 'auto' (to 10:1 when the ratio exceeds 10:1),
#                               'none', or a target ratio such as '5'; bare --balance means 'auto'
#   --group-column <column>     Query/group column for the ranking task
#   --weight <column>           Column that weights the rows (binary, multiclass, regression);
#                               never a feature. Same as `weight:` in mloop.yaml
#   --portable                  Train only with steps ONNX can express, so `mloop export onnx`
#                               works (binary, multiclass, regression). Same as `portable: true`

# Before and after training:
#   --analyze-data              Analyze data quality and print preprocessing recommendations
#                               without training
#   --generate-script <path>    Write a preprocessing script from that analysis
#   --no-promote                Skip automatic promotion to production
#   --json                      Newline-delimited JSON events (phase/trial/warning/result/error)
#                               on stdout instead of the rich display

# Examples
mloop train datasets/train.csv --label price --time 120
mloop train data/sales.csv --label revenue --time 300 --metric r_squared
mloop train data/fraud.csv --label is_fraud --time 180 --test-split 0.3

# Multi-file training (v0.4.0+)
mloop train --data normal.csv outlier.csv --label Target --task regression
mloop train --auto-merge --label Status --task binary-classification

# Class imbalance handling (v0.5.0+)
mloop train data.csv --label is_fraud --task binary-classification --balance auto
mloop train data.csv --label defect --task binary-classification --balance 5  # Target 5:1 ratio

# Weighted rows — e.g. labels from a recheck sample, each row weighted 1/π
mloop train data.csv --label risk --task binary-classification --weight inv_pi
```

**Weighted training** (`--weight <column>`, or `weight: <column>` under the model in `mloop.yaml`):
each row counts by its weight — the relative frequency it should have, such as 1/π for a row drawn
with probability π, or a lower weight for a less reliable label source. AutoML takes a weight
column and does not pass it to the trainers it builds, so MLoop resamples instead: the training
rows are split into the rows models are fitted on and the rows they are chosen on, each part is
resampled in proportion to the weight on its own (the same row is never on both sides), and the test
metrics are weighted too — every metric of the task, under its usual name, with `report.md` and the
training output saying "Each test row counts by its *column*". The weight is never a feature. It
applies to binary-classification, multiclass-classification and regression; on any other task, with
`--balance`, on the label column itself, or on a column also declared a feature, training refuses it
instead of ignoring it. Weights must be zero or positive numbers, and not all zero.

Metrics weighted by different columns — or one weighted and one not — share names, not a
distribution. Training therefore does not promote a model over a production model measured another
way (it says so and leaves production unchanged; promote explicitly once both are scored on one
evaluation set with `mloop evaluate … --weight`), and `mloop compare` warns, adds a Weight row, and
carries `weightsDiffer` and each experiment's `weightColumn` in `--json`.

**Portable training** (`--portable`, or `portable: true` under the model in `mloop.yaml`) trains with
steps an ONNX runtime computes exactly, so `mloop export onnx` can export the model. The difference
is in text columns: instead of ML.NET's text featurizer (which has no ONNX form) each text column is
featurized as counts of character n-grams of lengths 1 to 4, case-sensitive — word tokenization and
text normalization do not reproduce exactly in ONNX Runtime. On the UCI SMS spam data (60 s) a
portable model scored accuracy 0.9848, AUC 0.9948 against 0.9865, 0.9874 for the default. Numeric
and category columns train the same either way. Binary classification, multiclass classification and
regression; the experiment records `portable: true`.

**Automatic Features**:
- **Data File Formats**: besides CSV, `train`, `predict`, `evaluate`, `info`, `analyze` and `detect`
  take `.parquet`, `.xlsx`/`.xls`, `.json` (an array of objects) and `.tsv` files. A Parquet struct
  column becomes one column per leaf, named by its dotted path — a label stored as `labels.label` is
  passed as `--label labels.label` — and a list column becomes one column of JSON text, which training
  does not read as a feature (`info` shows it as `Excluded (Structured)`); unpack it into rows first.
  SVMlight / libsvm files (`label [qid:q] index:value …`, as learning-to-rank sets are published) are
  read too — by the `.svm`/`.svmlight`/`.libsvm` extensions or by their lines, so `rank.train` works —
  as `label`, `qid` and `f<index>` columns; a group-size file beside the data (`rank.train.query`)
  supplies `qid`. Train a ranking model on one with `--label label --group-column qid`.
  **A folder of data files** in one format is read as one table, rows in file-name order — the shape
  of exports that write a file per record (`mloop train readings/ --label temp`), which could not be
  listed one by one past the command line's length limit. Files of mixed formats or different columns
  are refused by name; other files in the folder (`.done` markers, notes) are left out. A folder laid
  out for image classification or object detection stays that task's input.
  A JSON file whose rows sit in a nested array takes `--records` with the dotted path to that array
  (`train`, `predict`, `evaluate`, `info`, `analyze`, `detect`): `--records data.paragraphs.qas.answers` reads each answer
  as a row, and a row also carries the fields of the items it sits in, named by their array
  (`paragraphs.context`, `qas.question`). Without it, such a file is refused with the paths that would
  read it.
- **Encoding Detection**: CP949/EUC-KR files auto-converted to UTF-8 (Korean text support)
- **Schema Validation**: Same-schema files auto-detected for `--auto-merge`
- **Label Handling**: Missing labels auto-dropped for classification tasks
- **Data Quality Analysis**: Automatic class distribution and quality warnings
- **Feature Exclusion**: Columns that cannot carry signal are dropped before training (see below)

**Output** (under `models/<name>/staging/exp-XXX/`):
- `model.zip` — the trained model
- `metadata.json`, `metrics.json`, `config.json` — what was trained, how it scored, how it was configured.
  The config carries `dataFileHash`, the SHA-256 of the training file as you have it, so you can
  check later which data an experiment actually saw. Classification metrics include `ece`, the
  expected calibration error of the model's probabilities (ten bins; 0 = a stated 0.8 is right 80% of
  the time) — read it before using a confidence as a threshold. It is reported, never optimized, and
  absent for a model without probabilities.
- `trials.ndjson`, `leaderboard.json` — every trial the search completed, in completion order and ranked.
  The model is not always the top-ranked trial: when the top trial's model is large (100,000 tree leaves or
  more — tens of megabytes), the search keeps the smallest model within one standard error of its score
  (across folds, or across slices of a large dataset's holdout) — trees × leaves, a linear model being the
  smallest — since two trials inside that margin cannot be told apart and the larger one costs time to save,
  load and score, and disk. That trial carries
  `"selected": true`, `mloop list --trials <exp-id>` marks it `(model)`, and the run says which it kept over which
- `report.md` — the experiment as a page: result, metrics, ranked trials, and each column's role (with
  the reason for any exclusion). A rendering of the files above for reading and sharing; no command
  reads it back
- First successful model auto-promoted to production. After that, a new model replaces production
  only when it passes the quality gate and beats production on the task's primary metric; a task
  that reports no primary metric is saved to staging and left for `mloop promote`

#### Which columns training drops, and how to change that

Before column inference runs, these kinds of column are removed from the feature set. Each removal is
reported once, as `Excluded from features (<reason>): <columns> — <what that means>` — as a warning on the terminal, and as a
`warning` event for a `--json` consumer — and recorded in the model's schema, so `predict` and
`evaluate` drop exactly the same columns from the data you hand them later.

| Reason | Rule | Why |
|---|---|---|
| `DateTime` | The values parse as dates or times | ML.NET would featurize the text into thousands of character n-grams; extract date features with `mloop prep` instead |
| `Sparse` | More than 90% of the values are missing | Nothing to learn from, and a wide sparse text column can exhaust memory |
| `Constant` | Every non-missing value is the same | Zero predictive signal |
| `Identifier` | A **text** column in which every row carries its own distinct, whitespace-free value — a customer id, an order number, a UUID | Pure noise as a feature, and a leak path whenever the id encodes something about the label |
| `Structured` | Every value is a JSON list or object — what a Parquet or JSON list column becomes in a table | Structure to unpack into rows or columns before training, not one value |

The identifier rule is deliberately narrow. Numeric columns are never identifiers (measurements are
distinct per row too); a value containing whitespace, a single repeated value, or a missing value
disqualifies the column; and fewer than 20 rows is not enough evidence. A strictly increasing integer
column is *not* removed — a day counter can be a real feature — but it is reported as a possible ID column, judged on the
whole file before the split, with the command to exclude it.

The decision is taken once, on the whole training file, and applied to every partition — so a
column that happens to be constant inside one split cannot give that split a different feature width.

**To exclude a column yourself**, declare it in the project (this is policy, not a data change):

```bash
mloop features select --drop customerID,notes   # excluded from features
mloop features select --keep age,income,region  # everything else excluded
mloop features select --reset                   # clear the policy
```

**String columns are either categories or text.** A string column whose values do not read as
language — short codes or levels such as `sex`, `workclass`, `INFO`/`WARN`/`ERROR` — and that has
at most 100 distinct values (or distinct values on under a tenth of its rows) is a **category** and
is one-hot encoded. Any other string column is **text** and is featurized with word and character
n-grams (TF-IDF). Training prints which is which (`Categorical column(s): …`), the decision is made
once from the whole training file, and the saved schema records it. To decide yourself, declare the
column `type: categorical` or `type: text` below.

**Yes/no columns are read as 0/1.** A column whose every value is a yes/no word — `yes`/`no`,
`true`/`false`, `t`/`f`, `y`/`n` in any case — is a boolean feature in every task, including anomaly
detection, clustering and ranking. The saved schema records it as `Boolean`, and a prediction accepts
any of those spellings for it, or a JSON `true`/`false` (`mloop serve`); drift counts them by meaning,
not spelling. A column that holds `0`/`1` is a number, and one with a missing value or a third value is
a category.

This writes column overrides into `mloop.yaml`, which you can also edit directly. An explicit type
override is honoured over inference, and — for the identifier rule — over the heuristic: a column
you declare as `text`, `categorical` or `numeric` is never dropped as an identifier, and neither is
a ranking `group_column` or a recommendation `user_column`/`item_column`.

```yaml
models:
  default:
    task: binary-classification
    label: Churn
    columns:
      customerID:
        type: ignore          # excluded from features
      notes:
        type: text            # free text — TF-IDF, never dropped as an identifier
      zip:
        type: categorical     # one-hot, even though it looks numeric
```

**Text models read the columns you declare `text`.** Text classification and NER read one text
column, sentence similarity reads two (in declaration order: the first is sentence 1), and question
answering reads a passage and a question (the longer of the two is the passage). Undeclared, they take string columns whose values read as
language — several words, or long, or mostly distinct — and skip short codes such as a source tag or
a category, even when those come first in the file. If too few columns read as language, the rest
are filled from the remaining string columns and the choice is printed as a warning, with how to
declare it. `train` logs which columns it used (`Sentence similarity: s1='…', s2='…'`).

### Image Classification

Image classification reads a **directory of images** instead of a CSV file. Subfolder names are the
class labels (the `folder name = label` convention):

```
datasets/images/
├── OK/    img001.jpg  img002.jpg  ...
└── NG/    img101.jpg  img102.jpg  ...
```

Workflow:

```bash
# 1. Scaffold an image project (creates datasets/images/ + a README)
mloop init vision --task image-classification
cd vision

# 2. Drop images into datasets/images/<class>/

# 3. Install the TensorFlow CPU runtime once (~182MB, on-demand)
mloop runtime install tf

# 4. Train — the directory is auto-detected (or pass it explicitly)
mloop train --task image-classification
mloop train --task image-classification datasets/images

# 5. Predict a folder of new images (subfolders are searched too)
mloop predict new-photos/

# 6. Score the model on a labelled folder laid out like the training data
mloop evaluate exp-001 test-images/
```

Notes:
- `mloop info <folder>` counts the images in each class folder before you train.
- Training first runs every image through a pretrained network once — most of the time on a CPU —
  then trains for at most 200 epochs, stopping early once accuracy stops improving. The progress bar
  shows both. The time limit does not shorten it.
- A prediction CSV with an `ImagePath` column (one row per image) is read as well.
- Supported extensions: `.jpg .jpeg .png .bmp .gif`. Other files (e.g. `meta.json`) are ignored.
- The trainer uses TensorFlow transfer learning, so the `tf` runtime is required. Without it, training
  stops with a clear message pointing to `mloop runtime install tf`.
- CSV-only steps (encoding conversion, sampling, balancing, multi-line flattening) are skipped for image
  directories. The loader warns about empty class folders, single-class datasets, and severe class imbalance.

### Named Entity Recognition (NER)

NER tags every word of a sentence. The data is two columns: the sentence, and its tags — one per
word, separated by spaces, in the order the words appear. Words outside every entity are tagged `O`.

```csv
Sentence,Label
우리는 내일 오후 7시에 만나요,O B-DATE B-TIME I-TIME O
Alice lives in Paris,B-PER O O B-LOC
```

```bash
mloop init tagger --task ner
mloop train datasets/train.csv --label Label
mloop predict datasets/predict.csv      # a Sentence column; the answer is a Label-shaped tag string
mloop evaluate exp-001 datasets/test.csv
```

A sentence and its tags must have the same number of words. The model's answer is written the same
way the label is read — `O B-DATE B-TIME I-TIME O` — so a prediction file can be compared with, or
fed back as, a training file. `macro_accuracy` — the primary metric — averages the share of words
tagged correctly over the tags that occur, so tagging everything `O` scores 1/N; `micro_accuracy` is
the plain share of words tagged correctly, which on most text is dominated by `O`. The quality gate
reads the label's classes as its tags, so a model that does no better than answering `O` stays in
staging.

Both count words, and a word can be right while its entity is not: `B-PER I-PER` answered `B-PER O` is half
the words and no person. `entity_precision`, `entity_recall` and `entity_f1` count whole entities — the same
words with the same type — as NER is usually reported (CoNLL). The tags are read as the IOB family is written:
`B-`/`I-`, `E-`/`S-` (BIOES), or a bare type repeated over the words of one entity.

### Question Answering

One row per question, with the answer and where it starts in the passage:

```csv
context,question,answer,answer_start
"Seoul is the capital of Korea.",What is the capital of Korea?,Seoul,0
```

- The **label** is the answer text (`label: answer` in `mloop.yaml`).
- The passage and the question are the two text columns besides it; the longer one is the passage.
- The start position is found by what it holds, not by its name: the numeric column where the passage,
  read from that character position, gives the answer on at least nine in ten rows. Without one,
  training stops and says so — an extractive model learns *where* the answer is, and the position is
  that lesson.
- A SQuAD-format JSON file is read as published: `mloop train squad.json --label text --records
  data.paragraphs.qas.answers` gives one row per answer with its `paragraphs.context` and
  `qas.question`. If another text column it brings along is taken for the question, name the two
  with `type: text` in `mloop.yaml`.
- **Language**: ML.NET's question-answering model does not cut its answer out of the passage; it
  decodes it from English RoBERTa tokens, one byte per character. An answer outside Latin-1 —
  Korean, Chinese, Japanese, Cyrillic — comes out with those characters missing, so it can never
  match exactly. Training says how many answers are like that and where `exact_match` stops.
- A prediction needs only the passage and the question. Its `predictedLabel` is the model's best answer
  and `score` that answer's model score (not a probability).

Training and `mloop evaluate` score the best answer against the expected one the way the KorQuAD
benchmark does: `char_f1`, the F1 of the characters the two answers share after normalization
(lowercase, quotes and punctuation removed), is the primary metric; `exact_match` is reported beside
it. Characters rather than words, because an answer is often part of a word — `서울에서` contains the
answer `서울` — and a word-level F1 would score that zero.

### `mloop predict`

Run predictions with trained models.

```bash
# Auto-discovery mode (uses production model + datasets/predict.csv)
mloop predict

# Custom paths
mloop predict [data-file] [options]

# Options:
#   --name <model>               Model name (default: 'default')
#   --experiment <exp-id>        Predict with this experiment instead of production — a shadow
#                                prediction: a staging candidate answers the same rows, production
#                                does not move. --json names it in `experimentId`; --log records
#                                its rows under its own id, so `mloop drift` (production's rows)
#                                and feedback keep the two apart
#   --model <path>               Predict with a model file (outside the project's experiments)
#   --json                       Emit pure JSON to stdout (progress on stderr) — use this for
#                                machine/automation consumption. The default output mixes a
#                                human-readable summary into stdout.
#   --output <path>              Output file path
#                                (default: predictions/<model>-predictions-TIMESTAMP.csv;
#                                 object detection writes <model>-detections-TIMESTAMP.json and
#                                 forecasting writes <model>-forecast-TIMESTAMP.csv)
#   --unknown-strategy <mode>    Handle unknown categorical values (v0.6.1+):
#                                'auto'               - Auto-select based on ratio (default)
#                                'error'              - Fail on unknown values
#                                'use-most-frequent'  - Replace with most common value
#                                'use-missing'        - Replace with empty
#   --log                        Enable prediction logging to DataStore

# Examples
mloop predict                                          # Auto mode
mloop predict data/test.csv --experiment exp-003      # Candidate's answers, production untouched
mloop predict data/test.csv --model ../other/model.zip
mloop predict --output results/forecast.csv
mloop predict --unknown-strategy use-most-frequent     # Handle unseen categorical values
```

### `mloop list`

List all experiments with status and metrics.

```bash
# List all successful experiments
mloop list

# Show all experiments including failed
mloop list --all

# Output:
# ┌─────────┬──────────────────┬───────────┬─────────────┬──────────────┐
# │   ID    │    Timestamp     │  Status   │ Best Metric │    Stage     │
# ├─────────┼──────────────────┼───────────┼─────────────┼──────────────┤
# │ exp-003 │ 2025-11-10 10:30 │ Completed │    0.9523   │ ★ Production │
# │ exp-002 │ 2025-11-10 10:20 │ Completed │    0.9401   │   Staging    │
# │ exp-001 │ 2025-11-10 10:10 │ Completed │    0.9278   │   Staging    │
# └─────────┴──────────────────┴───────────┴─────────────┴──────────────┘
```

### `mloop promote`

Manually promote an experiment to production — the only promotion target; an unpromoted experiment
already lives under `staging/`, so there is no separate "promote to staging" step.

```bash
mloop promote <experiment-id>

# Examples
mloop promote exp-002              # Promote to production
mloop promote --latest             # Auto-select the most recent experiment
mloop promote --best               # Auto-select the best-scoring experiment
mloop promote exp-002 --force      # Promote even though the quality gate refuses it
mloop promote --history            # Every change of production so far (--json for machines)
```

`promote` applies the quality gate training applies before it promotes. A model no better than a
trivial baseline — at or below the floor for its metric (AUC 0.5, the accuracy of always answering
the most common class, R² 0), or a binary model that answers one class for every row and ranks no
better than chance (AUC 0.5 or below, or no AUC) — is refused with the reason, and the command exits
1. `--force` promotes it anyway and prints the reason as a warning. `--best` picks the best
experiment the gate accepts (with `--force`, the best of all of them), and `mloop train` says why it
left a model in staging. `--best` compares only experiments measured under one row weighting: when
weighted and unweighted experiments (or ones weighted by different columns) are both candidates it
refuses and lists them by weighting — score them on one evaluation set with
`mloop evaluate <experiment> <file> --weight <column>` and promote one by its ID.

A binary model that **ranks the classes well but answers one class at its default decision
threshold** is not refused: its predicted labels carry no information, but its scores do, and a
consumer that decides by probability can use it. The gate passes it and says so as a warning —
`train` and `promote` print it, `promote --json` and `POST /promote` return it as `qualityWarning`.
This applies when the project's metric is a ranking one (`auc`, `auprc`): a model that answers one
class sits on the floor of `accuracy` (the share of the most common class) and of `f1_score` (0), and
is refused on that metric as before.

`mloop list --json` carries each experiment's `metrics`, the gate's verdict as `qualityGate` (`null`
when it passes) and `qualityWarning` (`null` unless the gate passed it with the warning above).
`POST /promote` applies the same gate and answers 422; send `"force": true` to override.

#### A project's own gate policy

The fixed gate asks whether a model learned anything. A project can add its own rules under the
model in `mloop.yaml`, judged on an evaluation set at the moment of promotion — by `promote`, by a
training run, and by `POST /promote`:

```yaml
models:
  default:
    task: multiclass-classification
    label: risk
    gate:
      data: datasets/recheck.csv        # labeled rows the rules are judged on
      weight: inv_pi                    # optional; defaults as `evaluate --weight` does
      no_worse_than_production: true    # refused when production is better on these rows
      limits:                           # classifiers: of the rows labeled `actual`,
        - actual: [high, critical]      # at most `max` may be answered `predicted`
          predicted: [low]
          max: 0.02
      slice_by: site                    # the limits hold in every slice with 30+ rows too
```

`no_worse_than_production` is the row-by-row comparison of `evaluate --against production`: refused
when the paired interval says production is better (a first model has nothing to be worse than).
A limit reads the confusion matrix — of the rows labeled with one of `actual`, the share answered
with one of `predicted`. A refused model stays in staging with the rule that held it, `--force`
promotes it anyway, and the history records the promotion as `forced` with that rule. A rule that
cannot be judged — no `data`, a missing file, a class limit on a regressor, a production model
trained for another label — refuses: a policy that passed whenever its data went missing would be
no policy. `mloop validate` checks the policy and its file.

Every change of the production slot — `promote`, a training run promoting its own result,
`rollback`, and the same through the REST API — is recorded in `models/<name>/promotion-history.json`,
and `mloop promote --history` lists it: when, which experiment, how (`promote`, `auto-promote`,
`rollback`), the quality gate's outcome at that moment (`passed`, `forced` with the refusal, or
`not-applied` for a rollback) and the project metric's value. `--json` returns it as `promotions`,
each with the experiment it replaced (`previousExperimentId`); `GET /promotions?name=` returns the same. Entries written
before the gate was recorded carry `null` there.

### `mloop threshold`

A binary classifier answers its positive class from a probability of 0.5 unless told otherwise. When
a wrong positive costs more than a missed one, choose the threshold on labeled rows for the precision
you need: the lowest threshold that reaches it, which keeps the most recall.

The precision can be the **other** class's. When a «no» lets a row skip a check — «safe», «not a
defect» — a wrong «no» is the costly one, and the precision to reach is that class's: name it with
`--precision-of`. The threshold is then the highest one at which the rows answered «no» are right
that often, which keeps the most of their recall. Which class is positive is the training's
decision; `--precision-of` takes the label value either way.

```bash
mloop threshold --precision 0.95 --data datasets/recheck.csv           # the production model
mloop threshold exp-004 --precision 0.95 --data datasets/recheck.csv --weight inv_pi
mloop threshold --precision 0.99 --precision-of safe --data datasets/recheck.csv   # «safe» must be right 99%
mloop threshold                                                       # show the current one
mloop threshold --clear                                               # back to 0.5
```

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Model name (default: `default`) |
| `--precision` | | Target precision of the positive class (or of `--precision-of`), above 0 and at most 1 |
| `--precision-of` | | The class whose precision `--precision` targets, by its label value (default: the positive class) |
| `--data` | | Labeled evaluation file to choose on (required with `--precision`) |
| `--weight` | | Column that weights the rows (defaults as `evaluate --weight` does) |
| `--clear` | | Remove the threshold |
| `--json` | | Output as JSON |

The threshold is written beside the model (`decision-threshold.json`, with the precision and recall
it reached and the file it was chosen on) — on the experiment, and on production too when the
experiment is in production. From then on every path that loads the model answers at it: `predict`
(CSV and `--json`), `mloop serve` (its model cache reloads), `evaluate` (accuracy, precision, recall,
`negative_precision`, `negative_recall`, F1 and the confusion matrix — AUC does not depend on it) and the gate policy. `promote` carries it
with the model, and a promoted model without one answers at 0.5 again. Rows of one probability are
answered alike, so a threshold never splits them — and it sits halfway between the last probability
answered one way and the next, never on a row's own probability, so the arithmetic of another runtime
(an exported ONNX graph differs in the seventh digit) keeps every row's answer; when not even the rows the model is most confident
about reach the precision, the command says so and changes nothing. A model whose trainer gives no
probability cannot be given one. The ONNX export does not apply it — the manifest names it, for the
consumer to apply.

### `mloop calibrate`

A classifier's probability is a promise: of the rows it calls 0.8, about 8 in 10 should be
right. `ece` (reported by `train` and `evaluate`) measures how far it is from keeping it. When a
confidence is used to skip a check, recalibrate it on labeled rows the model was not trained on —
without retraining — and every path answers with the calibrated probability.

```bash
mloop calibrate --data datasets/recheck.csv                        # the production model, Platt
mloop calibrate exp-004 --data datasets/recheck.csv --method isotonic --weight inv_pi
mloop calibrate --name grade --data datasets/recheck.csv           # multiclass: temperature
mloop calibrate                                                    # show the current one
mloop calibrate --clear                                            # back to the model's own
```

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Model name (default: `default`) |
| `--data` | | Labeled rows to fit on |
| `--method` | | Binary: `platt` (default — two parameters, steady on a few hundred rows) or `isotonic` (no shape assumed, needs more rows). Multiclass: `temperature` (default and only — one parameter; every class's log-probability is scaled by 1/T and renormalized, so each row's answer and order of classes stay) |
| `--weight` | | Column that weights the rows (defaults as `evaluate --weight` does) — `1/π` for a recheck sample |
| `--clear` | | Remove the calibration |
| `--json` | | Output as JSON |

The calibration is written beside the model (`calibration.json`, with the method, its parameters and
the ECE before and after on the rows it was fit on) — on the experiment, and on production too when
the experiment is in production. `predict` (the probabilities and the label, which answers at 0.5 of
the calibrated probability), `mloop serve`, `evaluate` (its `ece` and the answer-dependent metrics)
and the gate policy then read it; `promote` carries it. Both methods keep the order of the
probabilities, so AUC does not move; a multiclass model's answers do not change at all, only how
sure it says it is (its `ece` and `log_loss` move). The after-ECE is measured on the fit rows and is optimistic —
check it with `mloop evaluate` on other rows; when it is worse than before, the command says so.

A decision threshold (binary) is chosen on probabilities, so the order is **calibrate, then threshold**: the
command refuses a model that has a threshold, and a threshold chosen afterwards answers from the
calibrated probability. The ONNX export carries the trained model's own probability; the manifest
names the calibration for the consumer to apply.

### `mloop rollback`

Returns production to the experiment it held before the current one, from the promotion history.
The current production is backed up first, and the rollback is recorded in the history, so a second
rollback undoes the first.

```bash
mloop rollback                     # back to what the current production replaced
mloop rollback --to exp-001        # back to a named experiment that has been production before
mloop rollback --json              # machine-readable result
```

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Model name (default: `default`) |
| `--to` | | Experiment to return to; it must have been in production before |
| `--json` | | Output the result as JSON |

Only a former production experiment qualifies — promoting one that never served is a new decision,
which is `mloop promote`'s, quality gate included. `POST /rollback` (admin) does the same and answers
409 when there is nothing to return to.

### `mloop status`

Everything the project currently holds, on one screen: the configuration each model was defined
with, how many experiments it has, which one is in production, and when it last predicted.

```bash
mloop status                       # the overview
mloop status --verbose             # adds the per-file data section
mloop status --json                # machine-readable, on stdout
```

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Show detailed status information |
| `--json` | | Output structured JSON instead of a console table |

```
── MLoop Project: demo ─────────────────────────────────────────────────────────

              Configuration (mloop.yaml)
╭─────────┬────────────┬───────┬────────────┬────────╮
│ Model   │ Task       │ Label │ Time Limit │ Metric │
├─────────┼────────────┼───────┼────────────┼────────┤
│ default │ regression │ Label │       auto │  rmse  │
╰─────────┴────────────┴───────┴────────────┴────────╯

                                Models Overview
╭─────────┬─────────────┬───────────┬────────────┬─────────────┬───────────────╮
│ Model   │ Experiments │ Completed │ Production │ Best Metric │  Last         │
│         │             │           │            │             │   Prediction  │
├─────────┼─────────────┼───────────┼────────────┼─────────────┼───────────────┤
│ default │      2      │     2     │ ✓ exp-001  │ 0.9564 rmse │      15m ago  │
╰─────────┴─────────────┴───────────┴────────────┴─────────────┴───────────────╯

── Summary ─────────────────────────────────────────────────────────────────────

Models: 1  Experiments: 2  Completed: 2  Production: 1
```

**`Best Metric`** is the best value among experiments **measured alike** — the same metric over the
same row weighting as the newest experiment. A weighted and an unweighted accuracy describe
different distributions, so neither outranks the other. When the best value is weighted, or some
experiments were measured differently and were left out, the cell carries a `*` and a note under
the table says which:

```
* default: accuracy, weighted by 'w' — best among experiments measured like the latest; 1 measured differently is not compared.
```

**`Last Prediction`** counts every kind of prediction output — scored rows, object-detection
results and forecasts alike — and is written as an age (`15m ago`, `3d ago`, `4mo ago`) rather
than a timestamp. The exact instant is in `--json`.

Under `--json` the document goes to **stdout** and the tables above go to stderr, so
`mloop status --json > status.json` gives a parseable file while you still see the summary:

```json
{
  "projectRoot": "/path/to/demo",
  "models": [
    {
      "modelName": "default",
      "totalExperiments": 2,
      "completedExperiments": 2,
      "hasProduction": true,
      "productionExperimentId": "exp-001",
      "bestMetric": 0.956375203955669,
      "bestMetricName": "rmse",
      "bestMetricWeightColumn": null,
      "bestExperimentId": "exp-002",
      "experimentsNotCompared": 0,
      "lastPredictionAt": null
    }
  ],
  "dataFiles": null,
  "summary": {
    "modelsCount": 1,
    "totalExperiments": 2,
    "completedExperiments": 2,
    "failedExperiments": 0,
    "productionCount": 1
  }
}
```

`dataFiles` is populated only with `--verbose`, where it lists each conventional data file and
whether it is present:

```json
"dataFiles": [
  { "type": "Train",   "path": "datasets/train.csv",   "exists": true },
  { "type": "Test",    "path": "datasets/test.csv",    "exists": true },
  { "type": "Predict", "path": "datasets/predict.csv", "exists": true }
]
```

---

### `mloop analyze`

Exploratory data analysis, one aspect per subcommand, so you ask for the answer you want rather
than reading a whole report. `mloop info --analyze` runs the lot; these are the parts.

```bash
mloop analyze profile datasets/train.csv        # per-column statistics
mloop analyze correlation datasets/train.csv    # pairwise correlation
mloop analyze importance datasets/train.csv     # feature importance against the label
mloop analyze outliers datasets/train.csv       # outlier detection
mloop analyze distribution datasets/train.csv   # distribution shape per column
```

| Subcommand | What it reports |
|------------|-----------------|
| `profile` | Per column: missing count and rate, approximate unique values, type, and mean/stddev/min/max |
| `correlation` | The full pairwise correlation matrix, label included |
| `importance` | Permutation importance of each feature against the label |
| `outliers` | Isolation Forest: how many rows are outliers, and the score threshold used |
| `distribution` | Median, quartiles, IQR, skewness and kurtosis per column |

Every subcommand takes the same arguments:

| Argument / Option | Short | Description |
|-------------------|-------|-------------|
| `<data-file>` | | Path to the CSV dataset to analyze (defaults to the project's `data.train` if omitted) |
| `--label` | `-l` | Label/target column name (overrides `mloop.yaml`) |
| `--name` | `-n` | Model name to read label configuration from `mloop.yaml` |
| `--json` | | Output structured JSON instead of a console table |

```
Profile: 3 column(s); 0 constant, 0 high-null, 0 likely-index.

── Data Statistics ─────────────────────────────────────────────────────────────

╭────────┬─────────┬─────────┬──────────┬──────┬─────────┬────────┬────────┬────────╮
│ Column │ Missing │ Missing │ Unique   │ Type │    Mean │ StdDev │    Min │    Max │
│        │         │       % │ (sample) │      │         │        │        │        │
├────────┼─────────┼─────────┼──────────┼──────┼─────────┼────────┼────────┼────────┤
│ x1     │       0 │   0.00% │     ~297 │ Num  │  5.2371 │ 2.8823 │ 0.0390 │ 9.9820 │
│ x2     │       0 │   0.00% │     ~291 │ Num  │  2.5251 │ 1.4540 │ 0.0040 │ 4.9620 │
│ Label  │       0 │   0.00% │     ~299 │ Num  │ 10.6740 │ 9.1006 │ -7.778 │ 29.602 │
╰────────┴─────────┴─────────┴──────────┴──────┴─────────┴────────┴────────┴────────╯
```

The unique count is approximate (`~297`) because it is taken from a sample — exact cardinality on
a large file costs a full pass for a number that is used as a hint.

---

### `mloop info`

Display dataset profiling information with automatic encoding detection.

```bash
mloop info <data-file> [options]

# Options (v0.5.0+):
#   --label, -l <column>   Specify label column for proper column inference
#   --name, -n <model>     Model name to read label from mloop.yaml (default: "default")

# Basic usage
mloop info datasets/train.csv

# With explicit label column
mloop info datasets/train.csv --label IsError

# Read label from mloop.yaml model configuration
mloop info datasets/train.csv --name fraud-detector

# Output:
# [Info] Converted CP949 → UTF-8    # (automatic encoding conversion)
# Label column: IsError (from mloop.yaml (model: default))
#
# 📊 Dataset Information
# Rows: 1,000
# Columns: 15
#
# Column Details:
# - price (Numeric): Min=10.5, Max=999.9, Mean=245.3
# - category (Text): 5 unique values
# - date (DateTime): Range: 2024-01-01 to 2024-12-31
```

**Label Column Resolution** (v0.5.0+):
1. `--label` option (explicit override)
2. `mloop.yaml` model configuration (if in project)
3. First column (fallback)

**Encoding Support**:
- Auto-detects CP949, EUC-KR, UTF-8, UTF-16 encodings
- Converts to UTF-8 with BOM for ML.NET compatibility
- Korean text displayed correctly (e.g., 설비명, 공정번호)

### `mloop detect`

One-shot anomaly detection over a time series. No model, no training: hand it a CSV and it scores
every point, estimates the series' own period, and puts control limits around what it expected.

```bash
mloop detect data.csv                      # single-column CSV: the column is picked for you
mloop detect data.csv --column value       # name the column to monitor
mloop detect data.csv -c value --threshold 5      # only the sharper anomalies
mloop detect data.csv -c value --sensitivity 95   # wider band than 3 sigma
mloop detect data.csv -c value -o result.csv      # every point, with its limits
mloop detect data.csv -c value --json
```

| Argument / Option | Short | Default | Description |
|-------------------|-------|---------|-------------|
| `<data-file>` | | | Path to the CSV file containing the time series |
| `--column` | `-c` | | Value column to monitor (auto-selected when the CSV has a single column) |
| `--threshold` | | `3` | Score above which a point is an anomaly (> 0) |
| `--sensitivity` | | `99.73` | Coverage of the control limits in percent (0–100 exclusive; 99.73 = 3 sigma) |
| `--output` | `-o` | | Write the full per-point result to this CSV file |
| `--json` | | | Output the full result as JSON |

```
── One-Shot Anomaly Detection - series.csv ─────────────────────────────────────

Column: value  Points: 200  Anomalies: 2  Period: 24

╭───────┬─────────┬────────┬──────────┬───────────────┬───────────────╮
│ Index │ Value   │ Score  │ Expected │ Control Lower │ Control Upper │
├───────┼─────────┼────────┼──────────┼───────────────┼───────────────┤
│ 60    │ 37.1846 │ 15.565 │ 24.6774  │ 24.2309       │ 25.124        │
│ 140   │ 34.9938 │ 12.310 │ 23.2817  │ 22.8352       │ 23.7283       │
╰───────┴─────────┴────────┴──────────┴───────────────┴───────────────╯
```

**`Score`** is the point's saliency — how far it stands out from the spectral residual of the
series — and `--threshold` is compared against it directly. **`Control Lower`/`Control Upper`** are
the band the series itself implies at the requested coverage, so a point outside them is unusual
for *this* series rather than unusual in the abstract. **`Period`** is estimated, not supplied, and
is `null` when the series has none.

An index marked **`*`** sits near an end of the series, where the score is least reliable: the
transform treats the series as circular, so its first and last few points are scored partly against
each other. The mark is a position, not a verdict — a real spike there is still an anomaly — but a
flag that appears only at an end is worth a second look. `--output` and `--json` carry it per point
as `NearEdge` / `nearEdge`.

The table lists only the anomalies. `--output` writes every point with its score, expected value
and limits — that is the file to chart. `--json` carries the same thing:

```json
{
  "column": "value",
  "totalPoints": 200,
  "anomalyCount": 2,
  "period": 24,
  "outputFile": null,
  "points": [
    {
      "index": 60,
      "value": 37.1846,
      "isAnomaly": true,
      "score": 15.565493686292202,
      "expectedValue": 24.677441826232204,
      "controlLower": 24.230905650284555,
      "controlUpper": 25.123978002179854,
      "nearEdge": false
    }
  ]
}
```

**Runs everywhere.** Detection is computed by the statistics engine MLoop already uses for
analysis, not by ML.NET's FFT native — so unlike `forecasting` and `time-series-anomaly` *training*
it needs no OpenMP runtime on Linux and works on Apple silicon.

> **Changed in the version after 0.32.0.** `--threshold` is now the score a point must exceed
> (greater than 0, default 3); it used to be a value in [0, 1] with a default of 0.3, on a
> different scale. `--sensitivity` is now the coverage of the control limits in percent. `--period`
> was removed — the score assumes no period, so supplying one could not change the result; passing
> it now explains that instead of failing to parse. Each point carries `score`, `expectedValue` and
> `controlLower`/`controlUpper`, where it used to carry `marginLower`/`marginUpper` and
> `residualSigma`.

---

### `mloop evaluate`

Evaluate model performance on test data.

```bash
mloop evaluate <experiment-id> <test-data>

# Example — the label column comes from the experiment's own training config, not an argument
mloop evaluate exp-003 datasets/test.csv

# Output:
# 📊 Evaluation Results
# R²: 0.9456
# RMSE: 13.21
# MAE: 9.87

# The same metrics for each value of a column of the test data
mloop evaluate exp-003 datasets/test.csv --slice-by site

# Rows weighted by a column of the test file (e.g. 1/π of a recheck sample)
mloop evaluate exp-003 datasets/rechecked.csv --weight inv_pi

# A staging candidate against production, on the same rows and weights
mloop evaluate exp-004 datasets/rechecked.csv --weight inv_pi --against production
```

A classifier's evaluation also shows its **confusion matrix** — rows are the actual classes, columns
the predicted ones, in the label's own words (the binary class names the model was trained on, the
multiclass label's values). Each cell is a weighted sum under `--weight`. Above 10 classes it stays in
`--json`, which always carries it as `confusion` (`classes`, `counts[actual][predicted]`, `total`) —
for each slice too. Any class-set measure is read from it: of the rows labeled with one of a set of
classes, the share answered with one of another set — the rate at which a class that must be caught
is let through as a harmless one, without the evaluator knowing which classes those are.

`--weight <column>` counts each test row by that column, scored as weighted training scores its own
test rows. An experiment trained with a weight uses that column by default when the test file has
it; when the file does not, the test metrics are unweighted and `evaluate` says that the training and
test columns then describe different distributions. `--json` carries `weightColumn` (`null` when
unweighted); `--slice-by` slices are weighted the same way.

`--slice-by <column>` adds one row per value of that column (any column of the test file — it does not
have to be a feature). Each slice is evaluated as a test file of its own, so its numbers mean what the
overall ones mean; a slice under 30 rows is marked, because its metrics move a lot between samples.
A slice holding one class still gets accuracy, precision and recall from its counts — AUC is not
defined there and is shown as `-`. More than 50 values is refused. `--json` adds `sliceBy` and
`slices` (`value`, `rows`, `metrics`, `error`, `small`).

`--against <experiment|production>` scores a second model on the same rows, with the same weights,
and compares the two. Its metrics go beside the candidate's, and — for a classifier or a regressor —
the two are paired row by row: each row's loss (a wrong class counts 1, or the squared error) under
both models, the weighted mean of each, their difference (candidate − baseline, so negative favors
the candidate) and a 95% paired bootstrap interval for it. Two models' metrics on one sample say how
good each is there; the interval says whether the gap between them is more than which rows happened
to be drawn. The verdict follows the interval:

```
Paired over 1,200 rows, by error rate: 0.0810 vs 0.0950 — difference -0.0140 (95% interval -0.0240 to -0.0042) — this experiment is better on these rows.
```

A model trained for another task or label is refused, as is the candidate itself. `--json` adds
`comparison` — `against`, `metrics`, and `paired` (`measure`, `rows`, `candidate`, `baseline`,
`difference`, `low`, `high`, `verdict`: `CandidateBetter`, `BaselineBetter` or `Indistinguishable`),
`null` with a reason in `notPaired` for tasks with no per-row loss.

`POST /evaluate` (admin) takes the same choices — `{ "testDataPath", "experimentId", "name", "weight",
"against", "sliceBy" }` — and prepares the file the same way: read as a table (Parquet, Excel, JSON, TSV), checked
against the experiment's schema (a file missing the model's columns is a `400` naming them in `missingColumns`), the
rows weighed the same way (the experiment's training weight by default, and a `warnings` entry when the file lacks
it). It returns `weightColumn`, `metrics`, `confusion`, `comparison`, and with `sliceBy` the per-value `slices`. A
relative `testDataPath` is the project's.

### `mloop validate`

Validate `mloop.yaml` — task types, column types, metrics, and prep steps.

```bash
mloop validate

# Detailed results
mloop validate --verbose

# Output:
# ✅ Configuration is valid
# ❌ models.default.task: Invalid task type 'regressoin'. Valid values: binary-classification, ...
```

---

## Deployment Commands

### `mloop serve`

Start REST API server for model serving.

```bash
# Start server on default port (5000)
mloop serve

# Custom port and host
mloop serve --port 8080 --host 0.0.0.0

# Run in background
mloop serve --detach

# API Endpoints:
# GET  /health              - Health check
# GET  /info                - Production model information
# GET  /models              - List all models
# POST /predict             - Make predictions (?experiment=<exp-id>: a candidate answers instead
#                              of production — a shadow prediction; production does not move)
# POST /embed               - Images to vectors with a pretrained encoder (see `mloop embed`)

# Authentication: every endpoint except /health requires a JWT bearer token. Mint one with the
# CLI (it signs with the same dev defaults the server validates out of the box), then attach it:
export MLOOP_TOKEN=$(mloop token --quiet)              # add --role admin for write endpoints
# A programmatic caller must issue/refresh this token itself — the server has no token endpoint.

# Example prediction request:
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $MLOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'

# Batch prediction:
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $MLOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 3.0, "feature2": 4.0}
  ]'
# A call without the Authorization header returns 401 with a body hint pointing to `mloop token`.
```

**Features**:
- One server serves one project — the directory it starts in (or `MLOOP_PROJECT_ROOT`); several models in it by `?name=`
- Auto-loads production model from registry
- Training jobs (`POST /train` → `GET /jobs/{id}`) run one at a time in the background while predictions keep
  answering. Each job is recorded in `.mloop/jobs/`, so a restarted server still knows it: a job that was running when
  the server stopped reads `Failed` (interrupted), and a queued one is trained when the server starts again
- A job promotes its experiment as `mloop train` does — when it beats production and passes the gates — unless the
  request sends `"promote": false` (`--no-promote`), for an application that decides promotion itself and then calls
  `POST /promote`. The completed job says which: `promoted`, and `keptInStagingReason` when it stayed in staging (the
  quality gate, the gate policy, production as good or better, a post-train hook, or `"promote": false`). The job
  reads `Completed` only after that decision, so production is in place by the time a poller sees it
- The completed job's `warnings` are the performance diagnostics `mloop train` prints and reports as `--json`
  warning events — a held-out score near-perfect on 500 rows or more (check for a leaked column), an accuracy that
  hides one class; an empty list when there are none
- Swagger/OpenAPI documentation at `/swagger`
- Single and batch predictions
- JSON request/response format
- CORS enabled for web clients

### `mloop token`

Every endpoint of the served API except `/health` needs a bearer token, and the write endpoints
need one carrying the `admin` role. This issues one.

```bash
mloop token                                  # a caller token, 24h
mloop token --role admin --subject ci        # for promote/train/evaluate
mloop token --expires-hours 1                # short-lived
export MLOOP_TOKEN=$(mloop token -q)         # just the token, for scripts
```

| Option | Short | Description |
|--------|-------|-------------|
| `--role` | `-r` | Role claim — use `admin` for the write endpoints |
| `--subject` | `-s` | Subject (`sub`) claim identifying the caller |
| `--expires-hours` | | Token lifetime in hours |
| `--key` | | Override signing key (must match the server's `Jwt:Key`, ≥32 chars) |
| `--quiet` | `-q` | Print only the raw token |

```
JWT issued (subject=ci, role=admin, expires in 24h)

eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjaSIsInJvbGUiOiJhZG1pbiIsIm5iZiI6...

Use it as a bearer token:
  curl -H "Authorization: Bearer <token>" http://localhost:5000/info
  export MLOOP_TOKEN=$(mloop token -q)   # for clients reading MLOOP_TOKEN
```

The signing key must be the one the server uses. In development that is the default key, so a
token issued here works against a locally started `mloop serve` with no further setup; in
production, where `Jwt:Key` is set, pass the same value with `--key` or the server will reject the
token it did not sign.

---

### `mloop compare`

Puts experiments side by side — every metric they recorded, plus the configuration that produced
them — and says which one wins on the metric the training actually optimized.

```bash
mloop compare exp-001 exp-002      # two named experiments
mloop compare --name churn --best 3  # the top three for a model
mloop compare --name churn -m rmse   # ranked by a metric you choose
```

| Argument / Option | Short | Description |
|-------------------|-------|-------------|
| `experiments` | | Experiment IDs to compare (e.g. `exp-001 exp-002`) |
| `--name` | `-n` | Model name (required when comparing all experiments) |
| `--best` | `-b` | Compare the top N experiments by metric (default: all completed) |
| `--metric` | `-m` | Sort by a specific metric (default: the metric the experiments optimized) |
| `--metrics-file` | | Rank and select from a JSON file of supplied metrics, with no local project |
| `--json` | | Output the result as JSON |

```
── Experiment Comparison ───────────────────────────────────────────────────────

╭────────────────────────┬─────────────────────────┬─────────────────────────╮
│ Metric                 │         exp-001         │         exp-002         │
│                        │      (Production)       │                         │
├────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Model                  │         default         │         default         │
│ Task                   │       regression        │       regression        │
│ Status                 │        Completed        │        Completed        │
│ Timestamp              │ 2026-09-20 12:11 +09:00 │ 2026-09-20 12:12 +09:00 │
│ Label Column           │          Label          │          Label          │
│                        │                         │                         │
│ mae                    │         1.4643          │         1.4643          │
│ r_squared              │         0.9564          │         0.9564          │
│ rmse                   │         1.8352          │         1.8352          │
╰────────────────────────┴─────────────────────────┴─────────────────────────╯

Compared 2 experiments
Best experiment exp-001 is already in production (r_squared: 0.9564)
```

Which direction counts as better comes from the metric, not from the order of the columns — a
lower `rmse` wins and a higher `r_squared` wins, and the closing line names the metric it decided
on.

**`--metrics-file`** is for callers that hold the metrics themselves — a distributed run, or a
system of record outside this project. It ranks what it is given and reports the winner without
reading `.mloop/` at all, which is why it pairs with `--json`.

---

### `mloop logs`

What the served and local models have actually been predicting. Entries are written by
`mloop predict --log` and by the API.

```bash
mloop logs                          # most recent entries
mloop logs --limit 3                # fewer
mloop logs --from 2026-09-01 --to 2026-09-20
mloop logs --json
```

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-l` | Maximum number of entries to show |
| `--from` | | Start date filter (`yyyy-MM-dd`) |
| `--to` | | End date filter (`yyyy-MM-dd`) |
| `--json` | | Output in JSON format |

```
── Prediction Logs ─────────────────────────────────────────────────────────────

╭──────────────────┬─────────────┬─────────┬────────────┬─────────────┬────────╮
│ ID               │ Timestamp   │ Model   │ Experiment │ Input       │ Output │
│                  │ (+09:00)    │         │            │ (summary)   │        │
├──────────────────┼─────────────┼─────────┼────────────┼─────────────┼────────┤
│ 2869808376524814 │ 2026-10-21  │ default │ exp-001    │ x1=0.899    │ yes    │
│                  │ 21:45:51    │         │            │ (+1 more)   │        │
│ 8dc968cd6ec741b0 │ 2026-10-21  │ default │ exp-001    │ x1=0.846    │ yes    │
│                  │ 21:45:51    │         │            │ (+1 more)   │        │
╰──────────────────┴─────────────┴─────────┴────────────┴─────────────┴────────╯

Showing 2 entries
```

The ID column is the prediction's id — what `mloop feedback add --prediction-id` takes.

The timestamps are in your own zone, named once in the column heading rather than repeated on
every row. With nothing logged yet the command says so and exits 0:

```
No prediction logs found.
Use mloop predict --log to log predictions.
```

#### Retention — `mloop logs prune`

Logs and feedback are kept until something removes them. A service that logs every prediction should set a period
in `mloop.yaml`:

```yaml
retention:
  days: 30      # today (UTC) and the 30 days before it are kept
```

`mloop serve` then removes older days when it starts and once a day while it runs, and `mloop logs prune` does the
same on demand (for a schedule of your own, or a project that is not served):

```bash
mloop logs prune --dry-run          # what retention.days would remove
mloop logs prune --days 14          # a period of its own
mloop logs prune --json
```

Both stores are written one file per model per UTC day, so whole days leave and nothing is rewritten; files whose
name is not a day are left alone. Logs and feedback share the period, since a retraining set joins them — and
feedback is matched to its logged prediction, so the period is also how late feedback can arrive. Without `retention:` (or with `days` below 1, which `mloop validate` reports) nothing is removed.

---

### `mloop feedback`

Once the actual value of a logged prediction is known, record it. Feedback is what `mloop feedback
metrics` and `mloop trigger check` measure the model with, and what a retraining dataset is built from.

```bash
mloop feedback add -p 2869808376524814 -v no -s reviewer-llm -w 4   # the actual value, where it came from, its weight
mloop feedback list [--model <name>] [--json]
mloop feedback metrics [--model <name>] [--json]
```

| Option (`add`) | Short | Description |
|--------|-------|-------------|
| `--prediction-id` | `-p` | The prediction's id (the ID column of `mloop logs`) |
| `--actual-value` | `-v` | The actual value |
| `--source` | `-s` | Where the value came from — free text (`human`, `reviewer-llm`, …) |
| `--weight` | `-w` | How much the row counts in training and evaluation built from feedback: for a row reviewed with probability *p*, 1/*p*. Default 1; finite and not negative |

Recording feedback again for the same prediction corrects it — the latest entry stands. When any entry
carries a weight, `feedback metrics` counts each prediction by its weight (`Accuracy (weighted)`, `weighted`
in `--json`): when some answers are checked more often than others, counting the checked rows alike misstates the
model's accuracy.

---

### `mloop sample`

`mloop sample create --model <name>` turns the logged predictions and their feedback into a training file:

```bash
mloop sample create --model default --rows 1000 --strategy feedback-priority -o feedback.csv
mloop train --data datasets/train.csv feedback.csv --weight weight   # the original rows count 1
```

```
x1,x2,churn,weight
0.846,0.383,yes,1
0.899,0.925,no,4
0.373,0.749,,1
```

- The logged inputs are the features, and the actual value is written under the **model's label column** (from its
  production experiment, else `label:` in `mloop.yaml`). A row without feedback has an empty label, and training drops it.
- The model's own answer and the feedback source are **not** written. Every column in the file is a feature to the
  next training, and a model given its own previous answer learns to copy it.
- A weight column is written when the model trains weighted (its own column name), or when any feedback carries a
  weight (`weight`, or `feedback_weight` if an input already has that name). Train with `--weight <column>`.
- `train --data a.csv b.csv` merges the files. A file without the weight column counts its rows 1, so the original
  training data and weighted feedback merge without editing either file.
- Strategies: `random` (default), `recent`, `feedback-priority` (rows with feedback first), `stratified` (by the
  model's answer), `low-confidence` (the least confident answers — the rows most worth labeling next).
- Over HTTP (`mloop serve`): `POST /samples` `{"name": "default", "rows": 1000, "strategy": "feedback-priority"}` writes
  the same file and answers with its `path`, `labeledCount` and `weightColumn`; `POST /train` takes it in
  `additionalDataFiles`, merged after `dataFile` the same way. A job prepares its data as `mloop train` does — rows
  without a label dropped, `maxRows` applied, a classification on one class refused, `balance` (as `--balance`)
  balancing the train part after the test part is split off — and a request without `timeLimitSeconds` or
  `testSplit` leaves them to `mloop.yaml`. A `balance` the run cannot honour (not `auto`/`none`/a ratio, a task that
  is not classification, a weight column) is refused with 400 before a job exists.
- `mloop sample create --from <csv>` samples a plain CSV instead (`random`, `head`, `stratified --label <col>`);
  `mloop sample stats --model <name>` reports how many logged predictions have feedback.

---

### `mloop drift`

How far the production model's recent inputs — and its answers — have moved from the data it was
trained on. Training records each feature's distribution beside the model (`drift-profile.json` —
numeric columns by decile, categorical ones by their most common values, missing values apart), and,
for a classifier or a regressor, the distribution of the model's own answers over the same rows
(the predicted class, or the predicted value by decile). This compares the production model's own
logged predictions with both, by the population stability index (PSI). Nothing runs in the
background — it is computed when you ask. A free-text column is profiled through two descriptors,
reported as two more columns: `<column> (length)` (characters, by decile) and `<column> (unseen words)`
(the share of each value's words that training never saw — none, up to 10 %, 25 %, 50 %, 75 %, more).
The second is the one that moves when the text is about something new; it is measured against
training rows the vocabulary was not built from, so ordinary new text reads as stable. Words are runs
of letters and digits; Chinese, Japanese and Thai characters count one by one.

```bash
mloop drift                         # the most recent 5,000 logged predictions
mloop drift --rows 1000
mloop drift --slice-by site         # also per value of an input column
mloop drift --json
```

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Model name |
| `--rows` | | How many of the most recent logged predictions to compare (at least 100) |
| `--slice-by` | | Report drift per value of this input column (values with 100+ predictions) |
| `--json` | | Output in JSON format |

```
   Input drift — default (exp-001, 400 predictions)
╭────────┬───────┬─────────────┬─────────────────────╮
│ Column │   PSI │    Level    │ Most moved          │
├────────┼───────┼─────────────┼─────────────────────┤
│ x      │ 0.781 │ significant │ > 13.629: 10% → 45% │
│ color  │ 0.200 │  moderate   │ red: 34% → 54%      │
│ flag   │ 0.001 │   stable    │ true: 54% → 52%     │
╰────────┴───────┴─────────────┴─────────────────────╯
                          By site
╭───────┬──────┬─────────┬─────────────┬───────────────────╮
│ Value │ Rows │ Max PSI │    Level    │ Most moved column │
├───────┼──────┼─────────┼─────────────┼───────────────────┤
│ A     │  200 │   4.745 │ significant │ x                 │
│ B     │  200 │   0.046 │   stable    │ x                 │
╰───────┴──────┴─────────┴─────────────┴───────────────────╯
Answers (default's own predictions): PSI 0.205 moderate — yes: 24% → 45%
PSI under 0.1 is stable, 0.1–0.25 moderate, 0.25+ significant.
```

**Answers** compares what the model said over the window with what it said over its training rows.
It is reported apart from the inputs and does not raise their level: answers that move while every
input holds still say something different — the mix of what the model is asked shifted inside the
ranges it knows, which the per-column PSI cannot see. In `--json` it is `drift.prediction` (the
same fields as a column, under the name `(prediction)`), `null` for a model trained before answers
were profiled or a task whose answer is not a class or a value. The reference is the model's
answers on rows it was trained on, so a model that memorized them reads slightly sharper there than
on new traffic.

**Most moved** is the bin whose share changed most — a decile range, a category, `(other)` for
values training never saw, or `(missing)`. Only predictions of the model now in production count:
an earlier model's traffic describes another model. A model
trained before profiles were recorded, or a task that reads no feature columns (time series,
recommendation, deep learning), has none — the command says so; retrain and promote to start.

The same measurement is a retraining condition: `mloop trigger check` and `GET /trigger` report
`input_drift` (met at PSI ≥ 0.25 on any column) with every column's PSI under `drift` in `--json`.
`GET /drift?name=&rows=&sliceBy=` returns what `mloop drift --json` does.

---

### `mloop trigger`

Asks whether the model should be retrained, from what production has seen since it was trained.

```bash
mloop trigger check                         # the default conditions for model 'default'
mloop trigger check --name churn --accuracy 0.8 --feedback 50
mloop trigger check --json                  # shouldRetrain, recommendedAction, conditions[]
```

| Condition | Met when | Measured from |
|-----------|----------|---------------|
| `accuracy_threshold` (`AccuracyDrop`) | accuracy on feedback < 70% (`--accuracy`) | `mloop feedback add` / `POST /feedback` |
| `feedback_count` (`FeedbackVolume`) | ≥ 100 feedback entries (`--feedback`) | the same feedback |
| `Scheduled Retraining` (`TimeBased`) | the model is older than 30 days | the production model's training time |
| `input_drift` (`DataDrift`) | PSI ≥ 0.25 on any column, or on the model's answers | logged predictions against the training profile (`mloop drift`) |

Any condition met recommends retraining. A condition that could not be measured — no feedback yet, fewer
than 100 logged predictions — is not met and shows `-` (`currentValue: null` in `--json`): an accuracy
nobody measured is not 0%. `GET /trigger?name=` answers the same evaluation over HTTP; both read one set of
conditions (`RetrainingDefaults`).

### `mloop export`

`mloop export onnx` writes a trained model as an [ONNX](https://onnx.ai) graph, for a runtime that cannot load ML.NET —
a Native AOT executable, an edge agent, another language (ONNX Runtime is available for most).

```bash
mloop export onnx                      # the production model
mloop export onnx exp-004 --name churn # a named experiment
mloop export onnx --data sample.csv    # read the input columns from another file
mloop export onnx -o ./edge --json     # write elsewhere; machine-readable result
mloop export onnx --verify             # run the graph and compare it with the model
```

Two files land in the experiment's directory (or `--output`):

- `model.onnx` — the graph. Its inputs are the model's input columns as ML.NET reads them: numeric
  columns arrive merged into one `float32` vector (`Features`), a string column as a `string` tensor.
  Its outputs are the prediction columns with `.output` appended (`PredictedLabel.output`,
  `Score.output`, `Probability.output`).
- `model.onnx.json` — the manifest: for each input, the data columns that fill it in slot order;
  the outputs; the classes and how to read the answer (`label.decision`); the experiment's metrics and
  weight column; SHA-256 of the graph and of the `model.zip` it came from.

The graph is the trained model. A calibration (`mloop calibrate`) and a decision threshold
(`mloop threshold`) beside it are not in the graph: the manifest carries them (`calibration`,
`decisionThreshold`) and `label.decision` says how to apply them, in the order `mloop predict` does —
calibrate the probability, then compare it with the threshold (a calibrated model without one answers
at 0.5). Binary without either: `PredictedLabel` true is the second class, `Score > 0`; multiclass:
`Score` per class in `classes` order.

The model was saved without an input schema, so the export reads one from data — by default the
experiment's training file, through the same loader prediction uses. A column the model does not read
(an identifier, a weight) is not an input. An input the graph declares but no output needs — a
multiclass label — is listed under `unusedInputs`: a runtime still requires it, so feed any value.

**`--verify`** runs the graph in ONNX Runtime on up to 1,000 rows of the data and compares every
output with the model's: `Score` and `Probability` within 1e-4, `PredictedLabel` exactly. When the
model has a calibration or a threshold, it also applies `label.decision`'s rule to the graph's outputs
and compares that with `mloop predict`'s answer (`verification.layered`). A graph that disagrees is deleted and the command exits 1; the manifest keeps the comparison under
`verification`. Use it: ML.NET's converter can report success and write a wrong graph — a text
n-gram step loaded from a saved model exports counting only its longest n-grams, which answered
differently on every row. ONNX Runtime's native library is not part of MLoop; install it once with
`mloop runtime install onnxruntime` (about 120 MB download).

Binary classification, multiclass classification and regression are supported. Numeric columns
and category columns (one-hot encoded) export; **a text column does not** — ML.NET's text
featurization (`FeaturizeText`) has no ONNX form — unless the model was trained **portable**
(`--portable`, see `mloop train`), which featurizes text with steps ONNX computes exactly. A model
that is not is refused with its string columns named, and nothing partial is written. If the column
actually holds codes or categories, declare it `type: categorical` under the model's `columns:` and
retrain; otherwise retrain with `--portable`.

---

### `mloop docker` (v0.4.0+)

Generate Docker configuration files for containerized deployment.

```bash
# Generate all Docker files
mloop docker

# Generated files:
# - Dockerfile           Multi-stage build with MLoop API
# - docker-compose.yml   Complete service configuration
# - .dockerignore        Exclude unnecessary files

# Build and run
docker build -t my-ml-model .
docker run -p 5000:5000 my-ml-model

# Or use docker-compose. It serves in the Production environment, where the API refuses to
# start on the default signing key, so the key is required rather than defaulted:
JWT_SIGNING_KEY=<random secret, 32+ characters> docker compose up -d
```

**How the generated image is put together**: the CLI is a dotnet tool, so the build stage exists
only to acquire it, and the runtime stage copies the installed tool across. Nothing installs a .NET
distribution — the `aspnet` image already carries the runtime the tool needs. Your model artifacts,
`.mloop/` and `mloop.yaml` are copied in, so the image serves without a mounted volume; the compose
file additionally mounts `models/` read-only so a promoted model can be picked up without rebuilding.

**docker-compose.yml Features**:
- Health check that issues a GET (a HEAD-only probe is answered 405 by a GET route, which reads to
  the orchestrator as a dead container even while the service is healthy). A successful probe is
  logged at Verbose, so the probes do not fill the server's log; a failing one is an error
- Volume mounts for models
- Environment variables for API configuration, including the required signing key
- Restart policy for production

### `mloop runtime`

Deep-learning tasks and ONNX verification need native libraries that are not shipped with the tool; they
are downloaded on demand into the user's profile, once per machine.

```bash
mloop runtime list                  # each runtime, its size, whether it is installed, the tasks that need it
mloop runtime list --json
mloop runtime install tf            # TensorFlow CPU — image-classification
mloop runtime install torch         # libtorch CPU — object-detection, text-classification, sentence-similarity, ner, question-answering
mloop runtime install onnxruntime   # ONNX Runtime CPU — `mloop export onnx --verify`
mloop runtime remove tf
```

A task whose runtime is missing stops before training with the `runtime install` line to run.

### `mloop encoder`

Pretrained image encoders — graphs MLoop runs but does not train — for `mloop embed` and `POST /embed`. They are kept
outside any project, in `~/.mloop/encoders/` (or the directory `MLOOP_ENCODERS` names), one directory per encoder
with its definition and its ONNX graph. They are not experiments: nothing promotes, gates or retrains them.

```bash
mloop encoder add encoder.json --model model.onnx   # install; records the graph's SHA-256
mloop encoder list                                  # id, version, dimension, input size, SHA-256
mloop encoder list --json
mloop encoder remove my-encoder
```

The definition names the graph's input and output, the vector's length, how a vector is read from the output, and
the image preprocessing the encoder was trained with:

```json
{
  "id": "my-encoder",
  "version": "1",
  "dimension": 384,
  "preprocessing": { "shortestEdge": 256, "cropSize": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] },
  "inputName": "pixel_values",
  "outputName": "last_hidden_state",
  "pooling": "first-token"
}
```

`pooling` is `first-token` for an output of `[images, tokens, dimension]` (a vision transformer's class token) or
`none` for `[images, dimension]`. An image is resized so its shortest edge is `shortestEdge` (bicubic, computed as
Pillow computes it — the resampling most published image processors use), cropped to a `cropSize` square at the
centre, scaled to [0, 1] and normalised per channel. Adding a `sha256` to the definition makes `add` refuse a graph
with other contents. Encoders run on ONNX Runtime (`mloop runtime install onnxruntime`).

### `mloop embed`

```bash
mloop embed a.png b.jpg                    # one vector per image (first values shown)
mloop embed a.png b.jpg --json             # {"encoder": {id, version, sha256, dimension, normalized}, "vectors": [[…], …]}
mloop embed *.png --encoder my-encoder     # when more than one encoder is installed
```

Vectors are L2-normalised, so the cosine similarity of two is their dot product. The encoder's id and SHA-256 come
with every answer: vectors from different weights are not comparable, so whoever keeps vectors re-embeds when they
change. At most 64 images per call. The same is served as `POST /embed` (`mloop serve`): a JSON body
`{"images": ["<base64>", …]}` or image files as multipart form data, `?encoder=` to choose one. It takes the images'
contents, not paths. An image that cannot be decoded is a 400 naming its position (`index`). A missing encoder or
runtime is a 409 saying how to install it. More than 64 images is a 413.

In a container made by `mloop docker`, embedding is opt-in. The Dockerfile has a commented
`RUN mloop runtime install onnxruntime` line, and `docker-compose.yml` has a commented `MLOOP_ENCODERS` variable and a
mount of the host's encoders. The image carries neither the runtime nor any encoder until those lines are uncommented.

### `mloop update`

```bash
mloop update --check   # is a newer version published?
mloop update           # install it (runs `dotnet tool update --global mloop`; its exit code is passed through)
```

---

## Data Preprocessing

MLoop integrates with **FilePrepper** for high-performance CSV preprocessing.

### `mloop prep` — preprocessing declared in `mloop.yaml`

Steps under a model's `prep:` run on every `mloop train`, in order. `prep plan` edits the declaration (no data
changes); `prep run` applies it to a file so you can look at the result.

```bash
mloop prep plan --set normalize:z-score --columns age,income   # add or replace a step
mloop prep plan --set fill-missing:mean --columns income
mloop prep plan --remove normalize                             # remove steps of a type
mloop prep plan --list                                         # the plan, with each step's leakage class
mloop prep run --dry-run                                       # the steps that would run
mloop prep run -i datasets/train.csv -o datasets/train_prep.csv
```

Step types: `fill-missing` (mean · median · mode · constant · forward · backward), `normalize`, `scale`,
`remove-columns`, `rename-columns`, `drop-duplicates`, `extract-date`, `parse-datetime`, `filter-rows`,
`add-column`, `parse-korean-time`, `parse-excel-date`, `rolling`, `resample`, `sample`, `explode`, `split-list`.

`explode` and `split-list` reshape list columns — a list per cell, the way a Parquet or JSON list is read. A file
holding a group per row (a query with its documents' grades and feature vectors) becomes one row per document with
one column per feature:

```yaml
prep:
  - type: explode          # one row per element; the lists of a row are paired by position
    columns: [labels, features]
  - type: split-list       # features holding [0.5, 1] becomes features.0, features.1
    column: features
```

The **Leakage** column says where a step runs. A statistic learned from the data (`normalize`, `scale`,
`fill-missing:mean`) is fitted inside training on the training rows only, when the task supports it
(`✓ fold-safe`); otherwise it is applied to the whole file before the train/test split and the evaluation
keeps a leak (`⚠ leak`). Steps that learn nothing (renames, filters, date parts) run on the CSV
(`ℹ csv-stage`). `POST /train` refuses a model whose `prep:` it would have to skip.

You do not need `normalize` or `scale` for the model's sake: the trainer search already scales the features ahead of
its linear trainers, and its trees do not need it. Declare them when the scaled values themselves matter to you.

**Missing numbers.** An empty numeric field is read as missing — never as zero, which would be a real value inside
most columns' range. The trainer search replaces a missing number by its column's mean, fitted on the training rows
only, and for every column that had missing values in training adds an indicator feature (`<column>.missing`, 1
where the value was missing), so a model can learn from the missingness itself; the saved model, `predict`,
`serve` and the ONNX export do the same. Clustering, anomaly detection and ranking replace a missing number by the
mean. A time series refuses a gap — dropping the row would close up the time around it — and names the column:
fill it first, e.g. `mloop prep plan --set fill-missing:forward --columns <series>`. Declare `fill-missing` yourself
when another fill (median, a constant) is the one you mean.

### `mloop features`

Declares which columns are features — policy in `mloop.yaml` (`columns: {name: {type: ignore}}`), no data
change. The automatic exclusions (dates, identifiers, constants — see `mloop train`) apply on top.

```bash
mloop features select --drop customerID,notes   # these are not features
mloop features select --keep age,income,region  # only these (and the label) are
mloop features select --reset                   # remove every 'ignore' override
mloop features select --drop notes --name churn --json
```

`--drop` checks the names against the training data's header and skips one that is not there, saying so —
a misspelt name would otherwise be recorded while the real column stayed a feature. `--keep` needs the
training data, to know what "all others" are. Use one of the three at a time.

### Quick Examples

```bash
# DateTime feature extraction
fileprepper datetime -i data.csv -o out.csv -c "OrderDate" -m features --header

# Expression calculations
fileprepper expression -i data.csv -o out.csv -e "total=price*quantity" --header

# Column selection
fileprepper select -i data.csv -o out.csv -c "id,price,quantity" --header

# Filtering
fileprepper filter -i data.csv -o out.csv -e "price>100" --header
```

### Performance

FilePrepper is **20x faster** than pandas for common preprocessing tasks:

| Operation | pandas | FilePrepper | Speedup |
|-----------|--------|-------------|---------|
| DateTime extraction | 2.5s | 0.12s | **20.8x** |
| Expression eval | 3.1s | 0.15s | **20.7x** |
| Filtering | 1.8s | 0.09s | **20.0x** |

### Preprocessing Pipeline

Create sequential preprocessing scripts:

```
.mloop/scripts/preprocess/
├── 01_datetime.cs
├── 02_calculate.cs
└── 03_normalize.cs
```

Each script receives the output of the previous script as input.

**Example Script** (`01_datetime.cs`):
```csharp
using MLoop.Extensibility.Preprocessing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class DateTimePreprocessor : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        // Read input CSV
        var data = await context.Csv.ReadAsync(context.InputPath);

        // Add datetime features
        var processed = data.Select(row =>
        {
            var newRow = new Dictionary<string, string>(row);
            if (row.ContainsKey("OrderDate") && DateTime.TryParse(row["OrderDate"], out var dt))
            {
                newRow["Year"] = dt.Year.ToString();
                newRow["Month"] = dt.Month.ToString();
                newRow["DayOfWeek"] = ((int)dt.DayOfWeek).ToString();
            }
            return newRow;
        }).ToList();

        // Write output CSV
        var outputPath = Path.Combine(context.OutputDirectory, "01_datetime.csv");
        await context.Csv.WriteAsync(outputPath, processed);

        return outputPath;
    }
}
```

---

### `mloop preprocess`

Runs the C# preprocessing scripts of `.mloop/scripts/preprocess/` on a data file — the scripts `mloop train` also runs,
after the `prep:` steps of `mloop.yaml`. This is the place for logic the step types do not cover. The scripts run in
file-name order (`01_*.cs`, `02_*.cs`, …), each reading the previous one's output.

```csharp
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class DropLowA : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var rows = await context.Csv.ReadAsync(context.InputPath);
        var kept = rows.Where(r => double.Parse(r["a"], CultureInfo.InvariantCulture) >= 0.1).ToList();
        context.Logger.Info($"Kept {kept.Count} of {rows.Count} rows");
        return await context.Csv.WriteAsync(System.IO.Path.Combine(context.OutputDirectory, "01_out.csv"), kept);
    }
}
```

```bash
mloop preprocess                          # datasets/train.csv
mloop preprocess datasets/raw.csv -o datasets/clean.csv
```

```
✓ Auto-detected: datasets/train.csv
🔄 Running 1 preprocessing script(s)...
  📝 [1/1] 01_drop_low_a.cs
Kept 2688 of 3000 rows
    ✅ Output: 01_out.csv
✅ Preprocessing complete
✓ Output: .mloop/temp/preprocess/01_out.csv
✓ Copied to: datasets/clean.csv
```

Without `-o` the result stays in `.mloop/temp/preprocess/`. A script that does not compile is reported with the
compiler's errors. `--incremental` runs a different workflow — pattern detectors that propose cleaning rules, with
prompts to approve them (`--skip-hitl`, `--auto-approve`); the scripts above do not run in that mode.

## Extensibility

MLoop supports code-based extensibility through hooks and custom metrics.

### Hooks

Custom logic before and after training. A hook is a C# file in a folder named for its stage —
`.mloop/scripts/hooks/pre-train/` or `.mloop/scripts/hooks/post-train/` — compiled when training
starts (and cached). `mloop train` and `POST /train` run the same hooks, each once. A file directly in
`.mloop/scripts/hooks/` belongs to no stage and does not run — training warns and names it.

| Stage | What the hook gets | `HookResult.Abort` |
|-------|--------------------|--------------------|
| `pre-train` | `DataView` (the training data) · `Metadata["DataFile"]` · `LabelColumn` · `TaskType` · `ModelName` · `ExperimentId` | stops the run before training |
| `post-train` | `Metrics` (name → value, as the run reports them) · `Model` · `Metadata["ModelPath"]` · `BestTrainer` | the model is saved but kept in staging — not promoted automatically; `mloop promote <exp-id>` promotes it anyway |

A script that does not compile is reported with the compiler's errors and skipped.

### `mloop new`

Writes a hook from a template, in the folder for its stage:

```bash
mloop new hook row-guard pre-train                           # validation template
mloop new hook gate post-train --template performance-gate   # hold the model to a metric floor
```

| Template | Writes |
|----------|--------|
| `basic` | a hook that logs and continues — the place to start your own |
| `validation` | (pre-train) refuses data under 100 rows; elsewhere logs the model name |
| `logging` | logs every metric of the run |
| `performance-gate` | refuses a model whose `accuracy` (edit `Metric` · `Floor`) is under 0.70 |
| `deploy` | marks a model at or above a threshold as ready to deploy |

Without `--template`, a pre-train or pre-predict hook gets `validation` and a post-train or post-evaluate hook gets
`logging`. An unknown template name is refused with the list. The stages are `pre-train`, `post-train`,
`pre-predict` and `post-evaluate`; training runs the first two.

```
✅ Hook created successfully!

Name: row-guard
Type: pre-train
Template: validation
Path: .mloop/scripts/hooks/pre-train/row-guard.cs
```

During `mloop train`:

```
🔗 Executing 1 PreTrain hook(s)...
  🪝 [1/1] row-guard.cs
Dataset size: 3,000 rows
Validation passed
```

### Custom Metrics

Business metrics of your own, computed after training over the held-out rows the run's metrics were measured on,
and recorded with them — in `metrics.json`, the results table, `report.md`, `mloop list --json` and
`mloop compare`. A post-train hook sees them in `Metrics`. `mloop evaluate` (and `POST /evaluate`) computes them
again over the file it evaluates, beside the standard metrics.

**Script Location**: `.mloop/scripts/metrics/` — every `IMLoopMetric` in every file, files in name order.
Ready-made examples (profit, churn value, ROI) are in `docs/examples/metrics/`.

```csharp
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "expected_profit";
    public string Description => "100 per true positive, -50 per false positive";

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        var predicted = (bool[])ctx.Predictions;   // binary-classification
        var actual = (bool[])ctx.Labels;
        double profit = 0;
        for (var i = 0; i < predicted.Length; i++)
            if (predicted[i]) profit += actual[i] ? 100 : -50;
        return Task.FromResult(new MetricResult { Name = Name, Value = profit });
    }
}
```

| `ctx.TaskType` | `Predictions` / `Labels` |
|---|---|
| `binary-classification` | `bool[]` |
| `multiclass-classification` | `int[]` — class indices, in the order of the model's score slots |
| `regression` | `float[]` |

Rows without a label are left out, so index `i` of both arrays is the same row. Other tasks do not compute metric
scripts (training says so). A result whose name the run already reports (`accuracy`, `auc`, …) or whose value is
not finite is not recorded, with a warning; a script that fails to compile or throws is reported and skipped —
training goes on.

### Validation

Training warns about a script that does not compile and goes on without it. `mloop validate` compiles every hook
(each stage folder), metric script and pattern detector, and reports what would not run:

```
Errors (2):
  ✗ .mloop/scripts/hooks/pre-train/nothing.cs: Compiles, but has no public class implementing IMLoopHook — nothing in it would run
  ✗ .mloop/scripts/metrics/broken.cs: Compilation failed:
  line 1, column 31: Invalid expression term ';'

Warnings (1):
  ⚠ .mloop/scripts/hooks/stray.cs: In .mloop/scripts/hooks/ itself, so it runs at no stage — move it into the folder of its stage (pre-train/, post-train/, …)
```

A script error fails validation (exit 1, `--json` lists it under `errors`). What compiles is cached, so the next
training run starts from it.

---

## Project Structure

### Convention over Configuration

MLoop uses intelligent defaults:

```
my-ml-project/
├── .mloop/                    # Project marker (like .git)
│   ├── config.json            # Optional project config
│   └── scripts/
│       ├── hooks/             # Pre/post-train hooks
│       ├── metrics/           # Custom metrics
│       └── preprocess/        # Preprocessing scripts
│
├── datasets/                  # Training and prediction data
│   ├── train.csv             # Required: training data
│   ├── validation.csv        # Optional: validation split
│   ├── test.csv              # Optional: test evaluation
│   └── predict.csv           # Optional: prediction input
│
├── models/
│   ├── staging/              # All trained experiments
│   │   ├── exp-001/
│   │   │   ├── model.zip     # Trained model
│   │   │   ├── metadata.json # Metrics and config
│   │   │   └── report.md     # The experiment as a readable page
│   │   └── exp-002/
│   └── production/           # Promoted production model
│       └── current -> ../staging/exp-003/  # Symlink
│
└── predictions/              # Prediction outputs
    └── default-predictions-20241110-143022.csv
```

### Auto-Discovery

**Training**:
- Reads: `datasets/train.csv`
- Generates: Unique experiment ID (`exp-001`, `exp-002`, ...)
- Saves to: `models/staging/exp-NNN/`
- Auto-promotes: First successful model → production

**Prediction**:
- Auto-discovers: Production model from registry
- Auto-discovers: `datasets/predict.csv` if no data file specified
- Auto-generates: Timestamped output in `predictions/`

**Manual Override**:
All paths can be explicitly specified when needed:
```bash
mloop predict data/custom.csv --model models/staging/exp-005/model.zip --output results.csv
```

---

## Best Practices

### Data Preparation

1. **Clean Data First**: MLoop expects preprocessed, clean CSV data
2. **Consistent Schema**: Ensure train/test/predict files have matching columns
3. **Label Column**: Must be numeric for regression, categorical for classification
4. **Missing Values**: An empty number is read as missing. Tables with gaps train as they are (see «Missing numbers» under `mloop prep`); a time series needs its gaps filled first

### Training Workflow

1. **Start Small**: Begin with short training times (60s) to validate
2. **Iterate**: Gradually increase time budget for better models
3. **Experiment**: Try different time budgets and metrics
4. **Track**: Use `mloop list` to compare experiments
5. **Promote**: Manually promote best model with `mloop promote`

### Model Management

```bash
# Typical workflow
mloop train datasets/train.csv --label price --time 60   # Experiment
mloop train datasets/train.csv --label price --time 120  # Improve
mloop train datasets/train.csv --label price --time 180  # Optimize

# Review results
mloop list

# Promote best
mloop promote exp-003

# Validate
mloop evaluate exp-003 datasets/test.csv

# Deploy
mloop predict
```

### Performance Tips

1. **Test Split**: Use 0.2 for small datasets, 0.1 for large datasets
2. **Time Budget**: 60s for prototyping, 300-600s for production
3. **Preprocessing**: Use FilePrepper for 20x speedup on large CSVs
4. **Parallelization**: Train multiple experiments in parallel (different terminals)

### Git Integration

MLoop is designed for version control:

```bash
# .gitignore recommendations
models/staging/*/model.zip    # Large model files (optional)
predictions/                   # Generated outputs
.mloop/temp/                  # Temporary files

# DO commit
.mloop/scripts/               # Custom hooks and metrics
.mloop/config.json            # Project configuration
models/staging/*/metadata.json # Experiment metadata (small)
```

### Troubleshooting

**Training fails with "Label column not found"**:
```bash
# Check column names
mloop info datasets/train.csv

# Use exact column name (case-sensitive)
mloop train datasets/train.csv --label Price  # If column is 'Price'
```

**Low model accuracy**:
1. Increase training time: `--time 300`
2. Verify data quality with `mloop info`
3. Try different metrics: `--metric f1_score` for classification
4. Check test split: `--test-split 0.1` for more training data

**Prediction output is wrong**:
1. Verify model: `mloop list` to check production model
2. Check schema: Input file must match training schema
3. Evaluate first: `mloop evaluate` on test data

---

## Exit Codes

Every command reports its outcome through the process exit code, so a script never has to parse
output to learn whether something worked.

| Code | Meaning |
|------|---------|
| `0` | The command did what was asked. |
| `1` | The command failed. A cause is on **stderr**. |
| other | Only from `mloop serve` and `mloop update`, which pass through the exit code of the process they launched (the API server, `dotnet tool update`). |

Two rules hold across every command:

- **A non-zero exit always leaves a cause on stderr.** Reading stderr when the exit code is non-zero
  is enough; you never need stdout to find out what went wrong.
- **stdout carries the result, not the diagnostics.** Under `--json`, stdout is the document and
  nothing else — progress, warnings and errors go to stderr, so `mloop … --json > out.json` yields a
  parseable file even on the runs that print warnings.

An outcome that is not a failure exits `0` even when nothing happened. `mloop evaluate` exits `0`
and reports `skipped` when the model has no production version yet — a pipeline that evaluates
opportunistically should not stop for that. `mloop update --check` exits `0` whether or not an
update exists; the finding is in the output, not the code.

```bash
# Fail the build on a bad config, and show why.
if ! mloop validate --json > validation.json 2> validation.err; then
    cat validation.err >&2
    exit 1
fi

# Distinguish "the server stopped cleanly" from "the server crashed".
mloop serve --port 5000
case $? in
    0) echo "server stopped" ;;
    *) echo "server failed with $?" >&2 ;;
esac
```

---

## Concurrent Job Management

MLoop follows the Unix philosophy: each command runs as an independent process that exits when complete. This design provides natural isolation but requires external tools for managing multiple concurrent training jobs.

### Problem: Resource Exhaustion

When running multiple training jobs simultaneously, system resources (CPU/memory) can be exhausted:

```bash
# ❌ All 3 jobs compete for CPU → system slowdown
$ mloop train dataset1.csv --label price --time 1800 &
$ mloop train dataset2.csv --label price --time 1800 &
$ mloop train dataset3.csv --label price --time 1800 &
```

### Solution 1: Sequential Wrapper Script (Recommended)

Use the provided `mloop-queue.sh` script for simple sequential execution:

```bash
# ✅ Jobs run one after another
$ examples/scripts/mloop-queue.sh \
    "train dataset1.csv price --time 1800" \
    "train dataset2.csv price --time 1800" \
    "train dataset3.csv price --time 1800"

# Output:
# [1/3] Running: mloop train dataset1.csv --label price --time 1800
# ✅ Job 1 complete (30 min)
# [2/3] Running: mloop train dataset2.csv --label price --time 1800
# ✅ Job 2 complete (30 min)
# [3/3] Running: mloop train dataset3.csv --label price --time 1800
# ✅ Job 3 complete (30 min)
# 🎉 All jobs complete: 3/3 succeeded
```

**Features**:
- Sequential execution (one at a time)
- Progress tracking
- Error handling
- Total time reporting

### Solution 2: GNU Parallel (Advanced)

For advanced control, use [GNU Parallel](https://www.gnu.org/software/parallel/):

```bash
# Install GNU Parallel
$ sudo apt-get install parallel  # Ubuntu/Debian
$ brew install parallel           # macOS

# Run jobs with control
$ cat jobs.txt | parallel -j 1 --progress mloop {}

# jobs.txt:
# train dataset1.csv price --time 1800
# train dataset2.csv price --time 1800
# train dataset3.csv price --time 1800
```

**Advanced Options**:
```bash
# Limit to 2 concurrent jobs (if you have 2 GPUs)
$ parallel -j 2 mloop train {} ::: dataset1.csv dataset2.csv dataset3.csv

# Resume on failure
$ parallel --resume --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv

# Different time budgets per job
$ parallel --colsep ',' mloop train {1} --label price --time {2} :::: jobs.csv

# jobs.csv:
# dataset1.csv,1800
# dataset2.csv,3600
# dataset3.csv,900
```

See `examples/scripts/mloop-parallel.sh` for complete examples.

### Solution 3: OS Job Control

Use standard Unix tools for background job management:

```bash
# Schedule jobs with 'at'
$ echo "mloop train dataset1.csv --label price --time 1800" | at now
$ echo "mloop train dataset2.csv --label price --time 1800" | at now + 30 minutes
$ echo "mloop train dataset3.csv --label price --time 1800" | at now + 60 minutes

# Monitor with screen/tmux
$ screen -S training1
$ mloop train dataset1.csv --label price --time 1800
# Ctrl+A, D to detach

$ screen -S training2
$ mloop train dataset2.csv --label price --time 1800
# Ctrl+A, D to detach

# List sessions
$ screen -ls

# Reattach
$ screen -r training1
```

### Solution 4: CI/CD Pipeline Orchestration

For production workflows, integrate with existing orchestration tools:

**GitHub Actions**:
```yaml
# .github/workflows/train-models.yml
jobs:
  train:
    strategy:
      matrix:
        dataset: [dataset1, dataset2, dataset3]
      max-parallel: 1  # Sequential execution
    steps:
      - name: Train model
        run: mloop train ${{ matrix.dataset }}.csv --label price --time 1800
```

**Kubernetes**:
```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mloop-training
spec:
  parallelism: 1  # One job at a time
  completions: 3   # Total jobs
  template:
    spec:
      containers:
      - name: mloop
        image: mloop:latest
        command: ["mloop", "train"]
```

**Airflow**:
```python
# dags/mloop_training.py
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('mloop_training', schedule_interval='@daily')

train1 = BashOperator(
    task_id='train_dataset1',
    bash_command='mloop train dataset1.csv --label price --time 1800',
    dag=dag
)

train2 = BashOperator(
    task_id='train_dataset2',
    bash_command='mloop train dataset2.csv --label price --time 1800',
    dag=dag
)

train1 >> train2  # Sequential execution
```

### Resource Monitoring

Monitor resource usage during training:

```bash
# CPU/Memory monitoring
$ watch -n 5 'ps aux | grep mloop'

# System resources
$ htop

# GPU usage (if applicable)
$ watch -n 1 nvidia-smi

# Disk I/O
$ iotop
```

### Best Practices

1. **Estimate Resource Needs**: Run one job first to measure CPU/memory usage
2. **Use Sequential Execution**: Default to sequential for safety (wrapper script)
3. **Enable Parallel Only When Safe**: If you have multiple GPUs or low CPU usage
4. **Monitor First Job**: Check resource usage before starting batch jobs
5. **Use Absolute Paths**: Avoid issues with working directory in background jobs
6. **Log Output**: Redirect stdout/stderr for debugging: `mloop train ... > job1.log 2>&1`

### Performance Tips

```bash
# Reduce CPU usage with nice (lower priority)
$ nice -n 19 mloop train dataset.csv --label price --time 1800

# Pin to specific CPU cores (Linux)
$ taskset -c 0-3 mloop train dataset.csv --label price --time 1800

# Set memory limits (Linux)
$ systemd-run --scope -p MemoryMax=4G mloop train dataset.csv --label price --time 1800
```

---

## Troubleshooting

Common issues and solutions when using MLoop.

### Encoding Issues (Korean/Chinese Text)

**Problem**: Garbled text like `㿀₩Ý¢Àà§°ü¬÷Àú¥ó` instead of Korean characters

**Solution**: MLoop automatically detects and converts encodings:
```bash
# Check encoding with mloop info
mloop info data.csv
# Output: [Info] Converted CP949 → UTF-8

# Train will also auto-convert
mloop train data.csv --label label --task regression
```

If still having issues:
1. Try saving your CSV as UTF-8 with BOM in Excel
2. Use a text editor that supports encoding conversion (VS Code, Notepad++)

### Multi-CSV Merge Failures

**Problem**: Schema mismatch error when merging multiple CSV files

**Solution**:
```bash
# Check schemas first
mloop info file1.csv
mloop info file2.csv

# If columns match, use --data to specify files
mloop train --data file1.csv file2.csv --label Target --task regression

# For auto-discovery, ensure files have identical columns
mloop train --auto-merge --label Target --task regression
```

### Missing Label Values

**Problem**: Training fails with "label column contains null values"

**Solution**: Rows with an empty label are dropped by default for every task whose label is a target
(classification, regression, ranking, recommendation, NER, question answering) — a row without a
target can neither teach the model nor score it. A time series keeps its rows. To keep them anyway,
pass `--drop-missing-labels false`:
```bash
# Explicit handling
mloop train data.csv --label label --task binary-classification --drop-missing-labels

# Check the output for dropped row count
# Output: "Warning: Dropped 113 rows with missing labels"
```

### Low Model Performance

**Problem**: R² < 0.5 or Accuracy < 0.7 after training

**Checklist**:
1. **Check data quality**: `mloop info data.csv` - look for missing values, outliers
2. **Increase training time**: `--time 300` or `--time 600`
3. **Check label column**: Ensure it's the correct target variable
4. **Check class imbalance**: MLoop shows distribution automatically for classification
5. **Consider preprocessing**: Add feature engineering scripts in `.mloop/scripts/preprocessing/`

### Prediction Fails with "Unknown Categorical Values"

**Problem**: Prediction fails with "Categorical values in prediction data do not match training data"

This occurs when a text column was treated as categorical during training, and new prediction data contains values not seen in training.

**Solution** (v0.6.1+):
```bash
# Use unknown-strategy to handle unseen values
mloop predict --unknown-strategy use-most-frequent

# Or use auto mode (replaces if ratio is low, errors if high)
mloop predict --unknown-strategy auto
```

**Note**: MLoop v0.6.1+ uses a composite heuristic to detect text columns more accurately (based on token count, string length, and cardinality), which prevents most cases of this error. If you still encounter it, consider retraining the model.

### Class Imbalance / AUC Calculation Error

**Problem**: Training fails with "AUC is not defined when there is no positive class"

This occurs when minority class samples are too few for cross-validation folds.

**Solution**:
```bash
# Use --balance option to oversample minority class
mloop train data.csv --label is_fraud --task binary-classification --balance auto

# For extreme imbalance (e.g., 50:1), use aggressive balancing
mloop train data.csv --label defect --task binary-classification --balance 5

# Alternative: use F1-score metric instead of AUC
mloop train data.csv --label is_fraud --task binary-classification --metric f1_score
```

**Class Imbalance Thresholds**:
- **Moderate** (5:1): Warning shown, training continues
- **High** (10:1): Strong warning, `--balance auto` recommended
- **Extreme** (20:1+): May require manual oversampling or `--balance` option
- **Critical** (50:1+): Cross-validation likely to fail without balancing

### Training Timeout

**Problem**: Training finishes but metrics are poor or "no valid model found"

**Solution**:
```bash
# Increase training time budget
mloop train data.csv --label label --time 600  # 10 minutes

# For complex datasets, try 30 minutes
mloop train data.csv --label label --time 1800
```

### API Server Won't Start

**Problem**: `mloop serve` fails or can't connect

**Checklist**:
1. **Port in use**: Try different port `mloop serve --port 8080`
2. **No production model**: Promote a model first `mloop promote exp-001`
3. **Model corrupted**: Re-train and promote a new model

### Docker Build Fails

**Problem**: Dockerfile generation or build fails

**Solution**:
```bash
# Regenerate Docker files
mloop docker

# Ensure production model exists
ls models/production/current/

# Build with verbose output
docker build -t my-model . --progress=plain
```

---

## Next Steps

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- **Examples**: Check `examples/` directory for complete workflows
- **Concurrent Execution**: See `examples/scripts/` for job management scripts
- **Contributing**: Review development guidelines in [ARCHITECTURE.md](ARCHITECTURE.md)

---

**MLoop**: Clean Data In, Trained Model Out - That's It.
