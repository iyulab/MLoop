# A reference pipeline: retrain, compare, promote, watch

Two files take a model from fresh data to production and keep an eye on it afterwards. Both are examples to copy and
change, not part of the tool.

| File | Runs | Does |
|------|------|------|
| [`retrain-and-promote.yml`](retrain-and-promote.yml) | GitHub Actions, weekly or by hand | trains a candidate that stays in staging, compares it with production on the same rows, promotes it through the project's gate, keeps the evidence |
| [`watch-production.sh`](watch-production.sh) | cron, on the machine that serves the model | asks `mloop trigger check` whether the model should be retrained and starts the workflow when it should |

## What decides the promotion

The workflow does not decide; the project does. Put the rules in `mloop.yaml` and every way of promoting — this workflow,
`mloop promote`, `POST /promote`, the end of a training run — applies them:

```yaml
models:
  default:
    task: binary-classification
    label: y
    gate:
      data: datasets/recheck.csv        # labeled rows of what the service sees now
      no_worse_than_production: true    # refused when production is better on these rows
```

`datasets/recheck.csv` should be labeled rows of the traffic the model serves today — for instance rows you reviewed from
the prediction log — not the file the model was trained on. With `no_worse_than_production` the candidate is compared with
production row by row on exactly these rows, which is the comparison the workflow's third step prints. A candidate the
gate refuses stays in staging, `mloop promote` exits non-zero and says which rule held it, and the job fails; nothing in
production has moved. See the guide's section on a project's own gate policy for `limits` (of the rows labeled
`actual`, at most `max` may be answered `predicted`).

Every change of production is recorded in the promotion history with who made it and how (`mloop promote --history`);
the workflow names itself with `--by`, so the history says which run promoted a model.

## Stages

1. **Check** — `mloop validate` checks `mloop.yaml` (task types, column types, metrics, prep steps, the gate), so a broken project fails before the search starts.
2. **Train a candidate** — `--no-promote` keeps the result in staging whatever it scores.
3. **Compare** — `evaluate <candidate> <rows> --against production` scores both models on the same rows and reports the
   paired difference with its interval and a verdict (`Indistinguishable`, `CandidateBetter` or `BaselineBetter`).
4. **Promote** — through the gate (`--json` never asks for confirmation, so the step runs unattended). A first model has
   nothing to be compared with and is promoted if it passes the fixed quality gate (it learned something).
5. **Evidence** — the training events and the comparison are kept as a workflow artifact.

Deployment is left to you: `mloop serve` reloads the production model when its file changes, so shipping the promoted
model is a copy of `models/<name>/production`. `mloop docker` writes Docker files for the project.

## Watching the served model

`watch-production.sh` asks the question the guide's trigger section describes: has accuracy on reported feedback fallen
below the threshold, has enough feedback come in, is the model old, has the input drifted from what it was trained on.
It runs where the prediction log and the feedback live — the machine that serves the model — and starts the workflow
with the GitHub CLI when any condition is met. Run on a machine that has neither, it reports that nothing could be
measured (`-`), which is not a reason to retrain.

## Try it without a repository

The same commands run on a table made on the spot. In an empty folder:

```bash
mloop init demo --task binary-classification --label y
cd demo
```

Make `datasets/train.csv` and `datasets/recheck.csv` (any table with a `y` column), then:

```bash
mloop validate
mloop train datasets/train.csv --no-promote --json
mloop evaluate exp-001 datasets/recheck.csv --against production --json
mloop promote exp-001 --by "me" --json
mloop promote --history
mloop trigger check --json
```

(Before the first promotion there is no production model to compare with, so skip the `--against production` line for the
first run.)
