# Prediction response contract

What `mloop predict --json` and `POST /predict` return, field by field.

Both come from the same code path, so a row means the same thing in either. They differ only in the
envelope around the rows.

## Envelope

`mloop predict --json` writes one JSON object to stdout:

```json
{
  "model": "default",
  "task": "multiclass-classification",
  "count": 3,
  "predictions": [ /* rows */ ],
  "warnings": ["…"]
}
```

`POST /predict` returns the same rows with more about where they came from:

```json
{
  "modelName": "default",
  "experimentId": "exp-001",
  "predictedAt": "2026-09-09T04:15:22.1147Z",
  "task": "multiclass-classification",
  "count": 3,
  "predictions": [ /* rows */ ],
  "warnings": null
}
```

Two differences worth knowing before you write a parser:

- The model's name is `model` on the CLI and `modelName` over HTTP; only the HTTP response names the
  experiment the prediction came from.
- 🔴 **The CLI omits null fields; the server writes them.** This applies to the envelope *and to
  every field of every row*. The same multiclass row is `{"predictedLabel":"cat","probabilities":
  {…},"confidence":0.5}` from the CLI and `{"predictedLabel":"cat","probabilities":{…},"score":null,
  "clusterId":null,"distances":null,…,"confidence":0.5}` over HTTP. **Absent and `null` mean the same
  thing** — read a field as "has a value" or "does not", never as "the key is there".

`count` is `predictions.length`. It is there so a consumer can check a truncated read.

## Rows

A row carries values only for the fields its task produces. Which fields have a **non-null** value is
therefore information about the task; which keys are *present* is not, because that depends on which
of the two surfaces you asked (see above). A field without a value is absent or null — never zero.

| Field | Type | Carries a value for |
|---|---|---|
| `predictedLabel` | string | classification (binary, multiclass, text, image) |
| `probabilities` | object: class name → number | classification |
| `score` | number | binary classification (P of the positive class), regression, forecasting, ranking, recommendation |
| `scoreLowerBound`, `scoreUpperBound` | number | regression, when the model carries a conformal band |
| `intervalConfidence` | number | with the band above |
| `clusterId` | integer | clustering |
| `distances` | number[] | clustering |
| `isAnomaly` | boolean | anomaly detection, time-series anomaly |
| `anomalyScore` | number | anomaly detection, time-series anomaly |
| `confidence` | number in [0,1] | wherever the task has a usable uncertainty signal |

### `probabilities` — keyed by class name

The keys are **the classes the model was trained on**, not positions:

```json
{"predictedLabel": "bird", "probabilities": {"dog": 0.248, "bird": 0.504, "cat": 0.248}, "confidence": 0.504}
```

Two guarantees follow from that, and they are the reason to read this field rather than reconstruct
it:

- **A row's `predictedLabel` is always one of its own `probabilities` keys** — and it is the key with
  the largest value. You can join the two without knowing anything else about the model.
- **A label written as class ids behaves identically.** Training on `0`/`1`/`2` gives
  `{"predictedLabel": "0", "probabilities": {"2": …, "0": …, "1": …}}` — the ids are the class names.

Key **order** is the trainer's, not sorted, and carries no meaning. Do not index into the object.

Binary classification returns both classes even though the model computes one number, so
`max(probabilities)` is a correct confidence for a negative prediction too — a confident `NG`
(P(`OK`) ≈ 0.02) reads as 0.98, not 0.02.

**If the names are unavailable, the keys stay positional.** A model whose scored output does not
record its class names — or records names that do not cover its classes exactly and distinctly — is
keyed `class_0`, `class_1`, … instead, which is what every multiclass model returned before naming
was introduced. A consumer that must handle both can test one key: if it matches `^class_\d+$` the
names were unavailable and the position is all there is.

This is a compatibility path, not something you should expect to meet: every tabular multiclass model
this version trains maps its label through a key conversion, and that conversion is what leaves the
names behind — measured on both a word-valued label and an id-valued one. The deep-learning
classification paths (image, text) key their labels the same way, but their scored output has not
been measured here.

A model whose scores contain `NaN` or an infinity omits `probabilities` for that row entirely rather
than reporting a distribution that is not one.

### `confidence` — one number, comparable across tasks

`confidence` is MLoop's normalized per-row uncertainty in `[0,1]`, so a consumer does not re-derive
one per task family:

- classification → the winning class probability
- anomaly detection → distance from the 0.5 decision boundary
- regression → how narrow this row's conformal band is against the model's typical error

It is `null` when the model exposes no usable signal — including for **time-series** anomaly, whose
raw detector score is not a probability with a 0.5 boundary, so mapping it would fabricate a number.

Do not confuse it with `intervalConfidence`, which is the band's **coverage level** (0.90 = "90% of
rows fall inside"), a property of the model rather than of the row.

### Regression bands

`score` is the prediction; `scoreLowerBound`/`scoreUpperBound` bracket it at the `intervalConfidence`
coverage level. The width is not always constant — a model trained with a residual model widens the
band on rows it expects to get wrong, which is what makes band width usable as a per-row escalate
signal.

Both bounds are absent for a model trained before conformal bands, or for any non-regression task.

## Errors

`mloop predict --json` exits non-zero and writes the reason to stderr; stdout stays empty. Never
parse stdout without checking the exit code first.

`POST /predict` returns `400` for a caller error (for example, input rows sharing no column with the
trained schema) and `500` otherwise, both as a problem document.

One failure is deliberately *not* silent in either: if every row's defining output comes back null or
non-finite — a model scored for a different task than declared, or a degenerate model — the request
fails rather than returning rows of nulls that look like "nothing to report".
