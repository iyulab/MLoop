# Prediction response contract

What `mloop predict --json` and `POST /predict` return, field by field.

Both come from the same code path, so a row means the same thing in either. They differ only in the
envelope around the rows.

## Envelope

`mloop predict --json` writes one JSON object to stdout:

```json
{
  "model": "default",
  "experimentId": "exp-001",
  "task": "multiclass-classification",
  "count": 3,
  "predictions": [ /* rows */ ],
  "warnings": ["…"]
}
```

`experimentId` names the model that answered — production's experiment by default, the one you asked
for with `--experiment <exp-id>` (a shadow prediction: a staging candidate answers, production stays
where it is), and absent for a `--model <path>` file outside the project's experiments. Comparing a
candidate with production is two calls on the same file, keyed by it.

`POST /predict` returns the same rows with more about where they came from:

```json
{
  "modelName": "default",
  "experimentId": "exp-001",
  "predictedAt": "2026-09-09T04:15:22.1147Z",
  "task": "multiclass-classification",
  "count": 3,
  "predictions": [ /* rows */ ],
  "warnings": ["…"]
}
```

One difference worth knowing before you write a parser: the model's name is `model` on the CLI and
`modelName` over HTTP, and only the HTTP response carries `predictedAt`. A shadow prediction over
HTTP is `POST /predict?name=<model>&experiment=<exp-id>` — `404` when the experiment does not exist
or has no trained model.

🔴 **A field without a value is absent — on both surfaces, in the envelope and in every row.** The
same multiclass row is `{"predictedLabel":"cat","probabilities":{…},"confidence":0.5}` whichever way
you asked; `warnings` is there only when there is something to warn about. Read a field as "has a
value" or "does not", never as "the key is there". (Before 0.32.0 the server wrote such fields as
`null`; a consumer that also has to read those responses should treat `null` as absent.)

`count` is `predictions.length`. It is there so a consumer can check a truncated read.

## Rows

A row carries values only for the fields its task produces — so which fields a row *has* is
information about the task. A field without a value is absent, never zero and never `null`.

| Field | Type | Carries a value for |
|---|---|---|
| `predictionId` | string (16 hex characters) | `POST /predict` rows that were logged — the id `POST /feedback` takes to record the actual value. Absent from `mloop predict --json` (find a CLI prediction's id with `mloop logs`) and from a served row the server could not log, which then carries a warning |
| `predictedLabel` | string | classification (binary, multiclass, text, image); NER (one tag per word); question answering (the best answer) |
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

### Object detection — boxes per image

An object-detection model answers each image with the objects it found, so its rows are images, not table rows.
`POST /predict` takes the images by their contents — `{"images": ["<base64>", …]}` or a multipart form, up to 64 per
request, as `POST /embed` takes them; rows sent to it are a `400`. Each prediction is one image, in the order sent:

```json
{
  "imageSha256": "4f2b0a41…",
  "detections": [
    { "label": "red", "score": 0.887, "x0": 72.5, "y0": 108.0, "x1": 105.8, "y1": 142.2 }
  ],
  "predictionId": "53b147ab28e14713"
}
```

A box is in the image's pixels: `x0 y0` its top-left corner, `x1 y1` its bottom-right. `score` is the detector's
confidence in that object; an image with nothing found has an empty `detections`. The image is named by its SHA-256
— the server never sees a path — and is logged under it, with the detections as the answer and the highest score as
the confidence. `mloop predict <images>` writes the same detections to a file (`ImagePath` in place of the hash).

## Errors

`mloop predict --json` exits non-zero and writes one document to stdout — `{"error": "…"}` — with
the same reason rendered for a person on stderr. A schema mismatch reads
`{"error": "Schema validation failed. Missing required columns: …"}`. Check the exit code first, then
parse: a zero exit carries the envelope above, a non-zero exit carries the error document, and stdout
is never empty in either case.

`POST /predict` returns `400` for a caller error (for example, input rows sharing no column with the
trained schema) and `500` otherwise, both as a problem document.

One failure is deliberately *not* silent in either: if every row's defining output comes back null or
non-finite — a model scored for a different task than declared, or a degenerate model — the request
fails rather than returning rows of nulls that look like "nothing to report".
