# Scoring an exported model without MLoop

`mloop export onnx` writes the model as an ONNX graph (`model.onnx`) and a manifest beside it
(`model.onnx.json`). [`score.py`](score.py) answers from those two files alone — ONNX Runtime and
numpy, nothing of MLoop — and gives the answers `mloop predict` gives.

```bash
mloop export onnx --data datasets/test.csv --verify -o out
pip install onnxruntime numpy
python score.py out/model.onnx.json datasets/test.csv > answers.csv
```

What it reads from the manifest:

- **`inputs`** — for each graph input, the data columns that fill it, in slot order. A number column
  feeds `float32` (an empty cell is a missing number, `NaN`, as MLoop reads it); a category or text
  column feeds `string`.
- **`unusedInputs`** — inputs the graph declares but no output reads (a multiclass model's label).
  They are fed a blank.
- **`calibration`** and **`decisionThreshold`** — the layers `mloop predict` applies to the graph's
  outputs, in that order: the probability calibration (Platt, isotonic, or a multiclass temperature),
  then the threshold on the calibrated probability. `label.decision` states the same rule in words.
- **`label.classes`** — the class names, in the order of the graph's `Score` slots.

Checked on a binary model with a Platt calibration and a threshold (200 rows) and a multiclass model
with a temperature and missing numbers (150 rows): every label equal to `mloop predict`'s, every
probability within 1e-6.
