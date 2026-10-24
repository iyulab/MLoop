"""Answer as `mloop predict` does from an exported model, with nothing of MLoop installed.

    pip install onnxruntime numpy
    python score.py out/model.onnx.json datasets/test.csv > answers.csv

Reads only the files `mloop export onnx` writes: the graph (`model.onnx`) and its manifest
(`model.onnx.json`). The manifest says which data columns fill each graph input, in slot order, and
which layers sit beside the graph — a probability calibration, then a decision threshold — that
`mloop predict` applies to the graph's outputs. This script applies them in that order.
"""
import csv
import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort


def inputs_for(manifest, rows):
    """One array per graph input: its source columns, in slot order, for every row.

    `unusedInputs` are inputs the graph declares but no output reads (a multiclass model's label);
    they are fed a blank of their type.
    """
    feeds = {}
    for spec in manifest["inputs"] + manifest.get("unusedInputs", []):
        unused = spec in manifest.get("unusedInputs", [])
        values = [["" if unused else row[c] for c in spec["sourceColumns"]] for row in rows]
        if spec["elementType"] == "string":
            feeds[spec["name"]] = np.array(values, dtype=object)
        else:
            # An empty cell is a missing number (NaN), as MLoop reads it — never 0.
            feeds[spec["name"]] = np.array(
                [[float(v) if v.strip() else math.nan for v in r] for r in values], dtype=np.float32)
    return feeds


def calibrated_binary(p, calibration):
    if calibration["method"] == "platt":
        p = min(max(p, 1e-7), 1 - 1e-7)
        return 1.0 / (1.0 + math.exp(-(calibration["a"] * math.log(p / (1 - p)) + calibration["b"])))
    knots, values = calibration["knots"], calibration["values"]  # isotonic: linear between knots, flat outside
    return float(np.interp(p, knots, values))


def tempered(scores, temperature):
    raised = [max(s, 1e-12) ** (1.0 / temperature) for s in scores]
    total = sum(raised)
    return [r / total for r in raised]


def main(manifest_path, data_path):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    graph = os.path.join(os.path.dirname(manifest_path), manifest["onnxFile"])
    session = ort.InferenceSession(graph)
    with open(data_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    outputs = {o["column"]: o["name"] for o in manifest["outputs"]}
    names = list(outputs.values())
    result = dict(zip(names, session.run(names, inputs_for(manifest, rows))))
    task, classes = manifest["task"], manifest["label"]["classes"]
    calibration, threshold = manifest.get("calibration"), manifest.get("decisionThreshold")

    out = csv.writer(sys.stdout)
    if task == "binary-classification":
        out.writerow(["PredictedLabel", "Probability"])
        for p, label in zip(result[outputs["Probability"]].ravel(), result[outputs["PredictedLabel"]].ravel()):
            p, label = float(p), bool(label)
            if calibration and not calibration.get("isMulticlass"):
                p = calibrated_binary(p, calibration)
                label = p >= 0.5
            if threshold:
                label = p >= threshold["threshold"]
            out.writerow([classes[1] if label else classes[0], f"{p:.6f}"])
    elif task == "multiclass-classification":
        out.writerow(["PredictedLabel", *classes])
        for scores in result[outputs["Score"]]:
            scores = [float(s) for s in scores]
            if calibration and calibration.get("isMulticlass"):
                scores = tempered(scores, calibration["temperature"])
            out.writerow([classes[int(np.argmax(scores))], *(f"{s:.6f}" for s in scores)])
    else:
        out.writerow(["Score"])
        for score in result[outputs["Score"]].ravel():
            out.writerow([f"{float(score):.6f}"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
