"""Writes the image-encoder fixtures the embedding tests read.

    python make_fixtures.py   (needs numpy, pillow, onnx)

- pattern.png: a synthetic 37x23 RGB image (seeded noise over a gradient).
- tiny-encoder.onnx: pixel_values [N,3,6,6] -> mean over H,W -> [N,3] x W[3,4] -> [N,1,4] "last_hidden_state",
  shaped like a vision transformer's output so the CLS pooling path is the one exercised.
- expected.json: the preprocessed tensor (shortest edge 8, centre crop 6, ImageNet mean/std) as an image processor
  with Pillow's bicubic resize computes it, and the L2-normalised embedding of that tensor.
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SHORTEST, CROP = 8, 6
W = np.array([[0.5, -1.0, 0.25, 2.0], [1.5, 0.0, -0.75, 1.0], [-0.5, 1.0, 1.0, 0.5]], dtype=np.float32)

rng = np.random.default_rng(7)
y, x = np.mgrid[0:23, 0:37]
base = np.stack([x * 6, y * 10, (x + y) * 4], axis=-1)
pixels = np.clip(base + rng.integers(-40, 40, size=base.shape), 0, 255).astype(np.uint8)
Image.fromarray(pixels).save(os.path.join(HERE, "pattern.png"))

image = Image.open(os.path.join(HERE, "pattern.png")).convert("RGB")
w, h = image.size
short, long = (w, h) if w <= h else (h, w)
new_short, new_long = SHORTEST, int(SHORTEST * long / short)
nw, nh = (new_short, new_long) if w <= h else (new_long, new_short)
resized = np.asarray(image.resize((nw, nh), resample=Image.BICUBIC))
top, left = (nh - CROP) // 2, (nw - CROP) // 2
cropped = resized[top:top + CROP, left:left + CROP].astype(np.float32) * np.float32(1 / 255)
tensor = ((cropped - MEAN) / STD).transpose(2, 0, 1)  # CHW

embedding = tensor.mean(axis=(1, 2)) @ W
embedding = embedding / np.linalg.norm(embedding)

graph = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "W"], ["projected"]),
        helper.make_node("Unsqueeze", ["projected", "axes"], ["last_hidden_state"]),
    ],
    "tiny-encoder",
    [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, CROP, CROP])],
    [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["N", 1, 4])],
    [helper.make_tensor("W", TensorProto.FLOAT, [3, 4], W.flatten().tolist()),
     helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="mloop-fixture")
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(HERE, "tiny-encoder.onnx"))

with open(os.path.join(HERE, "expected.json"), "w") as f:
    json.dump({"resized": [nw, nh], "tensor": [float(v) for v in tensor.flatten()],
               "embedding": [float(v) for v in embedding]}, f, indent=1)
