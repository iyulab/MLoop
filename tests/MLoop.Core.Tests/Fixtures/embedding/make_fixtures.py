"""Writes the image-encoder fixtures the embedding tests read.

    python make_fixtures.py   (needs numpy, pillow, onnx)

- pattern.png: a synthetic 37x23 RGB image (seeded noise over a gradient).
- tiny-encoder.onnx: pixel_values [N,3,6,6] -> mean over H,W -> [N,3] x W[3,4] -> [N,1,4] "last_hidden_state",
  shaped like a vision transformer's output so the CLS pooling path is the one exercised.
- expected.json: the preprocessed tensor (shortest edge 8, centre crop 6, ImageNet mean/std) as an image processor
  with Pillow's bicubic resize computes it, and the L2-normalised embedding of that tensor.
- tiny-patch-encoder.onnx: the same class token, then one register token, then a 2 x 2 grid of 3-pixel patches (each
  patch's mean x W), row by row -> [N,6,4] "last_hidden_state" — a vision transformer with registers.
- patches.json: the L2-normalised patch vectors of pattern.png, row by row, and the centre crop in its own pixels.
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

# translucent.png / translucent.json: the same pattern with an alpha ramp (fully transparent to opaque), and the tensor
# a processor computes after converting to RGB — which drops the alpha and keeps the colour stored under it.
alpha = np.clip(x * 7, 0, 255).astype(np.uint8)[..., None]
Image.fromarray(np.concatenate([pixels, alpha], axis=-1)).save(os.path.join(HERE, "translucent.png"))
image = Image.open(os.path.join(HERE, "translucent.png")).convert("RGB")
resized = np.asarray(image.resize((nw, nh), resample=Image.BICUBIC))
cropped = resized[top:top + CROP, left:left + CROP].astype(np.float32) * np.float32(1 / 255)
translucent = ((cropped - MEAN) / STD).transpose(2, 0, 1)
with open(os.path.join(HERE, "translucent.json"), "w") as f:
    json.dump({"tensor": [float(v) for v in translucent.flatten()]}, f, indent=1)

# tiny-patch-encoder.onnx / patches.json: a class token, a register token, then the patch tokens row by row.
PATCH = 3
patch_graph = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "W"], ["projected"]),
        helper.make_node("Unsqueeze", ["projected", "axes"], ["cls"]),
        helper.make_node("Neg", ["cls"], ["register"]),
        helper.make_node("AveragePool", ["pixel_values"], ["cells"], kernel_shape=[PATCH, PATCH], strides=[PATCH, PATCH]),
        helper.make_node("Reshape", ["cells", "cells_shape"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["tokens"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["tokens", "W"], ["patches"]),
        helper.make_node("Concat", ["cls", "register", "patches"], ["last_hidden_state"], axis=1),
    ],
    "tiny-patch-encoder",
    [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, CROP, CROP])],
    [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["N", 6, 4])],
    [helper.make_tensor("W", TensorProto.FLOAT, [3, 4], W.flatten().tolist()),
     helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
     helper.make_tensor("cells_shape", TensorProto.INT64, [3], [0, 3, -1])],
)
patch_model = helper.make_model(patch_graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="mloop-fixture")
patch_model.ir_version = 8
onnx.checker.check_model(patch_model)
onnx.save(patch_model, os.path.join(HERE, "tiny-patch-encoder.onnx"))

g = CROP // PATCH
patches = []
for r in range(g):
    for c in range(g):
        v = tensor[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH].mean(axis=(1, 2)) @ W
        patches.append([float(x) for x in v / np.linalg.norm(v)])
crop_region = [left * w / nw, top * h / nh, (left + CROP) * w / nw, (top + CROP) * h / nh]
with open(os.path.join(HERE, "patches.json"), "w") as f:
    json.dump({"patches": patches, "crop": crop_region}, f, indent=1)
