#!/usr/bin/env python3
"""Generated pictures standing in for "the user's own pictures" in the install walks (the picture counterpart of walk-tables.py).

    walk-pictures.py <picture directory> <truth directory>

    shapes/train/<circle|square|triangle>/   40 pictures each - the shape is the only thing that tells the classes apart
    shapes/new/                              15 pictures to classify, named so that a name says nothing about its class
    shapes/test/<class>/                     10 labelled pictures each, to evaluate
    boxes/train/                             60 pictures and annotations.json (COCO), 2 classes: red and blue rectangles
    boxes/photos/                            12 test pictures alone, to predict
    boxes/test/                              the same 12 with annotations.json, to evaluate
    surface/normal/                          60 pictures of a plain brushed surface
    surface/probes/                          10 pictures to score: 5 normal, 5 with a planted defect, names that say neither
    surface/check/{good,scratch,spot}/       a labelled folder to evaluate (10 normal, 5 + 5 defective)
    bad/                                     picture sets that are wrong on purpose

What the generator knows (each new picture's class, each test picture's boxes, which probes carry a defect and where) goes to
<truth directory>/pictures.json, which the reader never sees. The seeds are fixed, so every run draws the same pictures.

Only the standard library is used: pictures are written as PNG with zlib.
"""
import json
import math
import os
import random
import struct
import sys
import zlib

root, truth_root = sys.argv[1], sys.argv[2]
truth = {}


def write_png(path, width, height, pixels):
    """`pixels` is a bytearray of width * height * 3 RGB values, row by row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(pixels[y * stride:(y + 1) * stride]) for y in range(height))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def clamp(v):
    return 0 if v < 0 else 255 if v > 255 else int(v)


def background(rng, width, height, base, noise):
    pixels = bytearray(width * height * 3)
    for i in range(0, len(pixels), 3):
        n = rng.randint(-noise, noise)
        pixels[i], pixels[i + 1], pixels[i + 2] = clamp(base[0] + n), clamp(base[1] + n), clamp(base[2] + n)
    return pixels


def paint(pixels, width, x, y, colour):
    i = (y * width + x) * 3
    pixels[i], pixels[i + 1], pixels[i + 2] = colour


# ---------------------------------------------------------------- classification: three shapes
SIZE = 128
SHAPES = ("circle", "square", "triangle")
COLOURS = ((186, 52, 48), (52, 92, 178), (46, 128, 74), (150, 96, 34), (110, 64, 142), (40, 40, 44))


def inside_polygon(x, y, points):
    inside = False
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def shape_picture(rng, kind):
    base = rng.randint(212, 244)
    pixels = background(rng, SIZE, SIZE, (base, base, base), 8)
    colour = tuple(clamp(c + rng.randint(-18, 18)) for c in rng.choice(COLOURS))
    radius = rng.randint(24, 46)
    cx, cy = rng.randint(radius + 2, SIZE - radius - 3), rng.randint(radius + 2, SIZE - radius - 3)
    if kind == "circle":
        test = lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    else:
        corners, turn = (4, rng.uniform(-0.35, 0.35) + math.pi / 4) if kind == "square" else (3, rng.uniform(-0.45, 0.45) - math.pi / 2)
        points = [(cx + radius * math.cos(turn + 2 * math.pi * k / corners), cy + radius * math.sin(turn + 2 * math.pi * k / corners))
                  for k in range(corners)]
        test = lambda x, y: inside_polygon(x + 0.5, y + 0.5, points)
    for y in range(max(0, cy - radius), min(SIZE, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(SIZE, cx + radius + 1)):
            if test(x, y):
                paint(pixels, SIZE, x, y, colour)
    return pixels


rng = random.Random(41)
for kind in SHAPES:
    for i in range(40):
        write_png(f"{root}/shapes/train/{kind}/{kind}-{i:02d}.png", SIZE, SIZE, shape_picture(rng, kind))
rng = random.Random(42)
new_kinds = [SHAPES[i % 3] for i in range(15)]
rng.shuffle(new_kinds)
truth["shapes_new"] = {}
for i, kind in enumerate(new_kinds, 1):
    name = f"p{i:02d}.png"
    write_png(f"{root}/shapes/new/{name}", SIZE, SIZE, shape_picture(rng, kind))
    truth["shapes_new"][name] = kind
rng = random.Random(43)
for kind in SHAPES:
    for i in range(10):
        write_png(f"{root}/shapes/test/{kind}/t-{kind}-{i:02d}.png", SIZE, SIZE, shape_picture(rng, kind))

# ---------------------------------------------------------------- detection: red and blue rectangles
BOX_W, BOX_H = 220, 160
BOX_CLASSES = (("red", (196, 60, 60)), ("blue", (53, 80, 184)))


def box_picture(rng, must_have=None):
    pixels = background(rng, BOX_W, BOX_H, (233, 237, 243), 7)
    count = rng.randint(1, 3)
    placed = []
    while len(placed) < count:
        w, h = rng.randint(36, 64), rng.randint(36, 64)
        x, y = rng.randint(2, BOX_W - w - 3), rng.randint(2, BOX_H - h - 3)
        if any(x < px + pw + 6 and px < x + w + 6 and y < py + ph + 6 and py < y + h + 6 for _, px, py, pw, ph in placed):
            continue
        kind = must_have if must_have is not None and not placed else rng.randrange(2)
        placed.append((kind, x, y, w, h))
    for kind, x, y, w, h in placed:
        colour = tuple(clamp(c + rng.randint(-14, 14)) for c in BOX_CLASSES[kind][1])
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                paint(pixels, BOX_W, xx, yy, colour)
    return pixels, placed


def coco(images, annotations):
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": name} for i, (name, _) in enumerate(BOX_CLASSES)]}


def box_set(rng, count, prefix, folders):
    images, annotations, objects = [], [], {}
    for i in range(count):
        pixels, placed = box_picture(rng, must_have=i % 2)
        name = f"{prefix}{i + 1:02d}.png"
        for folder in folders:
            write_png(f"{folder}/{name}", BOX_W, BOX_H, pixels)
        images.append({"id": i + 1, "file_name": name, "width": BOX_W, "height": BOX_H})
        objects[name] = []
        for kind, x, y, w, h in placed:
            annotations.append({"id": len(annotations) + 1, "image_id": i + 1, "category_id": kind + 1,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            objects[name].append({"class": BOX_CLASSES[kind][0], "box": [x, y, x + w, y + h]})
    return coco(images, annotations), objects


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1)


train_coco, _ = box_set(random.Random(51), 60, "img", [f"{root}/boxes/train"])
write_json(f"{root}/boxes/train/annotations.json", train_coco)
test_coco, truth["boxes"] = box_set(random.Random(52), 12, "photo", [f"{root}/boxes/photos", f"{root}/boxes/test"])
write_json(f"{root}/boxes/test/annotations.json", test_coco)

# ---------------------------------------------------------------- anomaly: a brushed surface, normal and defective
SIDE = 224


def surface(rng):
    base = rng.randint(178, 192)
    streaks = [rng.randint(-5, 5) for _ in range(SIDE)]
    pixels = bytearray(SIDE * SIDE * 3)
    i = 0
    for y in range(SIDE):
        for _ in range(SIDE):
            v = clamp(base + streaks[y] + rng.randint(-4, 4))
            pixels[i], pixels[i + 1], pixels[i + 2] = v, clamp(v + 3), clamp(v + 8)
            i += 3
    return pixels


def defect(rng, pixels, kind):
    """Plants one defect in the middle part of the picture (an encoder reads a centre crop) and returns its box."""
    cx, cy = rng.randint(66, 158), rng.randint(66, 158)
    if kind == "scratch":
        angle, length, half = rng.uniform(0, math.pi), rng.randint(34, 56), 1.8
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        ax, ay, bx, by = cx - dx, cy - dy, cx + dx, cy + dy
        colour = (44, 44, 48)

        def hit(x, y):
            t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
            return math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)) <= half
    else:
        radius = rng.randint(8, 12)
        colour = (92, 64, 44)

        def hit(x, y):
            return math.hypot(x - cx, y - cy) <= radius
    xs, ys = [], []
    for y in range(SIDE):
        for x in range(SIDE):
            if hit(x, y):
                paint(pixels, SIDE, x, y, colour)
                xs.append(x)
                ys.append(y)
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


rng = random.Random(61)
for i in range(60):
    write_png(f"{root}/surface/normal/ok-{i:02d}.png", SIDE, SIDE, surface(rng))
rng = random.Random(62)
plan = [None] * 5 + ["scratch", "scratch", "scratch", "spot", "spot"]
rng.shuffle(plan)
truth["probes"] = {}
for i, kind in enumerate(plan, 1):
    name = f"probe-{i:02d}.png"
    pixels = surface(rng)
    box = defect(rng, pixels, kind) if kind else None
    write_png(f"{root}/surface/probes/{name}", SIDE, SIDE, pixels)
    truth["probes"][name] = {"defect": kind, "box": box}
rng = random.Random(63)
for folder, kind, count in (("good", None, 10), ("scratch", "scratch", 5), ("spot", "spot", 5)):
    for i in range(count):
        pixels = surface(rng)
        if kind:
            defect(rng, pixels, kind)
        write_png(f"{root}/surface/check/{folder}/{folder}-{i:02d}.png", SIDE, SIDE, pixels)

# ---------------------------------------------------------------- picture sets that are wrong on purpose
rng = random.Random(71)
for i in range(8):
    write_png(f"{root}/bad/one-class/circle/c{i}.png", SIZE, SIZE, shape_picture(rng, "circle"))
for kind in ("circle", "square"):
    for i in range(8):
        write_png(f"{root}/bad/empty-class/{kind}/{kind[0]}{i}.png", SIZE, SIZE, shape_picture(rng, kind))
os.makedirs(f"{root}/bad/empty-class/triangle", exist_ok=True)

broken, _ = box_set(random.Random(72), 6, "img", [f"{root}/bad/coco-broken"])
broken["images"].append({"id": 7, "file_name": "img-missing.png", "width": BOX_W, "height": BOX_H})
broken["annotations"].append({"id": 900, "image_id": 7, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0})
broken["annotations"].append({"id": 901, "image_id": 2, "category_id": 2, "bbox": [200, 140, 90, 80], "area": 7200, "iscrowd": 0})
write_json(f"{root}/bad/coco-broken/annotations.json", broken)

write_png(f"{root}/bad/corrupt/good.png", SIZE, SIZE, shape_picture(rng, "square"))
with open(f"{root}/bad/corrupt/broken.png", "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + bytes(random.Random(73).randrange(256) for _ in range(600)))
os.makedirs(f"{root}/bad/no-pictures", exist_ok=True)
with open(f"{root}/bad/no-pictures/notes.txt", "w") as f:
    f.write("no pictures here\n")

os.makedirs(truth_root, exist_ok=True)
write_json(f"{truth_root}/pictures.json", truth)
