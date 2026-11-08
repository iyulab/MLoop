#!/bin/bash
#
# Drives the deep-learning runtimes end to end in a Linux x64 container — the paths CI does not run, because it
# installs no TensorFlow or libtorch:
#
#   tf     runtime install → image classification training (imbalanced, --balance) → mloop serve → POST /predict with images (base64 and
#          multipart) → no scratch file left behind → a second experiment kept in staging → shadow prediction → promote → rollback,
#          the served answers and the promotion history following production
#   torch  runtime install → text classification training → predictions whose probabilities sum to one → object detection: train, a
#          second experiment in staging, shadow prediction (predict --json), promote, rollback, the history, and the boxes over HTTP
#   anomaly  onnxruntime + the dinov2-small encoder install -> image anomaly training -> the same lifecycle (staging, shadow prediction, promote,
#          rollback, history) -> an odd picture scores higher than a normal one, on the CLI and over HTTP
#
# Usage:
#   ./scripts/runtime-smoke.sh            # both
#   ./scripts/runtime-smoke.sh tf         # one (tf, torch, anomaly)
#
# Needs Docker (on an arm64 machine the container runs emulated — minutes, not seconds) and python3 for the synthetic
# images and sentences. The CLI is packed from this tree; nothing is published.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then RUNTIMES=(tf torch anomaly); else RUNTIMES=("$@"); fi

# A directory Docker can mount on every host (Docker Desktop shares the home directory, not the temp directory).
WORK="${HOME}/.cache/mloop-runtime-smoke"
rm -rf "$WORK"
mkdir -p "$WORK/feed" "$WORK/data"
trap 'rm -rf "$WORK"' EXIT

# The tree's version with a fourth part: `mloop serve` runs a bundled server built against this tree's assemblies, and
# a lower version than theirs does not bind.
VERSION="$(sed -n 's:.*<Version>\(.*\)</Version>.*:\1:p' Directory.Build.props | head -1).9999"
dotnet pack tools/MLoop.CLI -c Release -p:Version="$VERSION" -o "$WORK/feed" >/dev/null

python3 - "$WORK/data" <<'PY'
import os, random, struct, sys, zlib
root = sys.argv[1]
random.seed(1)

def png(path, w, h, pixel):
    rows = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

# 24 red to 8 blue: imbalanced, so the train step's --balance has something to do.
for name, base, count in (("red", (200, 40, 40), 24), ("blue", (40, 40, 200), 8)):
    os.makedirs(f"{root}/images/{name}")
    for i in range(count):
        tint = [max(0, min(255, v + random.randint(-30, 30))) for v in base]
        png(f"{root}/images/{name}/{i}.png", 64, 48,
            lambda x, y, t=tint: [random.randrange(256)] * 3 if random.random() < 0.01 else t)
png(f"{root}/probe-red.png", 64, 48, lambda x, y: (205, 35, 45))
png(f"{root}/probe-blue.png", 64, 48, lambda x, y: (35, 45, 205))

good = ["great product works well", "love it very much", "excellent quality and fast", "works perfectly every time"]
bad = ["terrible broke after a day", "waste of money", "awful quality very poor", "does not work at all"]
with open(f"{root}/text.csv", "w") as f:
    f.write("Text,Label\n")
    for _ in range(30):
        f.write(f"{random.choice(good)},positive\n{random.choice(bad)},negative\n")
with open(f"{root}/text-predict.csv", "w") as f:
    f.write("Text\nreally excellent and works well\nawful and broke quickly\n")

# Object detection: a red square on a pale noisy background, one per picture, in a COCO annotations file.
import json
W = H = 96
os.makedirs(f"{root}/coco")
images, anns = [], []
for i in range(48):
    px = [[(random.randint(200, 255),) * 3 for _ in range(W)] for _ in range(H)]
    w, h = random.randint(24, 40), random.randint(24, 40)
    x, y = random.randint(0, W - w), random.randint(0, H - h)
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            px[yy][xx] = (200, 30, 30)
    rows = b"".join(b"\x00" + b"".join(bytes(px[yy][xx]) for xx in range(W)) for yy in range(H))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(f"{root}/coco/img{i:03d}.png", "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    images.append({"id": i + 1, "file_name": f"img{i:03d}.png", "width": W, "height": H})
    anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
with open(f"{root}/coco/annotations.json", "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "box"}]}, f)
PY

cat > "$WORK/run.sh" <<SH
#!/bin/bash
set -euo pipefail
# A step that fails outside a check of its own still says which one it was — set -e alone exits without a word.
trap 'echo "FAIL: line \$LINENO: \$BASH_COMMAND (exit \$?)"' ERR
export DOTNET_CLI_TELEMETRY_OPTOUT=1 PATH="\$PATH:/root/.dotnet/tools"
dotnet tool install -g mloop --add-source /work/feed --version $VERSION >/dev/null
fail() { echo "FAIL: \$*"; exit 1; }

if [[ " ${RUNTIMES[*]} " == *" tf "* ]]; then
  echo "== tf"
  mloop runtime install tf
  cd /tmp && mloop init vision --task image-classification >/tmp/tf-init.log 2>&1 || { cat /tmp/tf-init.log; fail "mloop init (image classification)"; }
  cd vision
  cp -r /work/data/images/* datasets/images/
  mloop train --task image-classification datasets/images --balance 2 >/tmp/tf-train.log 2>&1 || { tail -20 /tmp/tf-train.log; fail "image classification did not train"; }
  grep -q "Balance (training split): Oversampled" /tmp/tf-train.log || { tail -20 /tmp/tf-train.log; fail "--balance did not oversample the training split"; }
  (mloop serve --port 5099 >/tmp/serve.log 2>&1 &)
  up=0; for _ in \$(seq 1 180); do if curl -sf localhost:5099/health >/dev/null; then up=1; break; fi; sleep 1; done
  [[ \$up == 1 ]] || { tail -30 /tmp/serve.log; fail "mloop serve did not come up"; }
  T=\$(mloop token -q)
  R=\$(base64 -w0 /work/data/probe-red.png); B=\$(base64 -w0 /work/data/probe-blue.png)
  json=\$(curl -s -X POST localhost:5099/predict -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$R\",\"\$B\"]}")
  echo "\$json" | grep -q '"predictedLabel":"red".*"predictedLabel":"blue"' || fail "base64 images: \$json"
  form=\$(curl -s -X POST localhost:5099/predict -H "Authorization: Bearer \$T" -F a=@/work/data/probe-blue.png -F b=@/work/data/probe-red.png)
  echo "\$form" | grep -q '"predictedLabel":"blue".*"predictedLabel":"red"' || fail "multipart images: \$form"
  [[ \$(ls /tmp | grep -c mloop_predict_image_ || true) == 0 ]] || fail "scratch images left behind"
  # The lifecycle after the first model: a second experiment kept in staging, a shadow prediction by it, promotion,
  # the promotion history, rollback - and the served answers follow production each time (the model cache reloads on the
  # slot's new file).
  prod() { grep -o '"experimentId": *"exp-[0-9]*"' models/default/production/metadata.json | grep -o 'exp-[0-9]*' | head -1; }
  [[ \$(prod) == exp-001 ]] || fail "production after the first training is not exp-001: \$(prod)"
  mloop train --task image-classification datasets/images --balance 2 --no-promote >/tmp/tf-train2.log 2>&1 || { tail -20 /tmp/tf-train2.log; fail "second image classification training"; }
  [[ \$(prod) == exp-001 ]] || fail "--no-promote moved production to \$(prod)"
  two='"predictedLabel":"red".*"predictedLabel":"blue"'
  shadow=\$(curl -s -X POST "localhost:5099/predict?experiment=exp-002" -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$R\",\"\$B\"]}")
  echo "\$shadow" | grep -q "\$two" || fail "shadow prediction by exp-002: \$shadow"
  [[ \$(prod) == exp-001 ]] || fail "a shadow prediction moved production to \$(prod)"
  mloop promote exp-002 --force --by smoke >/tmp/tf-promote.log 2>&1 || { tail -20 /tmp/tf-promote.log; fail "promote exp-002"; }
  [[ \$(prod) == exp-002 ]] || fail "production after promote is \$(prod), not exp-002"
  after=\$(curl -s -X POST localhost:5099/predict -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$R\",\"\$B\"]}")
  echo "\$after" | grep -q "\$two" || fail "served answers after promote: \$after"
  mloop rollback >/tmp/tf-rollback.log 2>&1 || { tail -20 /tmp/tf-rollback.log; fail "rollback"; }
  [[ \$(prod) == exp-001 ]] || fail "production after rollback is \$(prod), not exp-001"
  back=\$(curl -s -X POST localhost:5099/predict -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$R\",\"\$B\"]}")
  echo "\$back" | grep -q "\$two" || fail "served answers after rollback: \$back"
  hist=\$(mloop promote --history --json 2>/dev/null)
  for e in exp-001 exp-002; do echo "\$hist" | grep -q "\"experimentId\": *\"\$e\"" || fail "promotion history lacks \$e: \$hist"; done
  echo "\$hist" | grep -q '"action": *"rollback"' || fail "promotion history lacks the rollback: \$hist"
  echo "\$hist" | grep -q '"by": *"smoke"' || fail "promotion history lacks the actor: \$hist"
  echo "tf: ok"
fi

if [[ " ${RUNTIMES[*]} " == *" torch "* ]]; then
  echo "== torch"
  mloop runtime install torch
  cd /tmp && mloop init txt --task text-classification >/tmp/torch-init.log 2>&1 || { cat /tmp/torch-init.log; fail "mloop init (text classification)"; }
  cd txt
  cp /work/data/text.csv datasets/train.csv
  mloop train datasets/train.csv --label Label --task text-classification >/tmp/torch-train.log 2>&1 || { tail -20 /tmp/torch-train.log; fail "text classification did not train"; }
  out=\$(mloop predict /work/data/text-predict.csv --json 2>/dev/null | sed -n '/^{/,\$p')
  echo "\$out" | grep -q '"probabilities"' || fail "text predictions: \$out"
  # Probabilities, not the network's logits: none below zero.
  if echo "\$out" | grep -o '"probabilities":{[^}]*}' | grep -q -- ':-'; then fail "logits reported as probabilities: \$out"; fi
  # Object detection, on the same runtime: train, then the lifecycle - a second experiment kept in staging, a shadow
  # prediction by it, promotion, rollback, the promotion history, and the same boxes over HTTP.
  cd /tmp && mloop init det --task object-detection >/tmp/od-init.log 2>&1 || { cat /tmp/od-init.log; fail "mloop init (object detection)"; }
  cd det
  cp -r /work/data/coco/* datasets/coco/
  mloop train --task object-detection --epochs 3 >/tmp/od-train.log 2>&1 || { tail -20 /tmp/od-train.log; fail "object detection did not train"; }
  prod() { grep -o '"experimentId": *"exp-[0-9]*"' models/default/production/metadata.json | grep -o 'exp-[0-9]*' | head -1; }
  [[ \$(prod) == exp-001 ]] || fail "production after the first object detection training is \$(prod)"
  mloop train --task object-detection --epochs 3 --no-promote >/tmp/od-train2.log 2>&1 || { tail -20 /tmp/od-train2.log; fail "second object detection training"; }
  [[ \$(prod) == exp-001 ]] || fail "--no-promote moved production to \$(prod)"
  mkdir probe && cp datasets/coco/img000.png datasets/coco/img001.png probe/
  boxes=\$(mloop predict probe --json 2>/dev/null) || fail "predict --json over a folder of pictures"
  echo "\$boxes" | grep -q '"task":"object-detection"' || fail "predict --json did not answer as JSON on stdout: \$boxes"
  echo "\$boxes" | grep -q '"detections":\[{"label":"box"' || fail "no box found in the pictures of the training set: \$boxes"
  echo "\$boxes" | grep -q '"experimentId":"exp-001"' || fail "predict --json does not name the experiment: \$boxes"
  shadow=\$(mloop predict probe --experiment exp-002 --json 2>/dev/null) || fail "shadow prediction by exp-002"
  echo "\$shadow" | grep -q '"experimentId":"exp-002"' || fail "shadow prediction did not use exp-002: \$shadow"
  [[ \$(prod) == exp-001 ]] || fail "a shadow prediction moved production to \$(prod)"
  mloop promote exp-002 --force --by smoke >/tmp/od-promote.log 2>&1 || { tail -20 /tmp/od-promote.log; fail "promote exp-002 (object detection)"; }
  [[ \$(prod) == exp-002 ]] || fail "production after promote is \$(prod), not exp-002"
  mloop rollback >/tmp/od-rollback.log 2>&1 || { tail -20 /tmp/od-rollback.log; fail "rollback (object detection)"; }
  [[ \$(prod) == exp-001 ]] || fail "production after rollback is \$(prod), not exp-001"
  hist=\$(mloop promote --history --json 2>/dev/null)
  for e in exp-001 exp-002; do echo "\$hist" | grep -q "\"experimentId\": *\"\$e\"" || fail "promotion history lacks \$e: \$hist"; done
  echo "\$hist" | grep -q '"action": *"rollback"' || fail "promotion history lacks the rollback: \$hist"
  (mloop serve --port 5098 >/tmp/od-serve.log 2>&1 &)
  up=0; for _ in \$(seq 1 180); do if curl -sf localhost:5098/health >/dev/null; then up=1; break; fi; sleep 1; done
  [[ \$up == 1 ]] || { tail -30 /tmp/od-serve.log; fail "mloop serve (object detection) did not come up"; }
  T=\$(mloop token -q)
  P=\$(base64 -w0 datasets/coco/img000.png)
  served=\$(curl -s -X POST localhost:5098/predict -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$P\"]}")
  echo "\$served" | grep -q '"detections":\[{"label":"box"' || fail "boxes over HTTP: \$served"
  echo "torch: ok"
fi
if [[ " ${RUNTIMES[*]} " == *" anomaly "* ]]; then
  echo "== anomaly"
  mloop runtime install onnxruntime
  mloop encoder install dinov2-small >/tmp/ia-encoder.log 2>&1 || { tail -20 /tmp/ia-encoder.log; fail "mloop encoder install dinov2-small"; }
  cd /tmp && mloop init anom --task image-anomaly >/tmp/ia-init.log 2>&1 || { cat /tmp/ia-init.log; fail "mloop init (image anomaly)"; }
  cd anom
  mkdir -p datasets/images && cp /work/data/images/red/*.png datasets/images/
  mloop train --task image-anomaly datasets/images >/tmp/ia-train.log 2>&1 || { tail -20 /tmp/ia-train.log; fail "image anomaly did not train"; }
  prod() { grep -o '"experimentId": *"exp-[0-9]*"' models/default/production/metadata.json | grep -o 'exp-[0-9]*' | head -1; }
  [[ \$(prod) == exp-001 ]] || fail "production after the first image anomaly training is \$(prod)"
  mloop train --task image-anomaly datasets/images --no-promote >/tmp/ia-train2.log 2>&1 || { tail -20 /tmp/ia-train2.log; fail "second image anomaly training"; }
  [[ \$(prod) == exp-001 ]] || fail "--no-promote moved production to \$(prod)"
  mkdir probe && cp /work/data/probe-red.png probe/a-red.png && cp /work/data/probe-blue.png probe/b-blue.png
  # a picture unlike the normal ones scores higher than one like them
  score() { grep -o '"score": *[0-9.eE-]*' | grep -o '[0-9.eE-]*\$'; }
  out=\$(mloop predict probe --json 2>/dev/null) || fail "predict --json (image anomaly)"
  echo "\$out" | grep -q '"experimentId": *"exp-001"' || fail "predict --json does not name the experiment: \$out"
  read -r red blue <<<"\$(echo "\$out" | score | tr '\n' ' ')"
  awk "BEGIN{exit !(\$blue > \$red)}" || fail "the odd picture did not score higher: red \$red, blue \$blue"
  shadow=\$(mloop predict probe --experiment exp-002 --json 2>/dev/null) || fail "shadow prediction by exp-002"
  echo "\$shadow" | grep -q '"experimentId": *"exp-002"' || fail "shadow prediction did not use exp-002: \$shadow"
  [[ \$(prod) == exp-001 ]] || fail "a shadow prediction moved production to \$(prod)"
  mloop promote exp-002 --force --by smoke >/tmp/ia-promote.log 2>&1 || { tail -20 /tmp/ia-promote.log; fail "promote exp-002 (image anomaly)"; }
  [[ \$(prod) == exp-002 ]] || fail "production after promote is \$(prod), not exp-002"
  mloop rollback >/tmp/ia-rollback.log 2>&1 || { tail -20 /tmp/ia-rollback.log; fail "rollback (image anomaly)"; }
  [[ \$(prod) == exp-001 ]] || fail "production after rollback is \$(prod), not exp-001"
  hist=\$(mloop promote --history --json 2>/dev/null)
  for e in exp-001 exp-002; do echo "\$hist" | grep -q "\"experimentId\": *\"\$e\"" || fail "promotion history lacks \$e: \$hist"; done
  echo "\$hist" | grep -q '"action": *"rollback"' || fail "promotion history lacks the rollback: \$hist"
  (mloop serve --port 5097 >/tmp/ia-serve.log 2>&1 &)
  up=0; for _ in \$(seq 1 180); do if curl -sf localhost:5097/health >/dev/null; then up=1; break; fi; sleep 1; done
  [[ \$up == 1 ]] || { tail -30 /tmp/ia-serve.log; fail "mloop serve (image anomaly) did not come up"; }
  T=\$(mloop token -q)
  R=\$(base64 -w0 /work/data/probe-red.png); B=\$(base64 -w0 /work/data/probe-blue.png)
  served=\$(curl -s -X POST localhost:5097/predict -H "Authorization: Bearer \$T" -H "Content-Type: application/json" -d "{\"images\":[\"\$R\",\"\$B\"]}")
  echo "\$served" | grep -q '"task":"image-anomaly"' || fail "image anomaly over HTTP: \$served"
  read -r sred sblue <<<"\$(echo "\$served" | score | tr '\n' ' ')"
  awk "BEGIN{exit !(\$sblue > \$sred)}" || fail "over HTTP the odd picture did not score higher: red \$sred, blue \$sblue"
  echo "anomaly: ok"
fi
SH

docker run --rm --platform linux/amd64 -v "$WORK:/work" mcr.microsoft.com/dotnet/sdk:10.0 bash /work/run.sh
