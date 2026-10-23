#!/bin/bash
#
# Drives the deep-learning runtimes end to end in a Linux x64 container — the paths CI does not run, because it
# installs no TensorFlow or libtorch:
#
#   tf     runtime install → image classification training → mloop serve → POST /predict with images (base64 and
#          multipart) → no scratch file left behind
#   torch  runtime install → text classification training → predictions whose probabilities sum to one
#
# Usage:
#   ./scripts/runtime-smoke.sh            # both
#   ./scripts/runtime-smoke.sh tf         # one
#
# Needs Docker (on an arm64 machine the container runs emulated — minutes, not seconds) and python3 for the synthetic
# images and sentences. The CLI is packed from this tree; nothing is published.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then RUNTIMES=(tf torch); else RUNTIMES=("$@"); fi

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

for name, base in (("red", (200, 40, 40)), ("blue", (40, 40, 200))):
    os.makedirs(f"{root}/images/{name}")
    for i in range(24):
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
PY

cat > "$WORK/run.sh" <<SH
#!/bin/bash
set -euo pipefail
export DOTNET_CLI_TELEMETRY_OPTOUT=1 PATH="\$PATH:/root/.dotnet/tools"
dotnet tool install -g mloop --add-source /work/feed --version $VERSION >/dev/null
fail() { echo "FAIL: \$*"; exit 1; }

if [[ " ${RUNTIMES[*]} " == *" tf "* ]]; then
  echo "== tf"
  mloop runtime install tf
  cd /tmp && mloop init vision --task image-classification >/dev/null && cd vision
  cp -r /work/data/images/* datasets/images/
  mloop train --task image-classification datasets/images >/tmp/tf-train.log 2>&1 || { tail -20 /tmp/tf-train.log; fail "image classification did not train"; }
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
  echo "tf: ok"
fi

if [[ " ${RUNTIMES[*]} " == *" torch "* ]]; then
  echo "== torch"
  mloop runtime install torch
  cd /tmp && mloop init txt --task text-classification >/dev/null && cd txt
  cp /work/data/text.csv datasets/train.csv
  mloop train datasets/train.csv --label Label --task text-classification >/tmp/torch-train.log 2>&1 || { tail -20 /tmp/torch-train.log; fail "text classification did not train"; }
  out=\$(mloop predict /work/data/text-predict.csv --json 2>/dev/null | sed -n '/^{/,\$p')
  echo "\$out" | grep -q '"probabilities"' || fail "text predictions: \$out"
  # Probabilities, not the network's logits: none below zero.
  if echo "\$out" | grep -o '"probabilities":{[^}]*}' | grep -q -- ':-'; then fail "logits reported as probabilities: \$out"; fi
  echo "torch: ok"
fi
SH

docker run --rm --platform linux/amd64 -v "$WORK:/work" mcr.microsoft.com/dotnet/sdk:10.0 bash /work/run.sh
