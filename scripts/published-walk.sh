#!/bin/bash
#
# The first thirty minutes of someone who installs the tool: a clean Linux container that has only the .NET SDK, the tool
# packed from this tree into a local feed (nothing is published, nothing is pushed), and generated tables standing in for
# "the user's own file". A reader who knows only the README follows it inside that container.
#
#   published-walk.sh up [--platform linux/arm64|linux/amd64]   pack the tool, generate the tables, start the container
#   published-walk.sh up --published <version> [--platform ...] the same, with no packed tool: the reader installs the real
#                                                               <version> from nuget.org (plain `dotnet tool install --global mloop`);
#                                                               the README and guides are those of the tag v<version>
#   published-walk.sh down                                      remove the container and the work directory
#
# Series, ranking and recommendation tables are generated too. What the generator knows (the noise-free series, the spike rows,
# each ranking query's best row, the ratings behind the recommendation pairs) is written to "$WORK/truth/" on the host and is NOT
# mounted into the container: the reader sees tables, whoever checks the answers compares them with the truth.
#
# After `up`, the reader runs commands with:   docker exec mloop-walk bash -lc '<command>'
# The feed is at /feed (install with --add-source /feed), the tables are in /data, the README (as packed) and the guides it links
# to are in /docs, and the home directory is empty.
#
# Needs Docker and python3. On an arm64 machine linux/arm64 is native and fast, but there is no LightGBM native for it, so the
# trainer set differs from the one most users get; linux/amd64 is emulated (minutes, not seconds) and is run once before a
# stage is accepted.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

NAME="${MLOOP_WALK_CONTAINER:-mloop-walk}"
WORK="${HOME}/.cache/mloop-published-walk"
PLATFORM="linux/arm64"
PUBLISHED=""
IMAGE="mcr.microsoft.com/dotnet/sdk:10.0"

cmd="${1:-up}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --published) PUBLISHED="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

down() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }

generate_tables() {
python3 - "$1" "$2" <<'PY'
import csv, json, math, os, random, sys
root = sys.argv[1]
truth_root = sys.argv[2]
os.makedirs(root, exist_ok=True)
os.makedirs(truth_root, exist_ok=True)
rnd = random.Random(7)

def write(name, header, rows):
    with open(os.path.join(root, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# Binary: a subscription that is cancelled (yes/no).
plans = ["basic", "pro", "team"]
regions = ["north", "south", "east", "west"]
def customer(shift=0.0):
    tenure = rnd.randint(1, 60)
    plan = rnd.choice(plans)
    monthly = round(rnd.gauss(30 + 20 * plans.index(plan), 6) + shift, 2)
    calls = max(0, int(rnd.gauss(2, 1.5)))
    risk = 0.08 * calls + (0.4 if plan == "basic" else 0) - 0.012 * tenure + rnd.gauss(0, 0.15)
    return [tenure, monthly, plan, rnd.choice(regions), calls], "yes" if risk > 0.25 else "no"
head = ["tenure_months", "monthly_charge", "plan", "region", "support_calls"]
rows = []
for _ in range(700):
    x, y = customer(); rows.append(x + [y])
write("binary-train.csv", head + ["cancelled"], rows)
write("binary-new.csv", head, [customer()[0] for _ in range(25)])
write("binary-moved.csv", head, [customer(shift=70.0)[0] for _ in range(60)])
write("binary-moved-labels.csv", head + ["cancelled"], [customer(shift=70.0)[0] + [rnd.choice(["yes", "no"])] for _ in range(60)])

# Multiclass: three kinds of leaf, from four measurements.
kinds = {"oak": (11, 6, 3, 1.0), "maple": (8, 8, 2, 0.5), "birch": (5, 3, 4, 2.0)}
def leaf(kind=None):
    kind = kind or rnd.choice(list(kinds))
    a, b, c, d = kinds[kind]
    return [round(rnd.gauss(a, 1.2), 2), round(rnd.gauss(b, 1.0), 2), round(rnd.gauss(c, 0.6), 2), round(rnd.gauss(d, 0.4), 2)], kind
lhead = ["length_cm", "width_cm", "lobes", "vein_density"]
rows = []
for _ in range(450):
    x, y = leaf(); rows.append(x + [y])
write("multiclass-train.csv", lhead + ["species"], rows)
write("multiclass-new.csv", lhead, [leaf()[0] for _ in range(20)])

# Regression: the sale price of a flat.
areas = ["centre", "harbour", "hills", "suburb"]
def flat():
    sqm = rnd.randint(25, 140); rooms = max(1, sqm // 30 + rnd.randint(-1, 1)); age = rnd.randint(0, 80)
    area = rnd.choice(areas)
    price = 1200 * sqm + 9000 * rooms - 700 * age + 40000 * areas.index(area) + rnd.gauss(0, 15000)
    return [sqm, rooms, age, area], round(price)
fhead = ["area_sqm", "rooms", "age_years", "district"]
rows = []
for _ in range(500):
    x, y = flat(); rows.append(x + [y])
write("regression-train.csv", fhead + ["price"], rows)
write("regression-new.csv", fhead, [flat()[0] for _ in range(20)])

# Deliberate mistakes.
open(os.path.join(root, "empty.csv"), "w").close()
write("single-class.csv", head + ["cancelled"], [customer()[0] + ["no"] for _ in range(80)])
write("binary-missing-column.csv", ["tenure_months", "plan", "region", "support_calls"], [[r[0], r[2], r[3], r[4]] for r in (customer()[0] for _ in range(15))])
# ---- Series, ranking, recommendation: a separate generator, so the tables above stay byte for byte as they were. ----
rs = random.Random(11)
truth = {}

# A series: trend + a 24-step cycle + small noise. Row i (0-based) is 50 + 0.1 i + 10 sin(2 pi i / 24) + noise (sd 0.5).
def clean(i): return 50 + 0.1 * i + 10 * math.sin(2 * math.pi * i / 24)
def noisy(i): return round(clean(i) + rs.gauss(0, 0.5), 3)
write("series.csv", ["value"], [[noisy(i)] for i in range(300)])
truth["series"] = {"rows": 300, "horizon_default": 10, "noise_free_next": [round(clean(300 + k), 3) for k in range(10)],
                   "tolerance": 5}

# The same series, 400 rows, with three spikes of eight noise-widths at known data rows (1-based, header excluded). A noise width is
# the 4-sigma band of the noise (sd 0.5, so 2.0): +16. Eight sigmas (+4) is no spike at all next to a cycle that moves up to 2.6 a step.
spikes = [121, 251, 331]
vals = [noisy(i) for i in range(400)]
for r in spikes: vals[r - 1] = round(vals[r - 1] + 16.0, 3)
write("series-spikes.csv", ["value"], [[v] for v in vals])
truth["series_spikes"] = {"rows": 400, "spike_data_rows": spikes, "spike_amplitude": 16.0, "noise_sd": 0.5, "max_flag_share": 0.05}

# The first series with one empty cell at data row 150.
gap = [[noisy(i)] for i in range(300)]
gap[149] = [""]
write("series-gap.csv", ["value"], gap)
truth["series_gap"] = {"empty_data_row": 150}

# Ranking: 100 queries of ten rows; relevance 0..3 is the row's rank band by 2 f1 - f2 (+ noise) inside its query.
def query(qid):
    rows = []
    for _ in range(10):
        f = [round(rs.gauss(0, 1), 3) for _ in range(4)]
        rows.append([qid] + f)
    score = [2 * r[1] - r[2] + rs.gauss(0, 0.3) for r in rows]
    order = sorted(range(10), key=lambda k: -score[k])
    rel = [0] * 10
    for place, k in enumerate(order): rel[k] = 3 if place < 2 else 2 if place < 5 else 1 if place < 8 else 0
    return rows, rel, order[0]
rank_head = ["qid", "f1", "f2", "f3", "f4"]
rows = []
for q in range(1, 101):
    qr, rel, _ = query(q)
    rows += [r + [rel[k]] for k, r in enumerate(qr)]
write("ranking-train.csv", rank_head + ["label"], rows)
rows, best = [], {}
for q in range(101, 106):
    qr, rel, top = query(q)
    best[str(q)] = top + 1 + len(rows)   # 1-based data row of the most relevant row of this query
    rows += qr
write("ranking-new.csv", rank_head, rows)
truth["ranking"] = {"new_best_data_row_by_qid": best, "need_top_in_queries": 4}

# Recommendation: ratings 1..5 from two latent factors; about a third of the user-item pairs are rated.
users = ["u%03d" % n for n in range(1, 201)]; items = ["i%02d" % n for n in range(1, 81)]
uf = {u: [rs.gauss(0, 0.8), rs.gauss(0, 0.8)] for u in users}; vf = {i: [rs.gauss(0, 0.8), rs.gauss(0, 0.8)] for i in items}
def affinity(u, i): return 3 + uf[u][0] * vf[i][0] * 2 + uf[u][1] * vf[i][1] * 2
def rating(u, i): return min(5, max(1, round(affinity(u, i) + rs.gauss(0, 0.4))))
rated, unrated = [], []
for u in users:
    for i in items: (rated if rs.random() < 0.33 else unrated).append((u, i))
write("recs-train.csv", ["user", "item", "rating"], [[u, i, rating(u, i)] for u, i in rated])
cand = sorted(rs.sample(unrated, 80), key=lambda p: affinity(*p))
pick = cand[:8] + cand[36:40] + cand[-8:]
rs.shuffle(pick)
write("recs-new.csv", ["user", "item"], [list(p) for p in pick])
truth["recommendation"] = {"pairs": [{"user": u, "item": i, "generator_rating": rating(u, i)} for u, i in pick]}
write("recs-missing-item.csv", ["user", "rating"], [[u, rating(u, i)] for u, i in rated[:200]])

with open(os.path.join(truth_root, "truth.json"), "w") as f: json.dump(truth, f, indent=1)
PY
}

case "$cmd" in
  down)
    down; rm -rf "$WORK"; echo "removed $NAME and $WORK" ;;

  up)
    down; rm -rf "$WORK"; mkdir -p "$WORK/feed" "$WORK/data"
    if [[ -n "$PUBLISHED" ]]; then
      # The real thing: nothing is packed and the feed stays empty, so a plain `dotnet tool install --global mloop` reads nuget.org.
      VERSION="$PUBLISHED"
      echo "== published $VERSION"
      generate_tables "$WORK/data" "$WORK/truth"
      mkdir -p "$WORK/docs"
      git show "v$VERSION:README.md" > "$WORK/docs/README.md"
      for f in $(git ls-tree --name-only "v$VERSION" docs/ | grep '\.md$'); do git show "v$VERSION:$f" > "$WORK/docs/$(basename "$f")"; done
    else
      # A version above anything on nuget.org, so that `dotnet tool install mloop --add-source /feed` takes the packed tool and
      # not the published one. (The bundled server binds only against a tool at least as new as the tree's assemblies.)
      TREE="$(sed -n 's:.*<Version>\(.*\)</Version>.*:\1:p' Directory.Build.props | head -1)"
      VERSION="${TREE%%.*}.$(( $(echo "$TREE" | cut -d. -f2) + 1 )).0.9999"
      echo "== pack $VERSION"
      dotnet pack tools/MLoop.CLI -c Release -p:Version="$VERSION" -o "$WORK/feed" >/dev/null
      generate_tables "$WORK/data" "$WORK/truth"
      # What a reader has: the README as packed with the tool, and the guides it links to (public in the repository).
      mkdir -p "$WORK/docs"
      unzip -p "$WORK"/feed/mloop.*.nupkg README.md > "$WORK/docs/README.md"
      cp docs/*.md "$WORK/docs/"
    fi
    echo "$VERSION" > "$WORK/VERSION"
    echo "== container ($PLATFORM)"
    docker run -d --name "$NAME" --platform "$PLATFORM" \
      -v "$WORK/feed:/feed:ro" -v "$WORK/data:/data:ro" -v "$WORK/docs:/docs:ro" -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_NOLOGO=1 \
      "$IMAGE" sleep infinity >/dev/null
    # Left as a user would find it: the SDK image does not put the global tool directory on the PATH (the README says how).
    docker exec "$NAME" mkdir -p /root/work
    echo "ready: docker exec $NAME bash -lc '<command>'   (feed /feed, tables /data, docs /docs, version $VERSION)"
    ;;

  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
