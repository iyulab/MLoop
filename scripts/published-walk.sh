#!/bin/bash
#
# The first thirty minutes of someone who installs the tool: a clean Linux container that has only the .NET SDK, the tool
# packed from this tree into a local feed (nothing is published, nothing is pushed), and generated tables standing in for
# "the user's own file". A reader who knows only the README follows it inside that container.
#
#   published-walk.sh up [--platform linux/arm64|linux/amd64]   pack the tool, generate the tables, start the container
#   published-walk.sh smoke                                     walk the scenarios with a script (a quick check, not a judge)
#   published-walk.sh down                                      remove the container and the work directory
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
IMAGE="mcr.microsoft.com/dotnet/sdk:10.0"

cmd="${1:-up}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

down() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }

generate_tables() {
python3 - "$1" <<'PY'
import csv, os, random, sys
root = sys.argv[1]
os.makedirs(root, exist_ok=True)
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
PY
}

case "$cmd" in
  down)
    down; rm -rf "$WORK"; echo "removed $NAME and $WORK" ;;

  up)
    down; rm -rf "$WORK"; mkdir -p "$WORK/feed" "$WORK/data"
    # A version above anything on nuget.org, so that `dotnet tool install mloop --add-source /feed` takes the packed tool and
    # not the published one. (The bundled server binds only against a tool at least as new as the tree's assemblies.)
    TREE="$(sed -n 's:.*<Version>\(.*\)</Version>.*:\1:p' Directory.Build.props | head -1)"
    VERSION="${TREE%%.*}.$(( $(echo "$TREE" | cut -d. -f2) + 1 )).0.9999"
    echo "== pack $VERSION"
    dotnet pack tools/MLoop.CLI -c Release -p:Version="$VERSION" -o "$WORK/feed" >/dev/null
    generate_tables "$WORK/data"
    # What a reader has: the README as packed with the tool, and the guides it links to (public in the repository).
    mkdir -p "$WORK/docs"
    unzip -p "$WORK"/feed/mloop.*.nupkg README.md > "$WORK/docs/README.md"
    cp docs/*.md "$WORK/docs/"
    echo "$VERSION" > "$WORK/VERSION"
    echo "== container ($PLATFORM)"
    docker run -d --name "$NAME" --platform "$PLATFORM" \
      -v "$WORK/feed:/feed:ro" -v "$WORK/data:/data:ro" -v "$WORK/docs:/docs:ro" -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_NOLOGO=1 \
      "$IMAGE" sleep infinity >/dev/null
    # Left as a user would find it: the SDK image does not put the global tool directory on the PATH (the README says how).
    docker exec "$NAME" mkdir -p /root/work
    echo "ready: docker exec $NAME bash -lc '<command>'   (feed /feed, tables /data, docs /docs, version $VERSION)"
    ;;

  smoke)
    [[ -f "$WORK/VERSION" ]] || { echo "run '$0 up' first" >&2; exit 2; }
    exec "$HERE/published-walk-smoke.sh" "$NAME"
    ;;

  *) echo "usage: $0 up|smoke|down" >&2; exit 2 ;;
esac
