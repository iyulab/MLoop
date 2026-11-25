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
#   published-walk.sh exec [--timeout <s>] -- '<bash command>'  run a command in the container, from /root/work, and wait for it
#   published-walk.sh fetch <container path> <local directory>  copy a file or folder out of the container (to look at pictures)
#   published-walk.sh down                                      remove the container and the work directory
#
# With `--machine <name>` (on `up`, `exec`, `fetch` and `down`) the container runs in the Docker of another machine - a Windows
# machine reached by command execution, which gives a native Linux x64 container where this one would have to emulate it. The
# machine's address and key are read from the file MLOOP_WALK_WINDOWS_ENV names (walk_remote.py).
#
# Pictures are generated too (/data/pictures — walk-pictures.py says what is in it; what it knows is in "$WORK/truth/pictures.json").
# Series, ranking and recommendation tables are generated too. What the generator knows (the noise-free series, the spike rows,
# each ranking query's best row, the ratings behind the recommendation pairs) is written to "$WORK/truth/" on the host and is NOT
# mounted into the container: the reader sees tables, whoever checks the answers compares them with the truth.
#
# After `up`, the reader runs commands with:   published-walk.sh exec -- '<command>'
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
MACHINE=""
TIMEOUT=3600
IMAGE="mcr.microsoft.com/dotnet/sdk:10.0"

cmd="${1:-up}"; shift || true
REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --published) PUBLISHED="$2"; shift 2 ;;
    --machine) MACHINE="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --) shift; REST+=("$@"); break ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) REST+=("$1"); shift ;;
  esac
done
remote() { python3 -I "$HERE/published-walk-remote.py" "$@" --machine "$MACHINE" --container "$NAME"; }

down() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }

generate_tables() { python3 -I "$HERE/walk-tables.py" "$1" "$2"; python3 -I "$HERE/walk-pictures.py" "$1/pictures" "$2"; }

case "$cmd" in
  down)
    if [[ -n "$MACHINE" ]]; then remote down; else down; fi
    rm -rf "$WORK"; echo "removed $NAME and $WORK" ;;

  exec)
    [[ ${#REST[@]} -gt 0 ]] || { echo "exec needs a command after --" >&2; exit 2; }
    if [[ -n "$MACHINE" ]]; then
      python3 -I "$HERE/published-walk-remote.py" exec --machine "$MACHINE" --container "$NAME" --timeout "$TIMEOUT" -- "${REST[@]}"
    else
      docker exec -w /root/work "$NAME" bash -lc "${REST[*]}"
    fi ;;

  fetch)
    [[ ${#REST[@]} -eq 2 ]] || { echo "fetch needs <container path> <local directory>" >&2; exit 2; }
    if [[ -n "$MACHINE" ]]; then remote fetch "${REST[0]}" "${REST[1]}"
    else mkdir -p "${REST[1]}"; docker cp "$NAME:${REST[0]}" "${REST[1]}"; fi ;;

  up)
    [[ -n "$MACHINE" ]] || down; rm -rf "$WORK"; mkdir -p "$WORK/feed" "$WORK/data"
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
    if [[ -n "$MACHINE" ]]; then
      echo "== container (native linux/amd64 on another machine)"
      remote up --work "$WORK" --image "$IMAGE"
      echo "ready: published-walk.sh exec --machine <name> -- '<command>'   (feed /feed, tables and pictures /data, docs /docs, version $VERSION)"
      exit 0
    fi
    echo "== container ($PLATFORM)"
    docker run -d --name "$NAME" --platform "$PLATFORM" \
      -v "$WORK/feed:/feed:ro" -v "$WORK/data:/data:ro" -v "$WORK/docs:/docs:ro" -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_NOLOGO=1 \
      "$IMAGE" sleep infinity >/dev/null
    # Left as a user would find it: the SDK image does not put the global tool directory on the PATH (the README says how).
    docker exec "$NAME" mkdir -p /root/work
    echo "ready: published-walk.sh exec -- '<command>'   (feed /feed, tables and pictures /data, docs /docs, version $VERSION)"
    ;;

  *) echo "usage: $0 up|exec|fetch|down" >&2; exit 2 ;;
esac
