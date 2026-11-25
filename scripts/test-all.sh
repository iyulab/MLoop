#!/bin/bash
#
# Runs every test project, one at a time.
#
# Usage:
#   ./scripts/test-all.sh              # every test, Slow included
#   ./scripts/test-all.sh --ci         # the set CI runs (Slow, wall-clock-bound tests excluded)
#   ./scripts/test-all.sh --coverage   # collect code coverage
#
# Serially, never two at once: several suites run real trainer searches on a ten-second budget, and
# under CPU load no trial completes — a failure that describes the machine, not the code. One project
# per `dotnet test` also keeps the results of the others when one dies, and says which one it was.
# The projects are found, not listed, so a new test project cannot be left out.
#
# A zero exit code is not enough to count a project as passed: a run has been seen to end with exit 0 and no summary at
# all. Each project must also print a summary with at least one test. The summary line is localized, so the CLI language
# is pinned to English for the run.

set -u
cd "$(dirname "$0")/.."
export DOTNET_CLI_UI_LANGUAGE=en
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

FILTER=""
COVERAGE=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --ci|--exclude-heavy) FILTER="Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database"; shift ;;
        --coverage) COVERAGE=(--coverage --coverage-output-format cobertura); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--ci] [--coverage]"; exit 1 ;;
    esac
done

FAILED=()
for project in tests/*/*.Tests.csproj; do
    echo "=== $project"
    dotnet test "$project" --configuration Release --nologo ${FILTER:+--filter "$FILTER"} "${COVERAGE[@]+"${COVERAGE[@]}"}" 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        FAILED+=("$project")
    elif ! sed $'s/\x1b\\[[0-9;]*m//g' "$LOG" | grep -qE '^[[:space:]]*total: [1-9][0-9]*'; then
        FAILED+=("$project (exit 0, but no test summary)")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All test projects passed."
else
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
