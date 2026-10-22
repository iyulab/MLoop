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

set -u
cd "$(dirname "$0")/.."

FILTER=""
COVERAGE=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --ci|--exclude-heavy) FILTER="Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database"; shift ;;
        --coverage) COVERAGE=(--collect:"XPlat Code Coverage"); shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--ci] [--coverage]"; exit 1 ;;
    esac
done

FAILED=()
for project in tests/*/*.Tests.csproj; do
    echo "=== $project"
    if ! dotnet test "$project" --configuration Release --nologo ${FILTER:+--filter "$FILTER"} "${COVERAGE[@]+"${COVERAGE[@]}"}"; then
        FAILED+=("$project")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All test projects passed."
else
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
