#!/bin/bash
#
# MLoop Test Runner - Runs all tests including heavy tests
#
# Usage:
#   ./scripts/test-all.sh              # Run all tests
#   ./scripts/test-all.sh --ci         # Run only CI tests (exclude heavy)
#   ./scripts/test-all.sh --category LLM  # Run only LLM tests
#   ./scripts/test-all.sh --coverage   # Run with code coverage
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
CATEGORY=""
EXCLUDE_HEAVY=false
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ci|--exclude-heavy)
            EXCLUDE_HEAVY=true
            shift
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ci] [--category CATEGORY] [--coverage]"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}MLoop Test Runner${NC}"
echo -e "${CYAN}=================${NC}"
echo ""

# Build filter
FILTER=""
if [ -n "$CATEGORY" ]; then
    FILTER="--filter \"Category=$CATEGORY\""
    echo -e "${YELLOW}Running $CATEGORY tests only${NC}"
elif [ "$EXCLUDE_HEAVY" = true ]; then
    FILTER='--filter "Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database"'
    echo -e "${YELLOW}Running lightweight tests only (CI mode)${NC}"
else
    echo -e "${GREEN}Running ALL tests (including heavy tests)${NC}"
fi

# Check for API keys if running LLM tests
if [ "$EXCLUDE_HEAVY" = false ] && [ -z "$CATEGORY" -o "$CATEGORY" = "LLM" ]; then
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$AZURE_OPENAI_API_KEY" ]; then
        echo ""
        echo -e "${YELLOW}Warning: No LLM API keys found. LLM tests may be skipped.${NC}"
        echo -e "${YELLOW}Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable for LLM tests.${NC}"
        echo ""
    fi
fi

# Build coverage option
COVERAGE_OPT=""
if [ "$COVERAGE" = true ]; then
    COVERAGE_OPT='--collect:"XPlat Code Coverage"'
    echo -e "${CYAN}Code coverage enabled${NC}"
fi

echo ""

# Run tests
echo -e "Executing: dotnet test MLoop.sln --configuration Release --verbosity normal $FILTER $COVERAGE_OPT"
echo ""

eval "dotnet test MLoop.sln --configuration Release --verbosity normal $FILTER $COVERAGE_OPT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}Some tests failed. Exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
