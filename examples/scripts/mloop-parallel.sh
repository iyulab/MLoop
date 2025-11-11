#!/bin/bash
# MLoop GNU Parallel Examples
# Demonstrates advanced parallel job execution with GNU Parallel
#
# Prerequisites:
#   sudo apt-get install parallel  # Ubuntu/Debian
#   brew install parallel           # macOS
#
# Usage:
#   ./mloop-parallel.sh [example-number]
#
# Examples:
#   ./mloop-parallel.sh 1    # Basic sequential execution
#   ./mloop-parallel.sh 2    # Parallel with 2 jobs
#   ./mloop-parallel.sh 3    # Resume on failure
#   ./mloop-parallel.sh 4    # Different time budgets per job
#   ./mloop-parallel.sh 5    # Progress bar
#   ./mloop-parallel.sh 6    # Dry run (preview commands)

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null; then
    echo -e "${RED}Error: GNU Parallel not found${NC}"
    echo ""
    echo "Please install GNU Parallel:"
    echo "  Ubuntu/Debian: sudo apt-get install parallel"
    echo "  macOS:         brew install parallel"
    echo "  Other:         https://www.gnu.org/software/parallel/"
    exit 1
fi

# Check if mloop is available
if ! command -v mloop &> /dev/null; then
    echo -e "${RED}Error: mloop command not found${NC}"
    echo "Please install MLoop: dotnet tool install -g mloop"
    exit 1
fi

# Display header
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MLoop + GNU Parallel Examples       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Show usage if no argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [example-number]"
    echo ""
    echo "Available examples:"
    echo "  1. Basic sequential execution (one at a time)"
    echo "  2. Parallel execution (2 jobs concurrently)"
    echo "  3. Resume on failure (with job log)"
    echo "  4. Different time budgets per dataset"
    echo "  5. Progress bar with ETA"
    echo "  6. Dry run (preview commands without execution)"
    echo ""
    echo "Example: $0 1"
    exit 0
fi

example=$1

case $example in
    1)
        echo -e "${GREEN}Example 1: Basic Sequential Execution${NC}"
        echo -e "${YELLOW}One job at a time (-j 1)${NC}"
        echo ""

        # Create sample datasets list
        echo "train examples/equipment-anomaly-detection/datasets/sensor_data.csv failure_type --time 300" > /tmp/mloop-jobs.txt
        echo "train examples/equipment-anomaly-detection/datasets/sensor_data.csv failure_type --time 600" >> /tmp/mloop-jobs.txt
        echo "train examples/equipment-anomaly-detection/datasets/sensor_data.csv failure_type --time 900" >> /tmp/mloop-jobs.txt

        echo "Jobs to execute:"
        cat /tmp/mloop-jobs.txt
        echo ""

        # Execute sequentially
        parallel -j 1 mloop {} < /tmp/mloop-jobs.txt

        rm /tmp/mloop-jobs.txt
        ;;

    2)
        echo -e "${GREEN}Example 2: Parallel Execution${NC}"
        echo -e "${YELLOW}2 jobs concurrently (-j 2)${NC}"
        echo -e "${RED}⚠️  Use only if you have sufficient CPU/GPU resources!${NC}"
        echo ""

        # Create sample jobs
        echo "train dataset1.csv target --time 300" > /tmp/mloop-jobs.txt
        echo "train dataset2.csv target --time 300" >> /tmp/mloop-jobs.txt
        echo "train dataset3.csv target --time 300" >> /tmp/mloop-jobs.txt
        echo "train dataset4.csv target --time 300" >> /tmp/mloop-jobs.txt

        echo "Jobs to execute (2 at a time):"
        cat /tmp/mloop-jobs.txt
        echo ""

        # Execute 2 jobs in parallel
        parallel -j 2 mloop {} < /tmp/mloop-jobs.txt

        rm /tmp/mloop-jobs.txt
        ;;

    3)
        echo -e "${GREEN}Example 3: Resume on Failure${NC}"
        echo -e "${YELLOW}Automatically resume from last successful job${NC}"
        echo ""

        # Create sample jobs
        echo "train dataset1.csv target --time 300" > /tmp/mloop-jobs.txt
        echo "train dataset2.csv target --time 300" >> /tmp/mloop-jobs.txt
        echo "train dataset3.csv target --time 300" >> /tmp/mloop-jobs.txt

        echo "Jobs to execute:"
        cat /tmp/mloop-jobs.txt
        echo ""
        echo -e "${YELLOW}Job log will be saved to: /tmp/mloop-joblog.txt${NC}"
        echo ""

        # Execute with job logging and resume capability
        parallel --resume --joblog /tmp/mloop-joblog.txt -j 1 mloop {} < /tmp/mloop-jobs.txt

        echo ""
        echo -e "${GREEN}Job log:${NC}"
        cat /tmp/mloop-joblog.txt

        rm /tmp/mloop-jobs.txt
        # Keep joblog for demonstration
        ;;

    4)
        echo -e "${GREEN}Example 4: Different Time Budgets${NC}"
        echo -e "${YELLOW}Each job gets different training time${NC}"
        echo ""

        # Create CSV with dataset and time budget
        cat > /tmp/mloop-jobs.csv <<EOF
dataset1.csv,300
dataset2.csv,600
dataset3.csv,900
dataset4.csv,1800
EOF

        echo "Jobs configuration (dataset,time):"
        cat /tmp/mloop-jobs.csv
        echo ""

        # Execute with column separation
        parallel --colsep ',' -j 1 mloop train {1} target --time {2} :::: /tmp/mloop-jobs.csv

        rm /tmp/mloop-jobs.csv
        ;;

    5)
        echo -e "${GREEN}Example 5: Progress Bar with ETA${NC}"
        echo -e "${YELLOW}Visual progress tracking${NC}"
        echo ""

        # Create sample jobs
        echo "train dataset1.csv target --time 60" > /tmp/mloop-jobs.txt
        echo "train dataset2.csv target --time 60" >> /tmp/mloop-jobs.txt
        echo "train dataset3.csv target --time 60" >> /tmp/mloop-jobs.txt
        echo "train dataset4.csv target --time 60" >> /tmp/mloop-jobs.txt
        echo "train dataset5.csv target --time 60" >> /tmp/mloop-jobs.txt

        echo "Jobs to execute:"
        cat /tmp/mloop-jobs.txt
        echo ""

        # Execute with progress bar
        parallel -j 1 --bar mloop {} < /tmp/mloop-jobs.txt

        rm /tmp/mloop-jobs.txt
        ;;

    6)
        echo -e "${GREEN}Example 6: Dry Run (Preview Only)${NC}"
        echo -e "${YELLOW}Show commands without executing${NC}"
        echo ""

        # Create sample jobs
        echo "train dataset1.csv target --time 1800" > /tmp/mloop-jobs.txt
        echo "train dataset2.csv target --time 1800" >> /tmp/mloop-jobs.txt
        echo "train dataset3.csv target --time 1800" >> /tmp/mloop-jobs.txt

        echo "Commands that would be executed:"
        echo ""

        # Dry run (--dry-run)
        parallel --dry-run -j 1 mloop {} < /tmp/mloop-jobs.txt

        echo ""
        echo -e "${YELLOW}(Nothing was actually executed)${NC}"

        rm /tmp/mloop-jobs.txt
        ;;

    *)
        echo -e "${RED}Error: Unknown example number: $example${NC}"
        echo ""
        echo "Available examples: 1-6"
        echo "Run '$0' without arguments to see details"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Example complete${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Additional GNU Parallel Tips:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "1. From multiple datasets:"
echo "   parallel -j 1 mloop train {} ::: dataset1.csv dataset2.csv dataset3.csv"
echo ""
echo "2. From wildcard pattern:"
echo "   parallel -j 1 mloop train {} ::: datasets/*.csv"
echo ""
echo "3. With different labels:"
echo "   parallel -j 1 mloop train data.csv {1} --time {2} ::: price revenue profit ::: 300 600 900"
echo ""
echo "4. Retry failed jobs:"
echo "   parallel --retry-failed --joblog joblog.txt"
echo ""
echo "5. Limit execution time:"
echo "   parallel --timeout 3600 -j 1 mloop train {} ::: dataset*.csv"
echo ""
echo "6. Email notification when done:"
echo "   parallel -j 1 mloop train {} ::: dataset*.csv && echo 'Done' | mail -s 'Training Complete' you@example.com"
echo ""
echo "For more info: man parallel"
echo ""
