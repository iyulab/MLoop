#!/bin/bash
# MLoop Sequential Job Queue
# Executes multiple MLoop commands one after another
#
# Usage:
#   ./mloop-queue.sh "train data1.csv price --time 1800" "train data2.csv price --time 1800"
#
# Example:
#   ./mloop-queue.sh \
#     "train datasets/sales-2022.csv revenue --time 1800" \
#     "train datasets/sales-2023.csv revenue --time 1800" \
#     "evaluate models/staging/exp-001/model.zip datasets/test.csv"

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if any arguments provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No jobs specified${NC}"
    echo ""
    echo "Usage: $0 \"<mloop-command-1>\" \"<mloop-command-2>\" ..."
    echo ""
    echo "Example:"
    echo "  $0 \\"
    echo "    \"train dataset1.csv price --time 1800\" \\"
    echo "    \"train dataset2.csv price --time 1800\" \\"
    echo "    \"evaluate models/staging/exp-001/model.zip test.csv\""
    exit 1
fi

# Check if mloop is available
if ! command -v mloop &> /dev/null; then
    echo -e "${RED}Error: mloop command not found${NC}"
    echo "Please install MLoop: dotnet tool install -g mloop"
    exit 1
fi

# Initialize counters
total_jobs=$#
completed_jobs=0
failed_jobs=0
start_time=$(date +%s)

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MLoop Sequential Job Queue           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Total jobs: ${total_jobs}${NC}"
echo ""

# Execute each job
job_num=1
for job in "$@"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[${job_num}/${total_jobs}] Starting job${NC}"
    echo -e "${YELLOW}Command: mloop ${job}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    job_start=$(date +%s)

    # Execute the command
    if mloop $job; then
        job_end=$(date +%s)
        job_duration=$((job_end - job_start))
        job_duration_min=$((job_duration / 60))
        job_duration_sec=$((job_duration % 60))

        echo ""
        echo -e "${GREEN}✅ Job ${job_num} complete${NC} (${job_duration_min}m ${job_duration_sec}s)"
        completed_jobs=$((completed_jobs + 1))
    else
        job_end=$(date +%s)
        job_duration=$((job_end - job_start))
        job_duration_min=$((job_duration / 60))
        job_duration_sec=$((job_duration % 60))

        echo ""
        echo -e "${RED}❌ Job ${job_num} failed${NC} (${job_duration_min}m ${job_duration_sec}s)"
        echo -e "${RED}Command: mloop ${job}${NC}"
        failed_jobs=$((failed_jobs + 1))

        # Ask user whether to continue
        echo ""
        read -p "Continue with remaining jobs? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Stopping queue execution${NC}"
            break
        fi
    fi

    echo ""
    job_num=$((job_num + 1))
done

# Calculate total duration
end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

# Print summary
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Execution Summary                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total jobs:      ${total_jobs}"
echo -e "${GREEN}Completed:       ${completed_jobs}${NC}"

if [ $failed_jobs -gt 0 ]; then
    echo -e "${RED}Failed:          ${failed_jobs}${NC}"
fi

skipped_jobs=$((total_jobs - completed_jobs - failed_jobs))
if [ $skipped_jobs -gt 0 ]; then
    echo -e "${YELLOW}Skipped:         ${skipped_jobs}${NC}"
fi

echo ""
echo -e "Total time:      ${total_hours}h ${total_minutes}m ${total_seconds}s"
echo ""

# Exit with appropriate code
if [ $failed_jobs -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some jobs failed${NC}"
    exit 1
elif [ $completed_jobs -eq $total_jobs ]; then
    echo -e "${GREEN}🎉 All jobs completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Execution interrupted${NC}"
    exit 2
fi
