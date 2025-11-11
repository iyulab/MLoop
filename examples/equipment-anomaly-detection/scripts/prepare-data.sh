#!/bin/bash

# Equipment Anomaly Detection - Data Preparation Script
# This script demonstrates FilePrepper + MLoop integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Configuration
SOURCE_PATH="${1:-../../ML-Resource/014-장비이상 조기탐지/Dataset/data/5공정_180sec}"
OUTPUT_PATH="${2:-./datasets/train.csv}"

echo -e "${CYAN}=== Equipment Anomaly Detection - Data Preparation ===${NC}"
echo ""

# Step 1: Check source data exists
echo -e "${YELLOW}[1/5] Checking source data...${NC}"
if [ ! -d "$SOURCE_PATH" ]; then
    echo -e "${RED}ERROR: Source path not found: $SOURCE_PATH${NC}"
    echo -e "${RED}Please provide the correct path as the first argument${NC}"
    echo -e "${GRAY}Usage: $0 <source-path> [output-path]${NC}"
    exit 1
fi

CSV_COUNT=$(find "$SOURCE_PATH" -name "kemp-abh-sensor-*.csv" | wc -l)
echo -e "${GREEN}  Found $CSV_COUNT sensor data files${NC}"
echo -e "${GREEN}  Error Lot List exists: $([ -f "$SOURCE_PATH/Error Lot list.csv" ] && echo "Yes" || echo "No")${NC}"
echo ""

# Step 2: Copy raw data to project
echo -e "${YELLOW}[2/5] Copying raw data to project...${NC}"
mkdir -p "./raw-data"

# Copy first 10 sensor files
find "$SOURCE_PATH" -name "kemp-abh-sensor-*.csv" | head -10 | while read file; do
    cp "$file" "./raw-data/"
done

# Copy error lot list
cp "$SOURCE_PATH/Error Lot list.csv" "./raw-data/" 2>/dev/null || true

echo -e "${GREEN}  Copied 10 sample files for demonstration${NC}"
echo ""

# Step 3: FilePrepper demonstration
echo -e "${YELLOW}[3/5] Using FilePrepper to merge sensor data...${NC}"
echo -e "${GRAY}  (This would use FilePrepper CLI in production)${NC}"
echo -e "${GRAY}  Command: fileprepper merge --input raw-data/*.csv --output datasets/merged-sensors.csv${NC}"
echo ""
echo -e "${CYAN}  FilePrepper features demonstrated:${NC}"
echo -e "${GRAY}    - Multi-file CSV merging${NC}"
echo -e "${GRAY}    - Column type inference${NC}"
echo -e "${GRAY}    - Data validation${NC}"
echo -e "${GRAY}    - Missing value detection${NC}"
echo ""

# Step 4: Create labeled dataset
echo -e "${YELLOW}[4/5] Creating labeled dataset with Error Lot List...${NC}"
echo -e "${CYAN}  Logic:${NC}"
echo -e "${GRAY}    1. Load Error Lot List (Date -> Error Process IDs)${NC}"
echo -e "${GRAY}    2. Join sensor data with error list by Date and Process${NC}"
echo -e "${GRAY}    3. Create 'IsError' label column (1 if in error list, 0 otherwise)${NC}"
echo -e "${GRAY}    4. Add time-based features (hour, minute, day_of_week)${NC}"
echo ""

# Step 5: Generate sample training data
echo -e "${YELLOW}[5/5] Generating sample training dataset...${NC}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

# Create sample CSV with simulated labels
cat > "$OUTPUT_PATH" << 'EOF'
Process,Temp,Current,Date,Hour,Minute,IsError
1,75.14,1.61,2021-09-06,16,24,0
1,76.66,1.53,2021-09-06,16,24,0
1,77.18,1.70,2021-09-06,16,24,0
20,75.50,1.65,2021-09-06,16,25,1
21,76.20,1.58,2021-09-06,16,25,1
32,78.90,1.72,2021-09-06,16,26,1
33,77.45,1.68,2021-09-06,16,26,1
5,76.12,1.62,2021-09-07,10,15,0
6,75.89,1.59,2021-09-07,10,15,0
32,79.20,1.75,2021-09-07,10,16,1
33,78.15,1.70,2021-09-07,10,16,1
34,77.98,1.69,2021-09-07,10,16,1
EOF

ROWS=$(wc -l < "$OUTPUT_PATH")
echo -e "${GREEN}  Sample dataset created: $OUTPUT_PATH${NC}"
echo -e "${GREEN}  Rows: $ROWS${NC}"
echo ""

# Summary
echo -e "${CYAN}=== Data Preparation Complete ===${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${NC}  1. Review the prepared dataset: $OUTPUT_PATH${NC}"
echo -e "${NC}  2. Run MLoop agent for data analysis:${NC}"
echo -e "${GRAY}     mloop agent 'Analyze datasets/train.csv' --agent data-analyst${NC}"
echo ""
echo -e "${NC}  3. Generate preprocessing script:${NC}"
echo -e "${GRAY}     mloop agent 'Generate preprocessing for this dataset' --agent preprocessing-expert${NC}"
echo ""
echo -e "${NC}  4. Train model:${NC}"
echo -e "${GRAY}     mloop train --time 300 --metric F1Score${NC}"
echo ""

echo -e "${CYAN}For production use:${NC}"
echo -e "${NC}  - Use FilePrepper CLI for robust data merging${NC}"
echo -e "${NC}  - Implement proper Error Lot List joining logic${NC}"
echo -e "${NC}  - Add data validation and quality checks${NC}"
echo -e "${NC}  - Create automated pipeline with MLoop.AIAgent${NC}"
echo ""
