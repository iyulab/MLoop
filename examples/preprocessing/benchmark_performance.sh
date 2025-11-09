#!/bin/bash
# FilePrepper 성능 벤치마크
# 모든 데이터셋 전처리 시간 측정

set -e

echo "========================================"
echo "FilePrepper 성능 벤치마크"
echo "========================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 결과 저장
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "시작 시각: $(date)" | tee "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"

# Dataset 001
echo -e "${YELLOW}[1/4] Dataset 001 벤치마킹...${NC}"
START=$(date +%s.%N)
./dataset001_preprocessing.sh > /dev/null 2>&1
END=$(date +%s.%N)
TIME001=$(echo "$END - $START" | bc)
echo -e "${GREEN}✓ Dataset 001: ${TIME001} seconds (17,364 rows)${NC}" | tee -a "${RESULTS_FILE}"

# Dataset 002
echo -e "${YELLOW}[2/4] Dataset 002 벤치마킹...${NC}"
START=$(date +%s.%N)
./dataset002_preprocessing.sh > /dev/null 2>&1
END=$(date +%s.%N)
TIME002=$(echo "$END - $START" | bc)
echo -e "${GREEN}✓ Dataset 002: ${TIME002} seconds (34,617 rows)${NC}" | tee -a "${RESULTS_FILE}"

# Dataset 005
echo -e "${YELLOW}[3/4] Dataset 005 벤치마킹...${NC}"
START=$(date +%s.%N)
./dataset005_preprocessing.sh > /dev/null 2>&1
END=$(date +%s.%N)
TIME005=$(echo "$END - $START" | bc)
echo -e "${GREEN}✓ Dataset 005: ${TIME005} seconds (688 rows)${NC}" | tee -a "${RESULTS_FILE}"

# Dataset 006
echo -e "${YELLOW}[4/4] Dataset 006 벤치마킹...${NC}"
START=$(date +%s.%N)
./dataset006_preprocessing.sh > /dev/null 2>&1
END=$(date +%s.%N)
TIME006=$(echo "$END - $START" | bc)
echo -e "${GREEN}✓ Dataset 006: ${TIME006} seconds (177→655 rows)${NC}" | tee -a "${RESULTS_FILE}"

# 총계
TOTAL=$(echo "$TIME001 + $TIME002 + $TIME005 + $TIME006" | bc)

echo "" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo -e "${GREEN}총 처리 시간: ${TOTAL} seconds${NC}" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "결과 저장: ${RESULTS_FILE}"
