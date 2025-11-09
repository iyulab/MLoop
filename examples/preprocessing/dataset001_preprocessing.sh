#!/bin/bash
# Dataset 001: 공급망 최적화 전처리 워크플로우
# FilePrepper CLI 사용

set -e

DATASET_DIR="../ML-Resource/001-공급망 최적화/Dataset"
OUTPUT_DIR="../ML-Resource/001-공급망 최적화/mloop-project/datasets"

echo "=================================="
echo "Dataset 001 전처리 시작"
echo "=================================="

# Step 1: DateTime 파싱 (yyyyMMddHHmm → yyyy-MM-dd HH:mm:ss)
echo ""
echo "[1/2] DateTime 파싱 중..."
fileprepper datetime \
  -i "${DATASET_DIR}/data.csv" \
  -o "${OUTPUT_DIR}/step1_parsed.csv" \
  -c "CRET_TIME" \
  -m parse \
  -f "yyyyMMddHHmm" \
  -of "yyyy-MM-dd HH:mm:ss" \
  --header

# Step 2: DateTime 특성 추출 (Year, Month, DayOfWeek)
echo ""
echo "[2/2] DateTime 특성 추출 중..."
fileprepper datetime \
  -i "${OUTPUT_DIR}/step1_parsed.csv" \
  -o "${OUTPUT_DIR}/train_with_datetime.csv" \
  -c "CRET_TIME" \
  -m features \
  -ft "Year,Month,DayOfWeek" \
  --header

# 정리
rm "${OUTPUT_DIR}/step1_parsed.csv"

echo ""
echo "=================================="
echo "✓ 전처리 완료!"
echo "  출력: ${OUTPUT_DIR}/train_with_datetime.csv"
echo "=================================="
