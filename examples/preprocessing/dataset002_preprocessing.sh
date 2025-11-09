#!/bin/bash
# Dataset 002: 사출성형 공급망최적화 전처리 워크플로우
# FilePrepper CLI 사용

set -e

DATASET_DIR="../ML-Resource/002-사출성형 공급망최적화/Dataset/data"
OUTPUT_DIR="../ML-Resource/002-사출성형 공급망최적화/mloop-project/datasets"

echo "=================================="
echo "Dataset 002 전처리 시작"
echo "=================================="

# DateTime 특성 추출 (yyyy-MM-dd H:mm 자동 감지)
echo ""
echo "[1/1] DateTime 특성 추출 중..."
fileprepper datetime \
  -i "${DATASET_DIR}/사출성형.csv" \
  -o "${OUTPUT_DIR}/train_with_datetime.csv" \
  -c "DateTime" \
  -m features \
  -ft "Year,Month,DayOfWeek,Hour" \
  --header

echo ""
echo "=================================="
echo "✓ 전처리 완료!"
echo "  출력: ${OUTPUT_DIR}/train_with_datetime.csv"
echo "  Features: Year, Month, DayOfWeek, Hour"
echo "=================================="
