#!/bin/bash
# Dataset 005: 열처리 공급망최적화 전처리 워크플로우
# FilePrepper CLI 사용

set -e

DATASET_DIR="../ML-Resource/005-열처리 공급망최적화/Dataset/data"
OUTPUT_DIR="../ML-Resource/005-열처리 공급망최적화/Dataset/preprocessed"

mkdir -p "${OUTPUT_DIR}"

echo "=================================="
echo "Dataset 005 전처리 시작"
echo "=================================="

# Expression: 생산갭 = 생산필요량 - 재고
echo ""
echo "[1/1] 생산갭 계산 중..."
fileprepper expression \
  -i "${DATASET_DIR}/data.csv" \
  -o "${OUTPUT_DIR}/features-fileprepper.csv" \
  -e "생산갭=생산필요량-재고" \
  --header

echo ""
echo "=================================="
echo "✓ 전처리 완료!"
echo "  출력: ${OUTPUT_DIR}/features-fileprepper.csv"
echo "  새 컬럼: 생산갭 = 생산필요량 - 재고"
echo "=================================="
