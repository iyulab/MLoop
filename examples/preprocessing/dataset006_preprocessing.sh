#!/bin/bash
# Dataset 006: 표면처리 공급망최적화 전처리 워크플로우
# FilePrepper CLI 사용

set -e

DATASET_DIR="../ML-Resource/006-표면처리 공급망최적화/Dataset/data"
OUTPUT_DIR="../ML-Resource/006-표면처리 공급망최적화/Dataset/preprocessed"

mkdir -p "${OUTPUT_DIR}"

echo "=================================="
echo "Dataset 006 전처리 시작"
echo "=================================="

# Step 1: CSV Cleaning (천 단위 구분자 제거)
echo ""
echo "[1/2] CSV 천 단위 구분자 제거 중..."
fileprepper clean \
  -i "${DATASET_DIR}/일무사_표면처리.csv" \
  -o "${OUTPUT_DIR}/cleaned-fileprepper.csv" \
  --header

# Step 2: Unpivot (Wide → Long 변환)
echo ""
echo "[2/2] Unpivot (Wide → Long) 변환 중..."
fileprepper unpivot \
  -i "${OUTPUT_DIR}/cleaned-fileprepper.csv" \
  -o "${OUTPUT_DIR}/unpivoted-fileprepper.csv" \
  --header \
  -b "생산일자" "작업지시번호" "제품코드" "시작" "종료" "생산량(Kg)" \
  -g "1차 출고날짜" "1차 출고량" \
     "2차 출고날짜" "2차 출고량" \
     "3차 출고날짜" "3차 출고량" \
     "4차 출고날짜" "4차 출고량" \
     "5차 출고날짜" "5차 출고량" \
     "6차 출고날짜" "6차 출고량" \
     "7차 출고날짜" "7차 출고량" \
     "8차 출고날짜" "8차 출고량" \
     "9차 출고날짜" "9차 출고량" \
     "10차 출고날짜" "10차 출고량" \
  -idx "출고차수" -vc "출고날짜" "출고량" \
  --skip-empty

echo ""
echo "=================================="
echo "✓ 전처리 완료!"
echo "  출력: ${OUTPUT_DIR}/unpivoted-fileprepper.csv"
echo "  177 wide rows → 655 long rows"
echo "=================================="
