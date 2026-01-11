# MLoop 시뮬레이션 가이드 (v0.5.0)

> **목적**: LLM 에이전트가 MLoop CLI를 올바르게 사용하여 자율 ML 모델 빌드를 수행하도록 안내

## 핵심 CLI 옵션

### 데이터 경로 옵션

```bash
# 기본: datasets/train.csv 자동 탐색
mloop train --label Target --task regression

# 외부 단일 파일
mloop train --data /path/to/data.csv --label Target --task regression

# 외부 다중 파일 (자동 병합)
mloop train --data normal.csv outlier.csv --label Target --task regression

# 자동 병합 (datasets/ 내 동일 스키마 파일 탐색)
mloop train --auto-merge --label Target --task regression
```

### Task 유형

| Task | 옵션 값 | 사용 조건 |
|------|---------|-----------|
| Regression | `--task regression` | 숫자 예측 |
| Binary Classification | `--task binary-classification` | 2 클래스 분류 |
| Multiclass Classification | `--task multiclass-classification` | 3+ 클래스 분류 |

### 자동 처리 기능

| 기능 | 기본값 | 비활성화 |
|------|--------|----------|
| Label 결측치 제거 | ✅ 활성 (Classification) | `--drop-missing-labels false` |
| 인코딩 자동 변환 | ✅ 활성 (CP949→UTF-8) | 비활성화 불가 |
| 저성능 진단 | ✅ 활성 | 비활성화 불가 |
| 클래스 분포 분석 | ✅ 활성 (Classification) | 비활성화 불가 |

## 장애물별 대응 전략

### 1. Multi-CSV 데이터셋

**증상**: 여러 CSV 파일이 datasets/ 폴더에 존재

**해결책**:
```bash
# 방법 1: 명시적 파일 지정
mloop train --data file1.csv file2.csv file3.csv --label Target --task regression

# 방법 2: 자동 병합 (동일 스키마 파일 자동 탐색)
mloop train --auto-merge --label Target --task regression
```

**예시**: 006 표면처리 (4 CSV), 007 소성가공 (7 CSV), 010 공정운영 (34 CSV)

### 2. Label 결측치

**증상**: Label 컬럼에 빈 값 존재

**해결책**: 자동 처리됨 (v0.4.0+)
```bash
# 기본 동작: Classification 시 자동 제거
mloop train --label Status --task binary-classification
# 출력: "Warning: Dropped 113 rows with missing labels"
```

**예시**: 015 설비고장예측 (2.2% 결측 → 자동 제거)

### 3. 인코딩 문제 (한글 깨짐)

**증상**: CP949/EUC-KR 인코딩으로 인한 한글 깨짐

**해결책**: 자동 처리됨 (v0.5.0+)
```bash
# 자동 감지 및 UTF-8 변환
mloop train --data korean_data.csv --label Target --task regression
# 출력: "[Info] Converted CP949 → UTF-8: korean_data.csv"
```

**예시**: 018 열처리 예지보전 (CP949 인코딩)

### 4. 저성능 결과

**증상**: R² < 0.5 또는 Accuracy < 0.7

**해결책**: 자동 진단 (v0.3.0+)
```
출력 예시:
⚠ Model performance is low (R²: 0.04)
Suggestions:
  • Consider feature engineering
  • Check for data quality issues
  • Increase training time
```

### 5. 클래스 불균형

**증상**: 클래스 비율 불균형 (예: 97% vs 3%)

**해결책**: 자동 시각화 및 경고 (v0.3.0+)
```
출력 예시:
Class Distribution:
████████████████████████████████████████ 97%
██                                      3%
⚠ Imbalanced dataset (ratio: 32:1)
```

## 미해결 장애물

### 피처-라벨 분리 구조
- **해당 데이터셋**: 014, 016
- **증상**: 피처 파일과 라벨 파일이 분리됨
- **현재 해결책**: 수동 조인 필요
- **향후 계획**: 자동 조인 기능 개발

### 라벨 컬럼 부재
- **해당 데이터셋**: 020, 022
- **증상**: 라벨로 사용할 컬럼이 명확하지 않음
- **현재 해결책**: 도메인 분석 후 수동 지정 필요
- **향후 계획**: 라벨 추론 기능 개발

### Wide-format 변환
- **해당 데이터셋**: 021
- **증상**: 피벗된 Wide 형태 데이터
- **현재 해결책**: FilePrepper 전처리 스크립트 사용
- **향후 계획**: 자동 감지 및 변환

## 자율성 레벨 가이드

| Level | 설명 | 목표 |
|-------|------|------|
| L3 | 결과 확인만 | 단일 깨끗한 CSV |
| L2 | 계획 승인 | Multi-CSV, 간단한 전처리 |
| L1 | 단계별 승인 | 복잡한 전처리, 라벨 추론 |
| L0 | 실패 | 지원 불가 (이미지, 비정형) |

## 체크리스트

시뮬레이션 전 확인사항:

- [ ] 데이터셋 위치 확인 (datasets/ 또는 외부 경로)
- [ ] CSV 파일 개수 확인 (Multi-CSV → `--data` 또는 `--auto-merge`)
- [ ] 라벨 컬럼 존재 여부 확인
- [ ] 인코딩 확인 (한글 → 자동 처리됨)
- [ ] Task 유형 결정 (regression / binary-classification / multiclass-classification)

---

**버전**: v0.5.0
**최종 업데이트**: 2026-01-11
