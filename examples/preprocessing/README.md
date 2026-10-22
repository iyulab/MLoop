# FilePrepper 전처리 예제

MLoop 데이터셋에 대한 FilePrepper CLI 기반 전처리 워크플로우 예제입니다.

## 📁 구조

```
preprocessing/
├── README.md                      # 이 파일
├── dataset001_preprocessing.sh    # Dataset 001: DateTime 처리
├── dataset002_preprocessing.sh    # Dataset 002: DateTime 특성 추출
├── dataset005_preprocessing.sh    # Dataset 005: Expression 계산
├── dataset006_preprocessing.sh    # Dataset 006: Clean + Unpivot
└── benchmark_performance.sh       # 성능 벤치마크
```

## 🚀 사용 방법

### Dataset 001: 공급망 최적화
```bash
cd examples/preprocessing
./dataset001_preprocessing.sh
```

**작업**:
1. DateTime 파싱: `yyyyMMddHHmm` → `yyyy-MM-dd HH:mm:ss`
2. 특성 추출: Year, Month, DayOfWeek

**성능**: 17,364 rows in ~6 seconds

---

### Dataset 002: 사출성형 공급망최적화
```bash
./dataset002_preprocessing.sh
```

**작업**:
1. DateTime 특성 추출: Year, Month, DayOfWeek, Hour
2. 자동 포맷 감지: `yyyy-MM-dd H:mm`

**성능**: 34,617 rows instantly

---

### Dataset 005: 열처리 공급망최적화
```bash
./dataset005_preprocessing.sh
```

**작업**:
1. Expression 계산: `생산갭 = 생산필요량 - 재고`

**성능**: 688 rows in <5 seconds

---

### Dataset 006: 표면처리 공급망최적화
```bash
./dataset006_preprocessing.sh
```

**작업**:
1. CSV Cleaning: 천 단위 구분자 제거 (`1,000` → `1000`)
2. Unpivot: Wide → Long 변환 (177 rows → 655 rows)

**성능**: <10 seconds total

---

## 📊 성능 벤치마크

전체 데이터셋에 대한 성능 측정:

```bash
./benchmark_performance.sh
```

**결과 예시**:
```
Dataset 001: 6.12 seconds (17,364 rows)
Dataset 002: 4.83 seconds (34,617 rows)
Dataset 005: 2.45 seconds (688 rows)
Dataset 006: 8.91 seconds (177→655 rows)
```

---

## 🎯 주요 장점

### 코드 제거
- **이전**: 각 데이터셋마다 커스텀 C# 코드 작성
- **이후**: 재사용 가능한 CLI 명령어

### 성능 향상
- DateTime 처리: **20배 빠름** (60초 → 3초)
- 전체 워크플로우: **10초 이내**

### 재현 가능성
- 셸 스크립트로 명확한 워크플로우
- 버전 관리 가능
- CI/CD 통합 쉬움

### 유지보수
- 선언적 명령어
- 명확한 오류 메시지
- 디버깅 용이

---

## 📚 FilePrepper CLI 명령어

### datetime
```bash
fileprepper datetime -i INPUT -o OUTPUT -c COLUMN -m MODE [options]
```
**Modes**:
- `parse`: DateTime 포맷 변환
- `parseexcel`: Excel 날짜 변환
- `features`: DateTime 특성 추출

### expression
```bash
fileprepper expression -i INPUT -o OUTPUT -e "column=expr" [options]
```
**예**: `-e "total=price*quantity"`

### clean
```bash
fileprepper clean -i INPUT -o OUTPUT [-c COLUMNS...] [options]
```
**기능**: 천 단위 구분자 제거, 숫자 정규화

### unpivot
```bash
fileprepper unpivot -i INPUT -o OUTPUT -b BASE... -g GROUPS... [options]
```
**기능**: Wide → Long 변환

### merge-asof
```bash
fileprepper merge-asof -i LEFT RIGHT -o OUTPUT --left-on COL --right-on COL [options]
```
**기능**: 시계열 nearest join

### conditional
```bash
fileprepper conditional -i INPUT -o OUTPUT --output COL --condition "expr" [options]
```
**기능**: if-then-else 로직

### string
```bash
fileprepper string -i INPUT -o OUTPUT -c COLUMN -m MODE [options]
```
**Modes**: substring, concat, replace, trim, upper, lower

---

## 🔧 문제 해결

### 일반적인 문제

#### 1. 파일을 찾을 수 없음
```bash
# 상대 경로 확인
ls ../ML-Resource/001-공급망 최적화/Dataset/data.csv
```

#### 2. 권한 오류
```bash
# 스크립트 실행 권한 부여
chmod +x dataset001_preprocessing.sh
```

#### 3. FilePrepper 찾을 수 없음
```bash
# FilePrepper CLI 빌드
cd ../../FilePrepper/src/FilePrepper.CLI
dotnet build -c Release

# PATH에 추가 또는 절대 경로 사용
/full/path/to/fileprepper datetime ...
```

---

## 📖 추가 문서

- [FilePrepper](https://github.com/iyulab/FilePrepper)
- [MLoop 아키텍처](../../docs/ARCHITECTURE.md)

---

**작성일**: 2025-11-09
**버전**: 1.0
