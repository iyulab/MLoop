# MLoop 개발 태스크 목록

**Last Updated**: 2024-11-03
**Version**: 0.1.0-alpha
**Status**: 진행 중

---

## 목차

1. [프로젝트 현황](#프로젝트-현황)
2. [Phase 0: 기반 구축](#phase-0-기반-구축-완료)
3. [Phase 1: MVP (핵심 기능)](#phase-1-mvp-핵심-기능)
4. [Phase 2: 실험 관리 및 서빙](#phase-2-실험-관리-및-서빙)
5. [Phase 3: 고급 기능](#phase-3-고급-기능)
6. [Phase 4: 프로덕션 준비](#phase-4-프로덕션-준비)
7. [우선순위 가이드](#우선순위-가이드)

---

## 프로젝트 현황

### 완료된 작업 ✅
- [x] .NET 9 솔루션 구조 생성
- [x] Global tool 설정 (PackAsTool, ToolCommandName)
- [x] 핵심 의존성 추가 (ML.NET 4.0, AutoML 0.21.1, System.CommandLine, etc.)
- [x] CLI 엔트리 포인트 및 배너 구현
- [x] 프로젝트 템플릿 파일 (binary-classification.yaml, etc.)
- [x] 아키텍처 문서 작성 (Multi-Process Casual Design)
- [x] 빌드 설정 파일 (Directory.Build.props, .editorconfig, nuget.config)

### 현재 상태
- **빌드**: ✅ 성공 (mloop 실행 가능, 배너 표시됨)
- **커맨드**: ⚠️ 모든 커맨드 주석처리됨 (InitCommand, TrainCommand, etc.)
- **다음 단계**: Phase 1 커맨드 구현 시작

---

## Phase 0: 기반 구축 (완료)

### 목표
.NET 9 프로젝트 구조 및 개발 환경 설정

### 완료된 태스크

#### ✅ 프로젝트 구조
- [x] src/MLoop.sln 생성
- [x] src/MLoop/MLoop.csproj 생성 및 설정
- [x] tests/MLoop.Tests/MLoop.Tests.csproj 생성
- [x] Directory.Build.props 작성
- [x] .editorconfig 작성
- [x] nuget.config 작성
- [x] .gitignore 작성

#### ✅ CLI 기본 구조
- [x] Program.cs 엔트리 포인트
- [x] Spectre.Console 배너 구현
- [x] System.CommandLine 통합

#### ✅ 문서화
- [x] docs/ARCHITECTURE.md 작성
- [x] README.md 작성
- [x] Templates 파일 생성

---

## Phase 1: MVP (핵심 기능)

**목표**: 3단계 워크플로우 구현 (init → train → predict)
**예상 기간**: 3-4주
**우선순위**: 🔴 최고

### 1.1 Infrastructure Layer (기반)

**목표**: 파일시스템 관리 및 프로젝트 디스커버리

#### 태스크 목록

##### 1.1.1 FileSystem 기본 구조
**우선순위**: 🔴 P0 (다른 모든 작업의 의존성)

```
src/MLoop/Infrastructure/FileSystem/
├── IFileSystemManager.cs
├── FileSystemManager.cs
├── IProjectDiscovery.cs
├── ProjectDiscovery.cs
├── IExperimentStore.cs
├── ExperimentStore.cs
├── IModelRegistry.cs
└── ModelRegistry.cs
```

**세부 태스크**:
- [ ] **T1.1.1.1**: `IFileSystemManager` 인터페이스 정의
  - 메서드: `CreateDirectory`, `WriteJson`, `ReadJson`, `FileExists`, `DirectoryExists`
  - 예상 시간: 1시간

- [ ] **T1.1.1.2**: `FileSystemManager` 구현
  - JSON 직렬화/역직렬화 (System.Text.Json)
  - 디렉토리 생성 (recursive)
  - 파일 읽기/쓰기 (thread-safe)
  - 예상 시간: 3시간

- [ ] **T1.1.1.3**: `IProjectDiscovery` 인터페이스 정의
  - 메서드: `FindRoot()`, `IsProjectRoot()`, `EnsureProjectRoot()`
  - 예상 시간: 1시간

- [ ] **T1.1.1.4**: `ProjectDiscovery` 구현
  - `.mloop/` 디렉토리 찾기 (상위로 순회)
  - 프로젝트 루트 검증
  - 예상 시간: 2시간

**완료 조건**:
- [x] 모든 인터페이스 정의됨
- [x] 구현 클래스 작성 완료
- [x] 단위 테스트 작성 (coverage > 80%)
- [x] 멀티프로세스 안전성 확인

##### 1.1.2 Experiment Store
**우선순위**: 🔴 P0 (Training 의존)

**세부 태스크**:
- [ ] **T1.1.2.1**: `IExperimentStore` 인터페이스 정의
  - 메서드: `GenerateIdAsync()`, `SaveAsync()`, `LoadAsync()`, `ListAsync()`
  - 예상 시간: 1시간

- [ ] **T1.1.2.2**: `ExperimentStore` 구현
  - Experiment ID 생성 (atomic, exp-XXX 형식)
  - 실험 메타데이터 저장 (metadata.json, metrics.json, config.json)
  - 파일 잠금 처리 (concurrent ID generation)
  - 예상 시간: 4시간

- [ ] **T1.1.2.3**: `.mloop/experiment-index.json` 관리
  - next_id 자동 증가 (retry logic)
  - experiments 배열 업데이트
  - 예상 시간: 2시간

**완료 조건**:
- [x] Experiment ID 중복 없이 생성됨
- [x] 동시 실행 시 충돌 없음 (integration test)
- [x] 메타데이터 정확히 저장됨

##### 1.1.3 Model Registry
**우선순위**: 🟡 P1 (Phase 1 후반)

**세부 태스크**:
- [ ] **T1.1.3.1**: `IModelRegistry` 인터페이스 정의
  - 메서드: `PromoteAsync()`, `GetAsync()`, `ListAsync()`
  - 예상 시간: 1시간

- [ ] **T1.1.3.2**: `ModelRegistry` 구현
  - 모델 승격 (staging, production)
  - `.mloop/registry.json` 관리
  - 예상 시간: 3시간

**완료 조건**:
- [x] staging/production 승격 동작
- [x] registry.json 정확히 업데이트됨

##### 1.1.4 Configuration 관리
**우선순위**: 🔴 P0 (모든 커맨드 의존)

```
src/MLoop/Infrastructure/Configuration/
├── MLoopConfig.cs
├── ConfigLoader.cs
└── ConfigMerger.cs
```

**세부 태스크**:
- [ ] **T1.1.4.1**: `MLoopConfig` 데이터 모델 정의
  - ProjectName, Task, LabelColumn 등 속성
  - 예상 시간: 1시간

- [ ] **T1.1.4.2**: `ConfigLoader` 구현
  - .mloop/config.json 로드
  - mloop.yaml 로드 (YamlDotNet)
  - 예상 시간: 2시간

- [ ] **T1.1.4.3**: `ConfigMerger` 구현
  - CLI args > mloop.yaml > .mloop/config.json > defaults
  - 우선순위 기반 병합
  - 예상 시간: 3시간

**완료 조건**:
- [x] 4단계 설정 병합 동작
- [x] 우선순위 정확히 적용됨

---

### 1.2 Core Layer (비즈니스 로직)

**목표**: ML.NET 및 AutoML 통합

#### 태스크 목록

##### 1.2.1 Data Loaders
**우선순위**: 🔴 P0 (Training 의존)

```
src/MLoop/Core/Data/
├── IDataLoader.cs
├── CsvDataLoader.cs
└── DataSchema.cs
```

**세부 태스크**:
- [ ] **T1.2.1.1**: `IDataLoader` 인터페이스 정의
  - 메서드: `LoadAsync()`, `ValidateSchema()`
  - 예상 시간: 1시간

- [ ] **T1.2.1.2**: `CsvDataLoader` 구현
  - ML.NET TextLoader 사용
  - 스키마 자동 추론
  - Label 컬럼 검증
  - 예상 시간: 4시간

- [ ] **T1.2.1.3**: `DataSchema` 구현
  - 컬럼 타입 정의 (string, float, boolean 등)
  - 스키마 직렬화
  - 예상 시간: 2시간

**완료 조건**:
- [x] CSV 파일 정확히 로드됨
- [x] Label 컬럼 검증 동작
- [x] 에러 처리 완료

##### 1.2.2 AutoML Engine
**우선순위**: 🔴 P0 (핵심 기능)

```
src/MLoop/Core/AutoML/
├── ITrainingEngine.cs
├── TrainingEngine.cs
├── AutoMLRunner.cs
├── TrainingConfig.cs
└── TrainingResult.cs
```

**세부 태스크**:
- [ ] **T1.2.2.1**: `TrainingConfig` 데이터 모델
  - DataFile, LabelColumn, TimeLimitSeconds, Metric, TestSplit
  - 예상 시간: 1시간

- [ ] **T1.2.2.2**: `TrainingResult` 데이터 모델
  - ExperimentId, BestTrainer, Metrics, TrainingTime
  - 예상 시간: 1시간

- [ ] **T1.2.2.3**: `ITrainingEngine` 인터페이스 정의
  - 메서드: `TrainAsync(config, progress, cancellation)`
  - 예상 시간: 1시간

- [ ] **T1.2.2.4**: `AutoMLRunner` 구현
  - ML.NET AutoML Experiment 설정
  - Binary/Multiclass/Regression 지원
  - Progress 이벤트 처리
  - 예상 시간: 6시간

- [ ] **T1.2.2.5**: `TrainingEngine` 구현
  - ExperimentStore와 통합
  - AutoMLRunner 호출
  - 결과 저장 (model.zip, metadata.json, metrics.json)
  - 예상 시간: 4시간

**완료 조건**:
- [x] Binary classification 학습 동작
- [x] AutoML 정확히 실행됨
- [x] 결과 파일 올바르게 저장됨
- [x] Progress 리포팅 동작

##### 1.2.3 Prediction Engine
**우선순위**: 🔴 P0 (MVP 필수)

```
src/MLoop/Core/Models/
├── IPredictionEngine.cs
├── PredictionEngine.cs
├── ModelLoader.cs
└── PredictionResult.cs
```

**세부 태스크**:
- [ ] **T1.2.3.1**: `IPredictionEngine` 인터페이스 정의
  - 메서드: `PredictAsync(modelPath, dataSource, cancellation)`
  - 예상 시간: 1시간

- [ ] **T1.2.3.2**: `ModelLoader` 구현
  - model.zip 로드
  - ML.NET ITransformer 생성
  - 예상 시간: 2시간

- [ ] **T1.2.3.3**: `PredictionEngine` 구현
  - 배치 예측 (CSV → 결과)
  - 단일 예측 지원
  - 결과 저장 (outputs/predictions/)
  - 예상 시간: 4시간

**완료 조건**:
- [x] 모델 로드 동작
- [x] 배치 예측 정확함
- [x] 결과 파일 저장됨

##### 1.2.4 Evaluation Engine
**우선순위**: 🟡 P1 (Phase 1 후반)

```
src/MLoop/Core/Evaluation/
├── IEvaluator.cs
├── ClassificationEvaluator.cs
└── RegressionEvaluator.cs
```

**세부 태스크**:
- [ ] **T1.2.4.1**: `IEvaluator` 인터페이스 정의
  - 메서드: `EvaluateAsync(modelPath, testData, cancellation)`
  - 예상 시간: 1시간

- [ ] **T1.2.4.2**: `ClassificationEvaluator` 구현
  - ML.NET Evaluate() 호출
  - Accuracy, F1, AUC, Precision, Recall 계산
  - 예상 시간: 3시간

- [ ] **T1.2.4.3**: `RegressionEvaluator` 구현
  - RMSE, MAE, R² 계산
  - 예상 시간: 2시간

**완료 조건**:
- [x] 메트릭 정확히 계산됨
- [x] Binary/Multiclass/Regression 지원
- [x] 결과 저장됨

---

### 1.3 CLI Layer (사용자 인터페이스)

**목표**: 사용자 친화적 CLI 커맨드 구현

#### 태스크 목록

##### 1.3.1 InitCommand (프로젝트 초기화)
**우선순위**: 🔴 P0 (첫 번째 구현)

```
src/MLoop/Commands/InitCommand.cs
```

**세부 태스크**:
- [ ] **T1.3.1.1**: Command 정의
  - Arguments: project-name
  - Options: --task (binary-classification, multiclass, regression)
  - 예상 시간: 2시간

- [ ] **T1.3.1.2**: 디렉토리 구조 생성
  - .mloop/, data/, experiments/, models/ 생성
  - 예상 시간: 2시간

- [ ] **T1.3.1.3**: 파일 생성
  - .mloop/config.json
  - mloop.yaml (템플릿 기반)
  - .gitignore
  - README.md
  - 예상 시간: 3시간

- [ ] **T1.3.1.4**: 검증 로직
  - 프로젝트 이름 유효성
  - 디렉토리 이미 존재 시 에러
  - 예상 시간: 2시간

**완료 조건**:
- [x] `mloop init my-project --task binary-classification` 동작
- [x] 모든 디렉토리 및 파일 생성됨
- [x] .gitignore 올바르게 설정됨

##### 1.3.2 TrainCommand (모델 학습)
**우선순위**: 🔴 P0 (핵심 커맨드)

```
src/MLoop/Commands/TrainCommand.cs
```

**세부 태스크**:
- [ ] **T1.3.2.1**: Command 정의
  - Arguments: data-file
  - Options: --label, --time, --metric, --test-split
  - 예상 시간: 2시간

- [ ] **T1.3.2.2**: Input 검증
  - data-file 존재 확인
  - label 컬럼 존재 확인
  - 프로젝트 루트 찾기
  - 예상 시간: 2시간

- [ ] **T1.3.2.3**: TrainingEngine 호출
  - Config 병합 (CLI args + mloop.yaml + defaults)
  - TrainingEngine.TrainAsync() 호출
  - 예상 시간: 2시간

- [ ] **T1.3.2.4**: Progress 표시 (Spectre.Console)
  - 실시간 진행률 표시
  - AutoML trial 정보 출력
  - 예상 시간: 3시간

- [ ] **T1.3.2.5**: 결과 출력
  - Experiment ID
  - Best Trainer
  - Metrics 테이블
  - 저장 경로 안내
  - 예상 시간: 2시간

**완료 조건**:
- [x] `mloop train data.csv --label target` 동작
- [x] 학습 완료 및 model.zip 저장됨
- [x] Progress bar 표시됨
- [x] 결과 보기 좋게 출력됨

##### 1.3.3 PredictCommand (예측 실행)
**우선순위**: 🔴 P0 (MVP 필수)

```
src/MLoop/Commands/PredictCommand.cs
```

**세부 태스크**:
- [ ] **T1.3.3.1**: Command 정의
  - Arguments: model-path, data-file
  - Options: --output (출력 경로)
  - 예상 시간: 2시간

- [ ] **T1.3.3.2**: Input 검증
  - model.zip 존재 확인
  - data-file 존재 확인
  - 예상 시간: 1시간

- [ ] **T1.3.3.3**: PredictionEngine 호출
  - 모델 로드
  - 배치 예측 실행
  - 예상 시간: 2시간

- [ ] **T1.3.3.4**: 결과 출력
  - 콘솔 출력 (처음 10개 행)
  - 파일 저장 (outputs/predictions/result.csv)
  - 예상 시간: 2시간

**완료 조건**:
- [x] `mloop predict models/staging/model.zip data.csv` 동작
- [x] 예측 결과 정확함
- [x] CSV 파일로 저장됨

##### 1.3.4 EvaluateCommand (모델 평가)
**우선순위**: 🟡 P1 (Phase 1 후반)

```
src/MLoop/Commands/EvaluateCommand.cs
```

**세부 태스크**:
- [ ] **T1.3.4.1**: Command 정의
  - Arguments: model-path, test-data
  - Options: --output
  - 예상 시간: 2시간

- [ ] **T1.3.4.2**: Evaluator 호출
  - 모델 로드
  - 평가 실행
  - 예상 시간: 2시간

- [ ] **T1.3.4.3**: 결과 출력
  - Metrics 테이블 (Spectre.Console)
  - JSON 저장 (outputs/evaluations/)
  - 예상 시간: 2시간

**완료 조건**:
- [x] `mloop evaluate exp-001/model.zip test.csv` 동작
- [x] 메트릭 정확히 계산됨
- [x] 결과 보기 좋게 출력됨

##### 1.3.5 Progress Reporting
**우선순위**: 🟡 P1 (UX 향상)

```
src/MLoop/Infrastructure/Logging/
├── IProgressReporter.cs
└── SpectreProgressReporter.cs
```

**세부 태스크**:
- [ ] **T1.3.5.1**: `IProgressReporter` 인터페이스
  - 메서드: `Report(double percentage, string message)`
  - 예상 시간: 1시간

- [ ] **T1.3.5.2**: `SpectreProgressReporter` 구현
  - Spectre.Console Progress bar
  - 실시간 업데이트
  - 예상 시간: 3시간

**완료 조건**:
- [x] Progress bar 표시됨
- [x] 실시간 업데이트 동작

---

### 1.4 Testing (테스트)

**우선순위**: 🟡 P1 (Phase 1 후반)

#### 태스크 목록

##### 1.4.1 Unit Tests
**예상 시간**: 8시간

- [ ] **T1.4.1.1**: FileSystemManager 테스트
- [ ] **T1.4.1.2**: ProjectDiscovery 테스트
- [ ] **T1.4.1.3**: ExperimentStore 테스트 (동시성 포함)
- [ ] **T1.4.1.4**: ConfigMerger 테스트
- [ ] **T1.4.1.5**: DataLoader 테스트

##### 1.4.2 Integration Tests
**예상 시간**: 12시간

- [ ] **T1.4.2.1**: End-to-End: init → train → predict
- [ ] **T1.4.2.2**: 동시 학습 테스트 (concurrent training)
- [ ] **T1.4.2.3**: 프로젝트 루트 검색 테스트
- [ ] **T1.4.2.4**: Config 병합 통합 테스트

**완료 조건**:
- [x] Test coverage > 70%
- [x] 모든 E2E 시나리오 통과

---

### 1.5 Documentation (문서화)

**우선순위**: 🟢 P2 (Phase 1 완료 후)

#### 태스크 목록

- [ ] **T1.5.1**: CLI Reference 작성 (docs/cli-reference.md)
  - 모든 커맨드 및 옵션 설명
  - 예상 시간: 4시간

- [ ] **T1.5.2**: Getting Started 가이드 (docs/getting-started.md)
  - 설치부터 첫 모델 학습까지
  - 예상 시간: 3시간

- [ ] **T1.5.3**: Example Projects 작성
  - examples/sentiment-analysis/
  - examples/iris-classification/
  - 예상 시간: 4시간

**완료 조건**:
- [x] 신규 사용자가 문서만 보고 시작 가능
- [x] 모든 예제 동작함

---

### Phase 1 완료 조건

#### Acceptance Criteria
- [x] **AC1**: `mloop init` 커맨드로 새 프로젝트 생성 가능
- [x] **AC2**: `mloop train data.csv --label target` 커맨드로 모델 학습 가능
- [x] **AC3**: `mloop predict model.zip data.csv` 커맨드로 예측 실행 가능
- [x] **AC4**: `mloop evaluate model.zip test.csv` 커맨드로 평가 가능
- [x] **AC5**: 동시에 여러 학습 작업 실행 시 충돌 없음
- [x] **AC6**: 모든 상태가 파일시스템에 저장되어 Git으로 추적 가능

#### Definition of Done
- [x] 모든 Phase 1 태스크 완료
- [x] 단위 테스트 coverage > 70%
- [x] E2E 테스트 모두 통과
- [x] CLI Reference 및 Getting Started 문서 작성
- [x] Example 프로젝트 동작 확인
- [x] NuGet 패키지 빌드 가능 (dotnet pack)

---

## Phase 2: 실험 관리 및 서빙

**목표**: 실험 추적, 모델 레지스트리, API 서빙
**예상 기간**: 2-3주
**우선순위**: 🟡 중간

### 2.1 Experiment Management

#### 태스크 목록

##### 2.1.1 ExperimentCommand
**우선순위**: 🟡 P1

```
src/MLoop/Commands/ExperimentCommand.cs
```

**서브 커맨드**:
- [ ] **T2.1.1.1**: `mloop experiment list`
  - 모든 실험 목록 표시 (ID, 타임스탬프, 메트릭)
  - Spectre.Console Table 사용
  - 예상 시간: 3시간

- [ ] **T2.1.1.2**: `mloop experiment show <exp-id>`
  - 상세 정보 표시 (metadata, metrics, config)
  - 예상 시간: 2시간

- [ ] **T2.1.1.3**: `mloop experiment compare <exp-id1> <exp-id2>`
  - 두 실험 메트릭 비교
  - Diff 표시
  - 예상 시간: 4시간

- [ ] **T2.1.1.4**: `mloop experiment delete <exp-id>`
  - 실험 디렉토리 삭제 (확인 프롬프트)
  - 예상 시간: 2시간

**완료 조건**:
- [x] 모든 서브 커맨드 동작
- [x] 실험 비교 정확함

---

### 2.2 Model Registry

#### 태스크 목록

##### 2.2.1 ModelCommand
**우선순위**: 🟡 P1

```
src/MLoop/Commands/ModelCommand.cs
```

**서브 커맨드**:
- [ ] **T2.2.1.1**: `mloop model promote <exp-id> <staging|production>`
  - 실험 모델을 staging/production으로 승격
  - model.zip 및 metadata.json 복사
  - registry.json 업데이트
  - 예상 시간: 4시간

- [ ] **T2.2.1.2**: `mloop model list`
  - staging 및 production 모델 목록
  - 예상 시간: 2시간

- [ ] **T2.2.1.3**: `mloop model show <staging|production>`
  - 모델 상세 정보 표시
  - 예상 시간: 2시간

**완료 조건**:
- [x] 모델 승격 동작
- [x] registry.json 정확히 업데이트됨

---

### 2.3 API Serving (예외: 지속 프로세스)

#### 태스크 목록

##### 2.3.1 ServeCommand
**우선순위**: 🟡 P1

```
src/MLoop/Commands/ServeCommand.cs
src/MLoop/Infrastructure/Serving/
├── IApiServer.cs
├── ApiServer.cs
└── PredictionController.cs
```

**세부 태스크**:
- [ ] **T2.3.1.1**: ASP.NET Core Minimal API 통합
  - Microsoft.AspNetCore.Builder 의존성 추가
  - 예상 시간: 2시간

- [ ] **T2.3.1.2**: `ServeCommand` 구현
  - Arguments: model-path
  - Options: --port (기본 5000), --swagger
  - 예상 시간: 3시간

- [ ] **T2.3.1.3**: REST API 엔드포인트
  - POST /predict (단일 예측)
  - POST /predict/batch (배치 예측)
  - GET /health (헬스 체크)
  - 예상 시간: 5시간

- [ ] **T2.3.1.4**: Swagger UI 통합
  - OpenAPI 문서 자동 생성
  - Swagger UI 제공
  - 예상 시간: 2시간

**완료 조건**:
- [x] `mloop serve models/production/model.zip --port 5000` 동작
- [x] POST /predict 정확히 예측
- [x] Swagger UI 접근 가능

---

### 2.4 Long-Running Tasks Support

#### 태스크 목록

##### 2.4.1 --detach 플래그 (선택사항)
**우선순위**: 🟢 P2

**세부 태스크**:
- [ ] **T2.4.1.1**: TrainCommand에 --detach 옵션 추가
  - nohup을 통한 백그라운드 실행
  - PID 및 로그 경로 출력
  - 예상 시간: 3시간

- [ ] **T2.4.1.2**: 문서 작성 (docs/long-running-tasks.md)
  - nohup, screen, tmux 사용 가이드
  - 예상 시간: 2시간

**완료 조건**:
- [x] --detach 옵션 동작
- [x] 사용자 가이드 완성

---

### Phase 2 완료 조건

#### Acceptance Criteria
- [x] **AC1**: 실험 목록 조회 및 비교 가능
- [x] **AC2**: 모델 승격 (staging → production) 동작
- [x] **AC3**: REST API로 모델 서빙 가능
- [x] **AC4**: Swagger UI로 API 테스트 가능

#### Definition of Done
- [x] 모든 Phase 2 태스크 완료
- [x] API 서빙 E2E 테스트 통과
- [x] Long-running task 가이드 작성

---

## Phase 3: 고급 기능

**목표**: 파이프라인, 고급 AutoML 설정, 플러그인 시스템
**예상 기간**: 3-4주
**우선순위**: 🟢 낮음

### 3.1 Pipeline Automation

#### 태스크 목록

##### 3.1.1 PipelineCommand
**우선순위**: 🟢 P2

```
src/MLoop/Commands/PipelineCommand.cs
src/MLoop/Core/Pipeline/
├── IPipelineEngine.cs
├── PipelineEngine.cs
└── PipelineConfig.cs
```

**기능**:
- [ ] **T3.1.1.1**: Pipeline YAML 정의
  - steps: [preprocess, train, evaluate, promote]
  - 조건부 실행 (if metrics > threshold)
  - 예상 시간: 4시간

- [ ] **T3.1.1.2**: `mloop pipeline run <pipeline.yaml>`
  - 단계별 실행
  - 실패 시 중단 또는 계속
  - 예상 시간: 6시간

- [ ] **T3.1.1.3**: 파이프라인 결과 저장
  - pipeline-runs/run-001/ 디렉토리
  - 각 단계 결과 저장
  - 예상 시간: 3시간

**완료 조건**:
- [x] Pipeline YAML 파싱 동작
- [x] 다단계 실행 성공
- [x] 조건부 실행 동작

---

### 3.2 Advanced AutoML

#### 태스크 목록

##### 3.2.1 Custom Trainers
**우선순위**: 🟢 P2

**기능**:
- [ ] **T3.2.1.1**: Trainer 선택 옵션
  - --trainers LightGbm,FastTree (특정 알고리즘만)
  - 예상 시간: 3시간

- [ ] **T3.2.1.2**: Hyperparameter 튜닝 범위 설정
  - YAML 설정으로 파라미터 범위 지정
  - 예상 시간: 4시간

**완료 조건**:
- [x] Trainer 필터링 동작
- [x] 커스텀 파라미터 적용됨

##### 3.2.2 Feature Engineering
**우선순위**: 🟢 P3

**기능**:
- [ ] **T3.2.2.1**: 자동 특징 생성
  - Missing value imputation
  - One-hot encoding
  - Normalization
  - 예상 시간: 6시간

---

### 3.3 Plugin System

#### 태스크 목록

##### 3.3.1 Plugin Infrastructure
**우선순위**: 🟢 P3

```
src/MLoop/Infrastructure/Plugins/
├── IPlugin.cs
├── PluginLoader.cs
└── PluginManifest.cs
```

**기능**:
- [ ] **T3.3.1.1**: `IPlugin` 인터페이스 정의
  - Name, Version, Initialize()
  - 예상 시간: 2시간

- [ ] **T3.3.1.2**: `PluginLoader` 구현
  - .mloop/plugins/ 디렉토리 스캔
  - Assembly 로드 및 인스턴스 생성
  - 예상 시간: 5시간

- [ ] **T3.3.1.3**: Plugin Types
  - Custom Trainers
  - Custom Data Loaders
  - Custom Metrics
  - 예상 시간: 8시간

**완료 조건**:
- [x] 플러그인 로드 동작
- [x] 샘플 플러그인 작성 및 테스트

---

### Phase 3 완료 조건

#### Acceptance Criteria
- [x] **AC1**: Pipeline YAML로 다단계 워크플로우 실행 가능
- [x] **AC2**: Trainer 및 Hyperparameter 커스터마이징 가능
- [x] **AC3**: 플러그인 시스템 동작

---

## Phase 4: 프로덕션 준비

**목표**: 안정성, 보안, 배포 준비
**예상 기간**: 2주
**우선순위**: 🟡 중간

### 4.1 Production Features

#### 태스크 목록

##### 4.1.1 에러 처리 및 로깅
**우선순위**: 🟡 P1

- [ ] **T4.1.1.1**: Structured Logging
  - Serilog 통합
  - 파일 로깅 (.mloop/logs/)
  - 예상 시간: 4시간

- [ ] **T4.1.1.2**: Global Exception Handling
  - 모든 커맨드에 try-catch
  - 사용자 친화적 에러 메시지
  - 예상 시간: 3시간

##### 4.1.2 성능 최적화
**우선순위**: 🟢 P2

- [ ] **T4.1.2.1**: 모델 캐싱
  - 동일 모델 재로드 방지
  - 예상 시간: 3시간

- [ ] **T4.1.2.2**: 대용량 파일 처리
  - Streaming CSV 로드
  - 메모리 효율적 예측
  - 예상 시간: 5시간

##### 4.1.3 보안
**우선순위**: 🟡 P1

- [ ] **T4.1.3.1**: Input Validation
  - Path traversal 방지
  - 파일 크기 제한
  - 예상 시간: 3시간

- [ ] **T4.1.3.2**: 모델 암호화 (선택사항)
  - model.zip 암호화/복호화
  - 예상 시간: 4시간

---

### 4.2 Deployment

#### 태스크 목록

##### 4.2.1 NuGet 패키징
**우선순위**: 🔴 P0

- [ ] **T4.2.1.1**: NuGet 메타데이터 완성
  - Description, Authors, License, ProjectUrl
  - 예상 시간: 1시간

- [ ] **T4.2.1.2**: 패키지 테스트
  - dotnet pack → dotnet tool install -g
  - 설치 및 실행 확인
  - 예상 시간: 2시간

- [ ] **T4.2.1.3**: NuGet.org 퍼블리시
  - API 키 설정
  - dotnet nuget push
  - 예상 시간: 1시간

**완료 조건**:
- [x] `dotnet tool install -g mloop` 동작
- [x] NuGet.org에서 설치 가능

##### 4.2.2 GitHub Actions CI/CD
**우선순위**: 🟡 P1

- [ ] **T4.2.2.1**: Build & Test Workflow
  - 모든 커밋에 빌드 및 테스트
  - 예상 시간: 3시간

- [ ] **T4.2.2.2**: Release Workflow
  - 태그 푸시 시 자동 NuGet 퍼블리시
  - 예상 시간: 2시간

---

### Phase 4 완료 조건

#### Acceptance Criteria
- [x] **AC1**: 프로덕션급 에러 처리 및 로깅
- [x] **AC2**: NuGet.org에서 설치 가능
- [x] **AC3**: CI/CD 자동화

---

## 우선순위 가이드

### 우선순위 레벨

| 레벨 | 의미 | 예시 |
|------|------|------|
| 🔴 P0 | **Critical** - 즉시 필요 | FileSystemManager, TrainCommand |
| 🟡 P1 | **High** - Phase 내 필수 | ExperimentCommand, ModelCommand |
| 🟢 P2 | **Medium** - Phase 내 선택 | --detach 플래그, Logging |
| ⚪ P3 | **Low** - 미래 고려 | Plugin System, Feature Engineering |

### 의존성 체인

**Phase 1 의존성 순서**:
```
1. FileSystemManager → ProjectDiscovery
2. ConfigLoader → ConfigMerger
3. ExperimentStore (의존: FileSystemManager)
4. DataLoader
5. AutoMLRunner → TrainingEngine (의존: ExperimentStore, DataLoader)
6. PredictionEngine
7. InitCommand (의존: FileSystemManager, ConfigLoader)
8. TrainCommand (의존: TrainingEngine)
9. PredictCommand (의존: PredictionEngine)
10. EvaluateCommand (의존: PredictionEngine)
```

**권장 구현 순서**:
1. Infrastructure Layer 먼저 (T1.1.1 → T1.1.2 → T1.1.4)
2. Core Layer (T1.2.1 → T1.2.2 → T1.2.3)
3. CLI Layer (T1.3.1 → T1.3.2 → T1.3.3)
4. Testing (T1.4)
5. Documentation (T1.5)

---

## 진행 상황 추적

### 전체 진행률

| Phase | 진행률 | 상태 |
|-------|--------|------|
| Phase 0 | 100% | ✅ 완료 |
| Phase 1 | 0% | 🔄 대기 중 |
| Phase 2 | 0% | 🔄 대기 중 |
| Phase 3 | 0% | 🔄 대기 중 |
| Phase 4 | 0% | 🔄 대기 중 |

### Phase 1 상세 진행률

| 컴포넌트 | 태스크 수 | 완료 | 진행률 |
|----------|-----------|------|--------|
| Infrastructure | 14 | 0 | 0% |
| Core | 17 | 0 | 0% |
| CLI | 16 | 0 | 0% |
| Testing | 9 | 0 | 0% |
| Documentation | 3 | 0 | 0% |

---

## 다음 단계

### 즉시 시작 가능한 태스크 (No Dependencies)

1. **T1.1.1.1**: `IFileSystemManager` 인터페이스 정의 (1시간)
2. **T1.1.4.1**: `MLoopConfig` 데이터 모델 정의 (1시간)
3. **T1.2.1.1**: `IDataLoader` 인터페이스 정의 (1시간)
4. **T1.2.2.1**: `TrainingConfig` 데이터 모델 정의 (1시간)

### 첫 주 목표 (Week 1)

- [ ] Infrastructure Layer 완성 (T1.1.1 ~ T1.1.4)
- [ ] Data Loader 완성 (T1.2.1)
- [ ] InitCommand 완성 (T1.3.1)

**예상 시간**: 30-35시간

### 둘째 주 목표 (Week 2)

- [ ] AutoML Engine 완성 (T1.2.2)
- [ ] TrainCommand 완성 (T1.3.2)
- [ ] 첫 E2E 테스트 (init → train)

**예상 시간**: 35-40시간

---

## 참고 자료

- [ARCHITECTURE.md](ARCHITECTURE.md) - 전체 아키텍처 설계
- [README.md](../README.md) - 프로젝트 개요
- [ML.NET Documentation](https://docs.microsoft.com/dotnet/machine-learning/)
- [System.CommandLine Documentation](https://github.com/dotnet/command-line-api)
- [Spectre.Console Documentation](https://spectreconsole.net/)

---

**작성자**: MLoop Development Team
**마지막 업데이트**: 2024-11-03
**버전**: 0.1.0-alpha
