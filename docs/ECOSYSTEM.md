# MLoop 에코시스템 정의서

---

## Part 1. 전체 에코시스템 요약

### 컴포넌트 역할 한눈에 보기

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLoop Studio                             │
│              "대화로 만드는 ML" - 최종 제품                      │
│                                                                 │
│   SDK 직접 참조: MLoop.Core, MLoop.DataStore, MLoop.Ops        │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   mloop-mcp     │  │  MLoop (SDK)    │  │   FilePrepper   │
│                 │  │                 │  │                 │
│  AI ↔ MLoop     │  │  ML 빌드 SDK    │  │  파일 전처리    │
│  브릿지 (CLI)   │  │  + CLI/API 도구 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────────────────────────┐
│    Ironbees     │  │  MLoop 프로젝트 구조                │
│                 │  │  ┌─────────────┬─────────────────┐  │
│  에이전트       │  │  │ src/ (SDK)  │ tools/ (실행)   │  │
│  프레임워크     │  │  │ NuGet 배포  │ dotnet tool     │  │
└─────────────────┘  │  └─────────────┴─────────────────┘  │
                     └─────────────────────────────────────┘
```

### 컴포넌트 역할 요약표

| 컴포넌트 | 한 줄 정의 | 타겟 사용자 |
|----------|------------|-------------|
| **MLoop** | ML 모델을 빌드하고 서빙하는 CLI/API 도구 | 개발자 |
| **MLoop Studio** | MLoop 기반 노코드 ML 웹 플랫폼 | 비개발자, 비즈니스 |
| **mloop-mcp** | MLoop을 AI 에이전트가 사용할 수 있게 노출하는 MCP Server | AI 클라이언트 |
| **FilePrepper** | 다양한 파일을 ML 학습 가능한 형태로 전처리하는 라이브러리 | 개발자 |
| **Ironbees** | AI 에이전트를 정의하고 도구를 실행하는 오케스트레이션 프레임워크 | 개발자 |

### 의존성 방향

```
MLoop Studio
    ├── uses → MLoop (학습/예측)
    ├── uses → mloop-mcp (AI 대화)
    └── uses → FilePrepper (전처리)

mloop-mcp
    ├── calls → MLoop CLI
    └── runs on → Ironbees (또는 다른 MCP 호스트)

MLoop
    └── (optional) calls → FilePrepper

Ironbees
    └── requires → AI Provider (외부)
```

---

## Part 2. 외부 의존성 (간략)

### FilePrepper

```
역할: 파일 전처리 라이브러리
범위:
  ✅ 인코딩 변환 (CP949, EUC-KR → UTF-8)
  ✅ 포맷 변환 (XLS, JSON → CSV)
  ✅ 이미지 전처리
  ✅ 스키마 추론
  ✅ 데이터 정제 (결측치, 이상치)
  ❌ ML 학습
  ❌ 에이전트/AI
```

### Ironbees

```
역할: AI 에이전트 오케스트레이션 프레임워크
범위:
  ✅ 에이전트 정의 (YAML 기반)
  ✅ 도구 실행 관리
  ✅ 대화 컨텍스트 관리
  ✅ 멀티 에이전트 워크플로우
  ❌ AI Provider 구현 (외부 주입)
  ❌ 특정 도메인 로직
```

---

## Part 3. MLoop 상세

### 3.1 철학

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "MLoop은 grep이다"                                            │
│                                                                 │
│   grep이 패턴을 찾듯, MLoop은 모델을 만든다.                    │
│   하나의 일을 잘 한다. 그 이상도 이하도 아니다.                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 원칙:**

| 원칙 | 설명 |
|------|------|
| **Single Purpose** | ML 모델 빌드/서빙만 한다 |
| **Stateless** | 명령 실행 → 완료 → 종료, Daemon 없음 |
| **Filesystem-First** | 모든 상태는 파일, DB 없음, Git 친화적 |
| **Composable** | 다른 도구와 조합하여 파이프라인 구성 |
| **Zero AI Dependency** | LLM/에이전트 의존성 없음 |
| **Convention Over Configuration** | 설정 없이 즉시 동작 |

### 3.2 가치

**개발자에게:**
- 3-command workflow: `init → train → predict`
- Git으로 실험 추적 (DB 불필요)
- CI/CD 파이프라인에 자연스럽게 통합

**조직에게:**
- ML.NET CLI 단종 후 대안
- .NET 생태계 유지
- 인프라 비용 최소화

### 3.3 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                         MLoop 역할                              │
│                                                                 │
│   입력: 정제된 데이터 (CSV)                                     │
│   출력: 학습된 모델, 예측 결과, API 엔드포인트                  │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │  init   │ →  │  train  │ →  │ promote │ →  │  serve  │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 범위

**In Scope (하는 것):**

| 기능 | 명령 | 설명 |
|------|------|------|
| 프로젝트 초기화 | `mloop init` | 프로젝트 구조 생성 |
| 데이터 정보 | `mloop info` | 데이터셋 프로파일링 |
| 모델 학습 | `mloop train` | AutoML 기반 학습 |
| 모델 평가 | `mloop evaluate` | 성능 측정 |
| 실험 비교 | `mloop compare` | 두 실험 메트릭 비교 |
| 실험 관리 | `mloop list` | 실험 목록 조회 |
| 모델 프로모션 | `mloop promote` | staging → production |
| 모델 롤백 | `mloop rollback` | 이전 모델로 복원 |
| 배치 예측 | `mloop predict` | 일괄 예측 실행 |
| API 서빙 | `mloop serve` | REST API 제공 |
| 배포 지원 | `mloop docker` | Dockerfile 생성 |

**Out of Scope (하지 않는 것):**

| 기능 | 이유 | 대안 |
|------|------|------|
| 파일 전처리 | 단일 책임 | FilePrepper |
| AI 대화/에이전트 | 도구 순수성 | mloop-mcp + Ironbees |
| 피드백 수집 | 운영 영역 | MLoop.DataStore |
| 재학습 스케줄링 | 운영 영역 | MLoop.Ops, cron |
| Drift 감지 | 모니터링 영역 | 외부 도구 |
| 사용자 인증 | 제품 영역 | MLoop Studio |

### 3.5 아키텍처

MLoop은 **SDK**와 **Tools**로 분리됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│  src/ (SDK)                      │  tools/ (실행 도구)         │
│  NuGet.org 패키지 배포           │  dotnet tool / Docker 배포  │
├──────────────────────────────────┼─────────────────────────────┤
│  MLoop.Core                      │  MLoop.CLI                  │
│  MLoop.DataStore                 │  MLoop.API                  │
│  MLoop.Extensibility             │                             │
│  MLoop.Ops                       │                             │
└──────────────────────────────────┴─────────────────────────────┘
                    │                           │
                    ▼                           ▼
            라이브러리 참조              subprocess / HTTP 호출
            (MLoop Studio)              (mloop-mcp, 외부 시스템)
```

#### SDK (`src/`) - NuGet 패키지

```
src/
├── MLoop.Core/                # ML 엔진 라이브러리
│   ├── AutoML/               # ML.NET AutoML 래퍼
│   ├── Data/                 # 데이터 로딩
│   ├── Pipeline/             # ML 파이프라인
│   └── Models/               # 도메인 모델
│
├── MLoop.DataStore/           # 운영 데이터 저장
│   ├── FilePredictionLogger  # 예측 로그 (JSONL)
│   ├── FileFeedbackCollector # 피드백 수집
│   └── FileDataSampler       # 재학습 데이터 샘플링
│
├── MLoop.Extensibility/       # 확장 인터페이스
│   ├── IPreprocessingScript
│   ├── IHook<T>
│   └── ICustomMetric
│
└── MLoop.Ops/                 # 운영 자동화
    ├── FileModelComparer     # 모델 비교
    ├── TimeBasedTrigger      # 시간 기반 트리거
    └── FeedbackBasedTrigger  # 피드백 기반 트리거
```

#### Tools (`tools/`) - 실행 도구

```
tools/
├── MLoop.CLI/                 # 명령줄 도구
│   └── Commands/             # dotnet tool install mloop
│
└── MLoop.API/                 # REST API 서버
    └── (mloop serve 또는 독립 배포)
```

#### 사용 시나리오

| 시나리오 | 사용 방식 | 예시 |
|----------|-----------|------|
| **MLoop Studio** | SDK 직접 참조 | `TrainingEngine.TrainAsync()` |
| **mloop-mcp** | CLI subprocess | `exec("mloop train ...")` |
| **외부 시스템** | API HTTP 호출 | `POST /api/train` |
| **개발자** | CLI 직접 실행 | `mloop train --data ...` |

---

## Part 4. MLoop Studio 상세

### 4.1 철학

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "ML for Everyone"                                             │
│                                                                 │
│   ML 지식이 없어도, 코딩을 몰라도                               │
│   파일 업로드와 대화만으로 ML 모델을 만든다.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 원칙:**

| 원칙 | 설명 |
|------|------|
| **Zero ML Knowledge** | ML 용어/개념 몰라도 사용 가능 |
| **Conversation-First** | 스무고개 대화로 모델 완성 |
| **End-to-End** | 업로드 → 모델 → 엔드포인트 원스톱 |
| **Self-Improving** | 피드백으로 자동 개선 |
| **Composable Backend** | 내부는 독립 컴포넌트 조합 |

### 4.2 가치

**비개발자에게:**
- 코딩 없이 ML 모델 생성
- 자연어 대화로 요구사항 전달
- 즉시 사용 가능한 API 엔드포인트

**비즈니스에게:**
- ML 도입 장벽 제거
- 개발자 병목 해소
- 빠른 프로토타이핑

### 4.3 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                      MLoop Studio 역할                          │
│                                                                 │
│   사용자 여정:                                                  │
│                                                                 │
│   [파일 업로드] → [스무고개 대화] → [모델 빌드] → [엔드포인트]  │
│                                                                 │
│        │              │               │              │          │
│        ▼              ▼               ▼              ▼          │
│   FilePrepper     Ironbees       MLoop.Core      MLoop.Core    │
│                   (AI 대화)      .TrainAsync()   .PredictAsync()│
│                                                                 │
│   ※ SDK 직접 참조 - CLI/API subprocess 호출 없음               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 범위

**In Scope (하는 것):**

| 기능 | 설명 |
|------|------|
| **웹 UI** | 파일 업로드, 대화, 대시보드 |
| **스무고개 대화** | AI와 대화로 모델 요구사항 도출 |
| **프로젝트 관리** | 여러 모델/프로젝트 관리 |
| **데이터 관리 UI** | 학습 데이터, 피드백 데이터 조회/관리 |
| **모델 대시보드** | 성능 지표, 실험 비교 시각화 |
| **엔드포인트 관리** | API 키 발급, 사용량 조회 |
| **팀 협업** | 사용자/팀 관리 (멀티테넌트) |
| **자동 재학습** | 피드백 기반 모델 개선 (MLoop.Ops 호출) |

**Out of Scope (하지 않는 것, SDK에 위임):**

| 기능 | 위임 대상 | 참조 방식 |
|------|-----------|-----------|
| ML 학습 로직 | MLoop.Core | SDK 직접 참조 |
| 데이터 저장 로직 | MLoop.DataStore | SDK 직접 참조 |
| 재학습 판단 로직 | MLoop.Ops | SDK 직접 참조 |
| 에이전트 오케스트레이션 | Ironbees | 라이브러리 참조 |
| 파일 전처리 | FilePrepper | 라이브러리 참조 |

※ MLoop Studio는 .NET 프로젝트이므로 MLoop SDK를 직접 참조합니다.
   CLI나 API를 subprocess/HTTP로 호출하지 않습니다.

### 4.5 아키텍처

```
mloop-studio/
├── src/
│   ├── MLoop.Studio.Web/        # 프론트엔드 (Blazor/React)
│   │   ├── Pages/              # 대화, 대시보드, 관리
│   │   └── Components/         # UI 컴포넌트
│   │
│   └── MLoop.Studio.Backend/    # 백엔드 서비스
│       ├── Orchestration/      # 워크플로우 관리
│       ├── Sessions/           # 대화 세션 관리
│       └── Endpoints/          # 엔드포인트 프로비저닝
│
└── (NuGet 패키지 참조)
    │
    ├── MLoop SDK (직접 참조)
    │   ├── MLoop.Core          # ML 학습/예측 엔진
    │   ├── MLoop.DataStore     # 예측 로그, 피드백
    │   └── MLoop.Ops           # 재학습 트리거
    │
    └── 외부 라이브러리
        ├── FilePrepper         # 파일 전처리
        └── Ironbees            # AI 에이전트 오케스트레이션
```

**참조 방식:**
- ✅ SDK 직접 참조: `dotnet add package MLoop.Core`
- ❌ CLI 호출 없음: subprocess 오버헤드 제거
- ❌ API 호출 없음: HTTP 오버헤드 제거

### 4.6 사용자 시나리오

```
1. 파일 업로드
   사용자: "customer_data.xlsx 업로드"
   시스템: FilePrepper.ProcessAsync() → CSV 변환, 인코딩 수정

2. 스무고개 대화
   AI: "어떤 것을 예측하고 싶으세요?"
   사용자: "고객이 이탈할지 예측하고 싶어요"
   AI: "Churn 컬럼이 이탈 여부인가요?"
   사용자: "네"
   AI: "분류 모델을 만들게요. 학습을 시작할까요?"

3. 모델 빌드
   시스템: TrainingEngine.TrainAsync() 직접 호출  ← SDK 사용
   AI: "학습 완료! 정확도 92%입니다. 배포할까요?"

4. 엔드포인트 제공
   시스템: PredictionEngine 인스턴스 생성, API 키 발급
   AI: "여기 API 엔드포인트입니다: https://..."

5. 운영 중 개선
   시스템: FilePredictionLogger.LogAsync()        ← DataStore SDK
   시스템: FeedbackBasedTrigger.EvaluateAsync()   ← Ops SDK
   시스템: 피드백 1000건 도달 → 자동 재학습 트리거
   AI: "새 모델이 더 좋아서 자동 배포했어요. 정확도 94%"
```

**핵심 차이점:**
- CLI subprocess 호출 ❌ → SDK 메서드 직접 호출 ✅
- 프로세스 생성 오버헤드 없음
- 타입 안전성 보장
- 에러 핸들링 통합

---

## Part 5. mloop-mcp 상세

### 5.1 철학

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "AI가 MLoop을 사용한다. MLoop이 AI를 품지 않는다."            │
│                                                                 │
│   MLoop CLI의 모든 기능을 AI 에이전트가                         │
│   도구로서 호출할 수 있게 브릿지 역할만 한다.                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 원칙:**

| 원칙 | 설명 |
|------|------|
| **Pure Bridge** | 변환만, 로직 없음 |
| **MLoop Parity** | MLoop CLI와 1:1 매핑 |
| **Protocol Compliance** | MCP 표준 준수 |
| **Stateless** | 상태 없음, MLoop이 상태 관리 |
| **Prompt Separation** | 프롬프트는 텍스트 파일로 분리 |

### 5.2 가치

**AI 클라이언트에게:**
- Claude Desktop, Cursor 등에서 MLoop 사용
- 자연어로 ML 작업 지시

**개발자에게:**
- MLoop 핵심 코드 수정 없이 AI 통합
- 프롬프트만 수정하여 에이전트 개선

### 5.3 역할

```
┌─────────────────────────────────────────────────────────────────┐
│                       mloop-mcp 역할                            │
│                                                                 │
│   AI Client ←──[MCP Protocol]──→ mloop-mcp ←──[CLI]──→ MLoop   │
│                                                                 │
│   "MLoop CLI를 MCP 도구로 노출하는 어댑터"                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 범위

**In Scope (하는 것):**

| 기능 | 설명 |
|------|------|
| **도구 정의** | MLoop CLI 명령을 MCP 도구로 정의 |
| **명령 변환** | MCP 호출 → CLI 명령 변환 |
| **결과 파싱** | CLI 출력 → 구조화된 응답 |
| **프롬프트 제공** | ML 전문가 시스템 프롬프트 |

**Out of Scope (하지 않는 것):**

| 기능 | 이유 | 대안 |
|------|------|------|
| ML 로직 | MLoop 영역 | MLoop |
| 에이전트 오케스트레이션 | 프레임워크 영역 | Ironbees |
| AI Provider | 외부 | OpenAI, Claude API |
| 상태 관리 | MLoop이 담당 | MLoop filesystem |

### 5.5 아키텍처

```
mloop-mcp/
├── src/
│   ├── tools/                  # MCP 도구 정의
│   │   ├── train.ts           # mloop train → mloop_train
│   │   ├── predict.ts         # mloop predict → mloop_predict
│   │   ├── list.ts            # mloop list → mloop_list
│   │   ├── compare.ts         # mloop compare → mloop_compare
│   │   ├── promote.ts         # mloop promote → mloop_promote
│   │   └── info.ts            # mloop info → mloop_info
│   │
│   ├── prompts/                # 시스템 프롬프트
│   │   └── ml-expert.md       # ML 전문가 프롬프트
│   │
│   └── server.ts               # MCP 서버 진입점
│
├── package.json
└── README.md
```

### 5.6 도구 명세

```typescript
// tools/train.ts
export const trainTool = {
  name: "mloop_train",
  description: "Train ML model using AutoML. Returns experiment ID and metrics.",
  inputSchema: {
    type: "object",
    properties: {
      data: {
        type: "string",
        description: "Path to training CSV file"
      },
      label: {
        type: "string",
        description: "Target column name to predict"
      },
      task: {
        type: "string",
        enum: ["regression", "binary-classification", "multiclass-classification"],
        description: "ML task type"
      },
      time: {
        type: "number",
        description: "Training time in seconds",
        default: 60
      },
      name: {
        type: "string",
        description: "Model name (default: 'default')"
      }
    },
    required: ["data", "label", "task"]
  },
  handler: async (params) => {
    // mloop train 명령 실행
    const result = await execMLoopCLI("train", params);
    return parseTrainResult(result);
  }
};
```

### 5.7 프롬프트 예시

```markdown
# prompts/ml-expert.md

You are an ML expert helping users build models with MLoop.

## Your Role
- Guide users through ML model creation
- Recommend appropriate task types based on their goals
- Execute MLoop commands and explain results
- Suggest improvements based on metrics

## Available Tools
- `mloop_train`: Train a new model
- `mloop_predict`: Run predictions
- `mloop_list`: View all experiments
- `mloop_compare`: Compare two models
- `mloop_promote`: Deploy model to production
- `mloop_info`: Get dataset information

## Workflow Guide
1. Ask what they want to predict
2. Get dataset information with mloop_info
3. Recommend task type (classification vs regression)
4. Start with short training (60s) for quick feedback
5. Review metrics and suggest improvements
6. Promote when user is satisfied

## Communication Style
- Use simple language, avoid ML jargon
- Explain metrics in business terms
- Always confirm before executing commands
- Celebrate successes, be encouraging on failures
```

---

## Part 6. 요약 비교표

| 항목 | MLoop | MLoop Studio | mloop-mcp |
|------|-------|--------------|-----------|
| **타입** | CLI/라이브러리 | 웹 플랫폼 | MCP Server |
| **사용자** | 개발자 | 비개발자 | AI 클라이언트 |
| **인터페이스** | 명령줄 | 웹 UI | MCP 프로토콜 |
| **상태 관리** | Filesystem | DB + Filesystem | Stateless |
| **AI 의존성** | ❌ 없음 | ✅ 있음 (Ironbees) | ❌ 없음 (호스트가 제공) |
| **독립 실행** | ✅ 가능 | ❌ MLoop 필요 | ❌ MLoop + MCP Host 필요 |
| **주요 가치** | 단순함, 조합성 | 접근성, 편의성 | AI 통합 |

---

## Part 7. 의존성 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                        의존성 방향                              │
│                                                                 │
│   MLoop Studio (.NET 웹앱)                                      │
│       │                                                         │
│       ├──→ MLoop.Core (SDK, NuGet)      ← 직접 참조            │
│       ├──→ MLoop.DataStore (SDK, NuGet) ← 직접 참조            │
│       ├──→ MLoop.Ops (SDK, NuGet)       ← 직접 참조            │
│       ├──→ Ironbees (필수)                                     │
│       ├──→ FilePrepper (필수)                                  │
│       └──→ AI Provider (필수, 외부)                            │
│       ❌ CLI/API 불필요 (SDK 직접 사용)                         │
│                                                                 │
│   mloop-mcp (TypeScript)                                        │
│       │                                                         │
│       └──→ MLoop CLI (subprocess 호출)                         │
│                                                                 │
│   외부 시스템 (Python, Java 등)                                 │
│       │                                                         │
│       ├──→ MLoop CLI (subprocess)                              │
│       └──→ MLoop API (HTTP 호출)                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   MLoop 프로젝트 내부 구조                                      │
│                                                                 │
│   src/ (SDK - NuGet 패키지)                                     │
│       ├── MLoop.Core ──→ ML.NET (필수)                         │
│       ├── MLoop.DataStore ──→ (독립, 파일시스템 기반)          │
│       ├── MLoop.Extensibility ──→ (독립, 인터페이스만)         │
│       └── MLoop.Ops ──→ MLoop.DataStore (필수)                 │
│                                                                 │
│   tools/ (실행 도구 - dotnet tool / Docker)                     │
│       ├── MLoop.CLI ──→ src/* (SDK 전체)                       │
│       └── MLoop.API ──→ src/* (SDK 전체)                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   외부 프로젝트                                                 │
│                                                                 │
│   Ironbees ──→ AI Provider (필수, 외부 주입)                   │
│   FilePrepper ──→ (독립, 외부 의존성 최소)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 8. 핵심 메시지

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   MLoop      = "모델을 만드는 도구"      (grep)                 │
│   mloop-mcp  = "AI가 도구를 쓰는 방법"   (man page + 인터페이스)│
│   MLoop Studio = "누구나 쓰는 제품"      (GUI 앱)               │
│                                                                 │
│   도구는 단순하게, 제품은 풍부하게, 조합은 자유롭게.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: January 2026
**Version**: v1.6.0
