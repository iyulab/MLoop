# MLoop Architecture Refactoring Plan

**Version**: v1.2.0 → v2.0.0
**Date**: January 2026
**Status**: In Progress (Phase A ✅, Phase B ✅, Phase C ✅, Phase D pending)

---

## 1. Executive Summary

### 1.1 Core Change

```
Before (v1.1.0):                    After (v2.0.0):
────────────────                    ────────────────
MLoop.Core                          MLoop.Core (순수 ML 엔진)
MLoop.CLI                           MLoop.CLI (순수 CLI)
MLoop.API                           MLoop.API
MLoop.Extensibility                 MLoop.Extensibility
MLoop.AIAgent ←── 제거              MLoop.DataStore ←── 신규
                                    MLoop.Ops ←── 신규

                                    [별도 레포]
                                    mloop-mcp (MCP Server)
                                    mloop-studio (Web Platform)
```

### 1.2 Philosophy Change

```
Old: "MLoop이 AI를 품는다" (MLoop.AIAgent 내장)
New: "AI가 MLoop을 사용한다" (mloop-mcp가 CLI 호출)
```

---

## 2. Impact Analysis

### 2.1 Affected Components

| Component | Action | Impact |
|-----------|--------|--------|
| `MLoop.AIAgent/` | **DELETE** | 모든 AI 관련 코드 제거 |
| `agents/` 폴더 | **DELETE** | YAML 에이전트 정의 제거 |
| Memory Services | **DELETE** | Pattern/Failure 서비스 제거 |
| `mloop agent` 명령 | **DELETE** | CLI에서 제거 |
| `mloop orchestrate` 명령 | **DELETE** | CLI에서 제거 |
| TrainingMemoryCollector | **DELETE** | 백그라운드 수집 제거 |

### 2.2 Preserved Components

| Component | Status | Notes |
|-----------|--------|-------|
| `MLoop.Core/` | ✅ KEEP | 순수 ML 엔진 유지 |
| `MLoop.CLI/` | ✅ KEEP | AI 명령만 제거 |
| `MLoop.API/` | ✅ KEEP | 변경 없음 |
| `MLoop.Extensibility/` | ✅ KEEP | 변경 없음 |
| EncodingDetector | ✅ KEEP | Core에 유지 |
| ErrorSuggestions | ✅ KEEP | Core에 유지 |
| DataQualityAnalyzer | ✅ KEEP | Core에 유지 |

### 2.3 ROADMAP.md Impact

| Phase | Status | v1.2.0 After |
|-------|--------|--------------|
| Phase 0-4 | ✅ Complete | ✅ 유지 (Core 기능) |
| Phase 5 (Memory) | ✅ Complete | ❌ **삭제** (AIAgent 제거) |
| Phase 6 (Agent Intel) | ✅ Complete | ⚠️ **부분 삭제** (T6.2, T6.3만 유지) |
| Phase 7 | ✅ Complete | ✅ 유지 |
| Phase 8 | ✅ Complete | ⚠️ **T8.2 삭제** (Memory 관련) |
| Phase 9 | 📋 Planning | ❌ **취소** (Memory 기반 기능) |

---

## 3. Phased Execution Plan

### Phase A: Cleanup (v1.2.0-alpha)
**Goal**: AI 의존성 제거, 순수 CLI 도구로 회귀

```
Duration: 1-2 days
Risk: Low (기능 제거만)
```

#### A.1 MLoop.AIAgent 제거
- [x] `src/MLoop.AIAgent/` 폴더 삭제
- [x] `MLoop.sln`에서 프로젝트 참조 제거
- [x] `Directory.Packages.props`에서 AI 패키지 제거
  - Ironbees.AgentMode
  - MemoryIndexer
  - Microsoft.Extensions.AI.*
- [x] `tests/MLoop.AIAgent.Tests/` 삭제

#### A.2 CLI 정리
- [x] `AgentCommand.cs` 삭제
- [x] `OrchestrateCommand.cs` 삭제
- [x] `TrainingMemoryCollector` 참조 제거
- [x] DI 등록에서 AI 서비스 제거

#### A.3 에이전트 폴더 정리
- [x] `agents/` 폴더 삭제 (YAML 에이전트 정의) - N/A (이전에 삭제됨)
- [x] `.mloop/agents/` 문서화 업데이트

#### A.4 문서 업데이트
- [x] README.md에서 AI 기능 제거
- [x] docs/ARCHITECTURE.md 업데이트 (AIAgent 제거)
- [ ] docs/AI-AGENTS.md → mloop-mcp 참조로 변경 (Phase D)
- [ ] docs/AI-AGENT-USAGE.md → mloop-mcp 참조로 변경 (Phase D)
- [ ] docs/AI-AGENT-ARCHITECTURE.md → mloop-mcp 참조로 변경 또는 삭제 (Phase D)

---

### Phase B: New Projects (v1.2.0-beta) ✅ COMPLETE
**Goal**: DataStore, Ops 프로젝트 스켈레톤 생성

```
Duration: 1 day
Risk: Low (스켈레톤만)
Status: ✅ Completed (January 12, 2026)
```

#### B.1 MLoop.DataStore 생성 ✅
```
src/MLoop.DataStore/
├── MLoop.DataStore.csproj ✅
├── Interfaces/
│   ├── IPredictionLogger.cs ✅
│   ├── IFeedbackCollector.cs ✅
│   └── IDataSampler.cs ✅
├── Services/
│   └── (구현 예정)
└── Models/
    └── (Interfaces에 record로 포함)
```

#### B.2 MLoop.Ops 생성 ✅
```
src/MLoop.Ops/
├── MLoop.Ops.csproj ✅
├── Interfaces/
│   ├── IRetrainingTrigger.cs ✅
│   ├── IModelComparer.cs ✅
│   └── IPromotionManager.cs ✅
├── Services/
│   └── (구현 예정)
└── Models/
    └── (Interfaces에 record로 포함)
```

#### B.3 솔루션 업데이트 ✅
- [x] MLoop.sln에 새 프로젝트 추가
- [x] 빌드 검증 (0 errors)
- [x] 테스트 검증 (389 passed)

---

### Phase C: External Repos (v1.2.0) ✅ COMPLETE
**Goal**: 서브모듈 설정

```
Duration: 1 day (레포 생성 후)
Risk: Low
Status: ✅ Completed (January 12, 2026)
```

#### C.1 mloop-mcp 레포 ✅
```bash
git submodule add https://github.com/iyulab/mloop-mcp.git mcp
```
- [x] 레포 생성: https://github.com/iyulab/mloop-mcp
- [x] 서브모듈 추가: `mcp/`

#### C.2 mloop-studio 레포 ✅
```bash
git submodule add https://github.com/iyulab/mloop-studio.git studio
```
- [x] 레포 생성: https://github.com/iyulab/mloop-studio
- [x] 서브모듈 추가: `studio/`

---

### Phase D: Documentation (v1.2.0-release)
**Goal**: 문서 정합성 확보

```
Duration: 1 day
Risk: Low
```

#### D.1 ROADMAP.md 재작성
- [ ] Phase 5, 6, 9 "deprecated" 마킹
- [ ] 새로운 Phase 구조 추가:
  - Phase 10: DataStore (v1.3.0)
  - Phase 11: Ops (v1.4.0)
  - Phase 12: Studio (v2.0.0)

#### D.2 ARCHITECTURE.md 업데이트
- [ ] 5개 프로젝트 → 6개 프로젝트 구조
- [ ] 외부 컴포넌트 (mloop-mcp, mloop-studio) 추가

#### D.3 PHILOSOPHY.md 생성
- [ ] 사용자 제공 철학 문서 정리
- [ ] Unix 철학, 경계 정의 명문화

#### D.4 CLI-REFERENCE.md 생성
- [ ] 모든 명령어 레퍼런스
- [ ] AI 명령 제거 반영

---

## 4. Task Breakdown

### Phase A Tasks (15 tasks)

| ID | Task | Priority | Est. |
|----|------|----------|------|
| A.1.1 | Delete src/MLoop.AIAgent/ | 🔴 HIGH | 5min |
| A.1.2 | Remove from MLoop.sln | 🔴 HIGH | 5min |
| A.1.3 | Clean Directory.Packages.props | 🔴 HIGH | 10min |
| A.1.4 | Delete tests/MLoop.AIAgent.Tests/ | 🔴 HIGH | 5min |
| A.2.1 | Delete AgentCommand.cs | 🔴 HIGH | 5min |
| A.2.2 | Delete OrchestrateCommand.cs | 🔴 HIGH | 5min |
| A.2.3 | Remove TrainingMemoryCollector refs | 🟡 MED | 15min |
| A.2.4 | Clean DI registrations | 🟡 MED | 10min |
| A.3.1 | Delete agents/ folder | 🔴 HIGH | 5min |
| A.4.1 | Update README.md | 🟡 MED | 30min |
| A.4.2 | Update docs/AI-AGENTS.md | 🟡 MED | 20min |
| A.4.3 | Archive AI-AGENT-USAGE.md | 🟢 LOW | 10min |
| A.4.4 | Archive AI-AGENT-ARCHITECTURE.md | 🟢 LOW | 10min |
| A.5.1 | Build verification | 🔴 HIGH | 10min |
| A.5.2 | Test verification | 🔴 HIGH | 15min |

### Phase B Tasks (8 tasks) ✅ COMPLETE

| ID | Task | Priority | Status |
|----|------|----------|--------|
| B.1.1 | Create MLoop.DataStore.csproj | 🟡 MED | ✅ Done |
| B.1.2 | Create DataStore interfaces | 🟡 MED | ✅ Done |
| B.1.3 | Create DataStore models | 🟡 MED | ✅ (record in interfaces) |
| B.2.1 | Create MLoop.Ops.csproj | 🟡 MED | ✅ Done |
| B.2.2 | Create Ops interfaces | 🟡 MED | ✅ Done |
| B.2.3 | Create Ops models | 🟡 MED | ✅ (record in interfaces) |
| B.3.1 | Update MLoop.sln | 🟡 MED | ✅ Done |
| B.3.2 | Verify build | 🔴 HIGH | ✅ Done (389 tests pass) |

### Phase C Tasks (4 tasks) ✅ COMPLETE

| ID | Task | Priority | Status |
|----|------|----------|--------|
| C.1.1 | Create mloop-mcp repo | 🟡 MED | ✅ Done |
| C.1.2 | Add mcp/ submodule | 🟡 MED | ✅ Done |
| C.2.1 | Create mloop-studio repo | 🟡 MED | ✅ Done |
| C.2.2 | Add studio/ submodule | 🟡 MED | ✅ Done |

### Phase D Tasks (8 tasks)

| ID | Task | Priority | Est. |
|----|------|----------|------|
| D.1.1 | Update ROADMAP.md structure | 🟡 MED | 60min |
| D.1.2 | Mark deprecated phases | 🟡 MED | 20min |
| D.2.1 | Update ARCHITECTURE.md | 🟡 MED | 30min |
| D.3.1 | Create PHILOSOPHY.md | 🟡 MED | 30min |
| D.3.2 | Move philosophy content | 🟡 MED | 20min |
| D.4.1 | Create CLI-REFERENCE.md | 🟡 MED | 45min |
| D.5.1 | Update GUIDE.md | 🟡 MED | 30min |
| D.5.2 | Clean RECIPE-INDEX.md | 🟢 LOW | 15min |

---

## 5. Version Milestones

| Version | Focus | Key Deliverables | Status |
|---------|-------|------------------|--------|
| **v1.2.0-alpha** | Phase A | AIAgent 제거, 순수 CLI | ✅ Complete |
| **v1.2.0-beta** | Phase B | DataStore/Ops 스켈레톤 | ✅ Complete |
| **v1.2.0-rc** | Phase C | 서브모듈 설정 | ✅ Complete |
| **v1.2.0** | Phase D | 문서 완료 | ⏳ Pending |
| **v1.3.0** | DataStore | 예측 로깅, 피드백 수집 |
| **v1.4.0** | Ops | 재학습 트리거, 자동 프로모션 |
| **v2.0.0** | Studio | 웹 플랫폼 베타 |

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Build 실패 | Low | High | 단계별 빌드 검증 |
| 테스트 실패 | Medium | Medium | AI 테스트 분리 삭제 |
| 문서 불일치 | Medium | Low | 체크리스트 검증 |
| 서브모듈 충돌 | Low | Low | 별도 브랜치 작업 |

---

## 7. Rollback Plan

```bash
# Phase A 롤백 (필요시)
git checkout feature/phase8-polish-documentation -- src/MLoop.AIAgent/
git checkout feature/phase8-polish-documentation -- tests/MLoop.AIAgent.Tests/
git checkout feature/phase8-polish-documentation -- agents/
```

---

## 8. Success Criteria

| Criteria | Measurement |
|----------|-------------|
| Build 성공 | `dotnet build` 오류 없음 |
| 테스트 통과 | 모든 남은 테스트 통과 |
| CLI 동작 | `mloop train/predict/serve` 정상 |
| 문서 정합성 | 모든 링크 유효, AI 참조 제거 |
| 프로젝트 구조 | 6개 프로젝트 + 2개 서브모듈 |

---

**Last Updated**: January 12, 2026 (Phase C completed)
