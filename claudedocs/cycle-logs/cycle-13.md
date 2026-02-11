# Cycle 13: mloop list — Output Format Enhancement

## Date
2026-02-11

## Scope
- ExperimentSummary에 BestTrainer, MetricName, TrainingTimeSeconds 필드 추가
- ListCommand 테이블에 Trainer 컬럼 추가
- Metric 표시에 메트릭 이름 병기
- 상대 시간 표시 ("2h ago", "3d ago" 등)
- 기존 인덱스 파일과 하위 호환 유지 (nullable 필드)

## Philosophy Alignment
| Dimension | Score |
|-----------|-------|
| Core Mission Fit | 5/5 |
| Scope Boundaries | 5/5 |
| Architecture Patterns | 5/5 |
| Dependency Direction | 5/5 |

## Research Summary
- ExperimentData.Result에 BestTrainer, TrainingTimeSeconds 존재
- ExperimentData.Config.Metric에 메트릭 이름 존재
- 이 정보들이 ExperimentSummary (인덱스)에 전파되지 않아 list에서 표시 불가
- 기존 인덱스 파일은 새 필드 없음 → null로 역직렬화 → "-" 표시

## Implementation
- `tools/MLoop.CLI/Infrastructure/FileSystem/IExperimentStore.cs` (MODIFIED):
  - ExperimentSummary에 BestTrainer, MetricName, TrainingTimeSeconds 추가
- `tools/MLoop.CLI/Infrastructure/FileSystem/ExperimentStore.cs` (MODIFIED):
  - UpdateIndexWithExperimentAsync에서 새 필드 채움
- `tools/MLoop.CLI/Commands/ListCommand.cs` (MODIFIED):
  - Trainer 컬럼 추가
  - Metric 표시에 메트릭 이름 병기 (예: "0.9200 (RSquared)")
  - 상대 시간 표시 (FormatRelativeTime 헬퍼)
  - FormatMetric 헬퍼 추가
- `tests/MLoop.Tests/Infrastructure/FileSystem/ExperimentStoreTests.cs` (MODIFIED):
  - ListAsync_WithResult_IncludesTrainerAndMetricName 테스트 추가

## Test Results
- Pass: 508 / 508 (0 failed, 3 skipped) [+1 new test]
- Build: 0 warnings, 0 errors

## Evaluation
| Criterion | Score | Notes |
|-----------|-------|-------|
| Correctness | 9/10 | 기존 인덱스 하위 호환, 새 정보 표시 |
| Architecture | 9/10 | 인덱스 데이터 모델 자연 확장 |
| Philosophy Alignment | 10/10 | Convention: 실험 결과 한눈에 파악 |
| Test Quality | 8/10 | 새 필드 저장/조회 테스트 추가 |
| Documentation | 8/10 | |
| Code Quality | 9/10 | |
| **Average** | **8.8/10** | |

## Issues & Improvements
| # | Severity | Description | Action |
|---|----------|-------------|--------|
| I-01 | L | FormatRelativeTime이 UTC/Local 혼합 시 부정확할 수 있음 | 테스트 추가 권장 |
| I-02 | L | 기존 인덱스 파일의 새 필드는 null — 마이그레이션 없음 | 재훈련 시 자동 채워짐 |

## Pending Human Decisions
None

## Next Cycle Recommendation
mloop train 진단 출력 개선 (훈련 진행 상황, 알고리즘 선택 과정 표시)
