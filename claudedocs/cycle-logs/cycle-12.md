# Cycle 12: mloop predict — YAML Prep Pipeline Integration

## Date
2026-02-11

## Scope
- `mloop predict` 실행 시 YAML prep 파이프라인 자동 적용
- 트레이닝과 동일한 전처리가 예측 데이터에도 적용되도록 보장
- ConfigLoader → ModelDefinition.Prep → DataPipelineExecutor 재사용

## Philosophy Alignment
| Dimension | Score |
|-----------|-------|
| Core Mission Fit | 5/5 |
| Scope Boundaries | 5/5 |
| Architecture Patterns | 5/5 |
| Dependency Direction | 5/5 |

## Research Summary
- `mloop train`은 TrainCommand에서 prep 파이프라인을 적용
- `mloop predict`는 prep 적용 없이 raw 데이터로 예측 시도
- 트레이닝/예측 간 전처리 불일치는 모델 정확도 저하의 주요 원인
- DataPipelineExecutor와 ConfigLoader를 재사용하여 일관성 보장

## Implementation
- `tools/MLoop.CLI/Commands/PredictCommand.cs` (MODIFIED):
  - `using MLoop.Core.Preprocessing;` 추가
  - `using MLoop.Extensibility.Preprocessing;` 추가
  - Schema validation 후, prediction 전에 prep 파이프라인 실행 블록 삽입
  - ConfigLoader로 mloop.yaml 로드 → ModelDefinition.Prep 확인 → DataPipelineExecutor 실행
  - 전처리된 파일을 `{name}_prep.csv`로 저장하고 resolvedDataFile 갱신
  - `PredictPrepLogger` 내부 클래스 추가 (ILogger 구현)
  - InvalidOperationException catch로 프로젝트 외부 실행 시 graceful degradation

## Test Results
- Pass: 467 / 467 (0 failed, 3 skipped)
- Build: 0 warnings, 0 errors
- PredictCommand는 CLI 통합 명령 — end-to-end 검증 위주

## Evaluation
| Criterion | Score | Notes |
|-----------|-------|-------|
| Correctness | 9/10 | Train/predict 간 prep 일관성 보장 |
| Architecture | 9/10 | 기존 컴포넌트 재사용, 새 코드 최소화 |
| Philosophy Alignment | 10/10 | Convention: 동일 prep이 train/predict 모두에 적용 |
| Test Quality | 6/10 | CLI 통합 테스트 부재 (파일시스템 + model 의존) |
| Documentation | 8/10 | 사용자에게 prep 적용 상태 표시 |
| Code Quality | 9/10 | |
| **Average** | **8.5/10** | |

## Issues & Improvements
| # | Severity | Description | Action |
|---|----------|-------------|--------|
| I-01 | M | PredictCommand prep 통합 테스트 부재 | End-to-end 테스트 추가 필요 |
| I-02 | L | prep 후 임시 파일(_prep.csv) 정리 로직 없음 | 후속 개선 |
| I-03 | L | ConfigLoader를 PredictCommand에서 직접 생성 — DI 패턴 미적용 | CLI 특성상 허용 |

## Pending Human Decisions
None

## Next Cycle Recommendation
mloop list 출력 개선 또는 mloop train 진단 출력 강화
