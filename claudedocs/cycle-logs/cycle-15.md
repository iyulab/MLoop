# Cycle 15: mloop predict — Workflow Enhancement

## Date
2026-02-11

## Scope
- 예측 결과 미리보기 (상위 5행 테이블)
- 분류 예측 분포 표시 (PredictedLabel 값 분포 바차트)
- 예측 완료 후 사용자에게 즉시 결과 맥락 제공

## Philosophy Alignment
| Dimension | Score |
|-----------|-------|
| Core Mission Fit | 5/5 |
| Scope Boundaries | 5/5 |
| Architecture Patterns | 5/5 |
| Dependency Direction | 5/5 |

## Research Summary
- 기존 PredictCommand: 예측 행 수와 출력 경로만 표시
- 사용자가 결과를 확인하려면 CSV 파일을 열어야 함
- CLI 도구로서 즉시 결과 맥락을 제공하는 것이 UX 핵심
- 분류 결과 분포는 모델 행동 파악에 핵심 정보

## Implementation
- `tools/MLoop.CLI/Commands/PredictCommand.cs` (MODIFIED):
  - `DisplayPredictionPreview()`: 출력 CSV 상위 5행을 Spectre.Console 테이블로 표시
  - `DisplayPredictionDistribution()`: PredictedLabel 컬럼의 값 분포를 바차트로 표시
  - 2-50개 유니크 값 범위에서만 분포 표시 (회귀/연속값 제외)
  - 양쪽 메서드 모두 non-critical — 에러 시 graceful skip

## Test Results
- Pass: 508 / 508 (0 failed, 3 skipped)
- Build: 0 warnings, 0 errors

## Evaluation
| Criterion | Score | Notes |
|-----------|-------|-------|
| Correctness | 9/10 | Preview + Distribution 정확, edge case 처리 |
| Architecture | 9/10 | 기존 패턴 유지 |
| Philosophy Alignment | 10/10 | Convention: 결과를 즉시 확인 가능 |
| Test Quality | 6/10 | CLI 출력 테스트 부재 |
| Documentation | 8/10 | |
| Code Quality | 9/10 | |
| **Average** | **8.5/10** | |

## Issues & Improvements
| # | Severity | Description | Action |
|---|----------|-------------|--------|
| I-01 | L | 큰 파일에서 distribution 계산 시 전체 파일 읽기 | 샘플링 고려 |
| I-02 | L | 회귀 결과의 통계 요약 (min/max/mean) 미표시 | 후속 개선 |

## Pending Human Decisions
None

## Next Cycle Recommendation
Phase D 시작: Pipeline 테스트 강화 (Cycle 16)
