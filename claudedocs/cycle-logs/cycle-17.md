# Cycle 17: Error Handling Integration

## Date
2026-02-11

## Scope
- 6개 주요 커맨드에 ErrorSuggestions.DisplayError 통합
- ErrorSuggestions에 promote, validate, preprocessing 컨텍스트 추가
- ErrorSuggestions 단위 테스트 8개 추가

## Philosophy Alignment
| Dimension | Score |
|-----------|-------|
| Core Mission Fit | 5/5 |
| Scope Boundaries | 5/5 |
| Architecture Patterns | 5/5 |
| Dependency Direction | 5/5 |

## Research Summary
- 기존: TrainCommand, PredictCommand만 ErrorSuggestions 사용
- 나머지 커맨드: 인라인 AnsiConsole.Markup("[red]Error:[/]") 패턴
- 일관된 에러 경험은 CLI 도구의 핵심 품질 지표
- 컨텍스트별 맞춤 제안이 사용자 문제 해결 시간 단축

## Implementation
- **Commands updated** (6 files):
  - PromoteCommand.cs — ErrorSuggestions.DisplayError(ex, "promote")
  - ListCommand.cs — ErrorSuggestions.DisplayError(ex, "list")
  - InfoCommand.cs — ErrorSuggestions.DisplayError(ex, "info")
  - PrepRunCommand.cs — ErrorSuggestions.DisplayError(ex, "preprocessing")
  - ValidateCommand.cs — ErrorSuggestions.DisplayError(ex, "validate")
  - CompareCommand.cs — ErrorSuggestions.DisplayError(ex, "compare")
- **ErrorSuggestions.cs** (MODIFIED):
  - promote 컨텍스트: experiment 목록 확인 제안
  - validate 컨텍스트: YAML 문법 확인 제안
  - preprocessing 컨텍스트: dry-run, CSV 형식 확인 제안
- **ErrorSuggestionsTests.cs** (NEW): 8개 테스트

## Test Results
- Pass: 521 / 521 (0 failed, 3 skipped) [+8 new tests]
- Build: 0 warnings, 0 errors

## Evaluation
| Criterion | Score | Notes |
|-----------|-------|-------|
| Correctness | 9/10 | 모든 커맨드 일관된 에러 핸들링 |
| Architecture | 10/10 | 중앙화된 ErrorSuggestions 패턴 일관 적용 |
| Philosophy Alignment | 10/10 | Convention: 일관된 사용자 경험 |
| Test Quality | 9/10 | 8개 단위 테스트로 핵심 시나리오 커버 |
| Documentation | 8/10 | |
| Code Quality | 9/10 | |
| **Average** | **9.2/10** | |

## Issues & Improvements
| # | Severity | Description | Action |
|---|----------|-------------|--------|
| I-01 | L | 나머지 커맨드 (Docker, Sample, Feedback 등)도 ErrorSuggestions 미적용 | 후속 정리 가능 |

## Pending Human Decisions
None

## Next Cycle Recommendation
성능 최적화 (Cycle 18) — 대용량 CSV 처리 또는 빌드 최적화
