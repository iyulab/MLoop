# Cycle 19: CI/CD Workflow Cleanup

## Date
2026-02-11

## Scope
- CHANGELOG.md 생성 (release workflow가 참조하지만 파일 부재)
- CI workflow에 concurrency group 추가 (중복 실행 방지)
- Release workflow 정리 (주석 제거, heredoc 버그 수정, 불필요한 input 제거)
- NuGet cache key에서 존재하지 않는 packages.lock.json 참조 제거

## Philosophy Alignment
| Dimension | Score |
|-----------|-------|
| Core Mission Fit | 4/5 |
| Scope Boundaries | 5/5 |
| Architecture Patterns | 5/5 |
| Dependency Direction | 5/5 |

## Research Summary
- ci.yml: cache key에 `packages.lock.json` 참조하지만 실제 파일 없음 → `Directory.Packages.props` + `*.csproj`로 변경
- release.yml: heredoc 내부 8칸 들여쓰기가 markdown에 포함되어 code block으로 렌더링되는 버그
- release.yml: NuGet publishing 주석 블록 + 관련 workflow_dispatch input이 불필요
- concurrency group: 같은 브랜치에 연속 push 시 이전 CI 자동 취소로 비용 절감

## Implementation
- **CHANGELOG.md** (NEW):
  - Keep a Changelog 형식 준수
  - v0.1.0 ~ v0.5.1-alpha + Unreleased 섹션
  - 각 버전의 Added/Fixed/Changed 분류
- **ci.yml** (MODIFIED):
  - `concurrency: group: ci-${{ github.ref }}, cancel-in-progress: true` 추가
  - Cache key: `packages.lock.json` → `**/*.csproj` 로 변경
- **release.yml** (MODIFIED):
  - `concurrency: group: release-${{ github.ref }}, cancel-in-progress: false` 추가
  - NuGet publishing 주석 블록 전체 제거
  - `workflow_dispatch.inputs.publish_nuget` 제거
  - heredoc 렌더링 버그 수정 (sed로 leading spaces strip)

## Test Results
- Pass: 521 / 521 (0 failed, 3 skipped)
- Build: 0 warnings, 0 errors

## Evaluation
| Criterion | Score | Notes |
|-----------|-------|-------|
| Correctness | 9/10 | YAML 문법 검증은 CI에서만 가능하지만 구조적으로 올바름 |
| Architecture | 9/10 | CI/CD best practices 준수 |
| Philosophy Alignment | 9/10 | 최소 비용 원칙에 부합하는 concurrency 설정 |
| Test Quality | 7/10 | CI workflow 자체 테스트는 불가, 코드 테스트는 전체 통과 |
| Documentation | 9/10 | CHANGELOG.md가 릴리스 프로세스 문서화 |
| Code Quality | 9/10 | 불필요한 주석 제거로 가독성 향상 |
| **Average** | **8.7/10** | |

## Issues & Improvements
| # | Severity | Description | Action |
|---|----------|-------------|--------|
| I-01 | L | CHANGELOG.md의 이전 버전 날짜가 추정치 | 실제 태그 날짜와 동기화 필요 |
| I-02 | L | lint job의 continue-on-error: true — 포맷 검사가 항상 통과 | 코드 포맷 정리 후 해제 고려 |

## Pending Human Decisions
None

## Next Cycle Recommendation
최종 점검 및 릴리스 준비 (Cycle 20) — 전체 코드 리뷰, 테스트 커버리지 최종 확인, 버전 태그 준비
