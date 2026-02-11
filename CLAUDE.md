# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLoop is a CLI tool for building, running, and managing ML.NET models with filesystem-based MLOps. It replaces the discontinued ML.NET CLI with modern AutoML capabilities.

**Philosophy**: Excellent MLOps with Minimum Cost
- Convention over configuration
- AutoML-first, minimal coding required
- Filesystem-based state (Git-friendly, no databases)
- Multi-process casual design (each command runs independently and exits)

## Build Commands

```bash
# Restore and build
dotnet build MLoop.sln

# Run tests
dotnet test MLoop.sln

# Run specific test project
dotnet test tests/MLoop.Core.Tests
dotnet test tests/MLoop.API.Tests

# Pack CLI tool
dotnet pack tools/MLoop.CLI -c Release

# Install CLI locally for testing
dotnet tool install --global --add-source ./tools/MLoop.CLI/bin/Release mloop
```

## Project Architecture

```
src/                       # SDK packages (NuGet distribution)
├── MLoop.Extensibility/   # Extension interfaces (no dependencies)
│   └── IPreprocessingScript, IMLoopHook, IMLoopMetric
├── MLoop.Core/            # ML engine, AutoML wrapper, data loading
│   ├── AutoML/            # Training engine, configs
│   ├── Data/              # DataLoaderFactory, CsvHelper, EncodingDetector
│   ├── Preprocessing/     # Script execution, FilePrepper integration
│   └── Scripting/         # Roslyn compilation, DLL caching
├── MLoop.DataStore/       # Prediction logging, feedback collection (JSONL)
└── MLoop.Ops/             # Model comparison, retraining triggers

tools/                     # Executable distribution
├── MLoop.CLI/             # Command-line interface (dotnet tool)
│   └── Commands/          # TrainCommand, PredictCommand, etc.
└── MLoop.API/             # REST API server (ASP.NET Core Minimal API)
```

**Dependency Flow**:
```
MLoop.Extensibility (interfaces only)
        ↑
    MLoop.Core (ML.NET, FilePrepper)
        ↑
    ┌───┼───────────┐
    │   │           │
MLoop.CLI  MLoop.API  MLoop.DataStore  MLoop.Ops
```

## Key Technical Patterns

### Multi-Process Casual Design
Each CLI command runs as an independent process that exits when complete. No daemon, no shared state between executions. All state persists to filesystem.

### Central Package Management
Uses `Directory.Packages.props` for centralized NuGet version management. Do NOT add version numbers in individual `.csproj` files - only `PackageReference` without version.

### Encoding Detection
`EncodingDetector` handles CP949/EUC-KR auto-conversion to UTF-8 for Korean text support. Applied in CsvDataLoader, CsvHelper, and InfoCommand.

### Script Extensibility
`.mloop/scripts/` directory for user extensions:
- `hooks/`: Pre/post-train lifecycle hooks (`IMLoopHook`)
- `metrics/`: Custom optimization metrics (`IMLoopMetric`)
- `preprocess/`: Data preprocessing scripts (`IPreprocessingScript`)

Scripts are compiled with Roslyn and cached as DLLs. Zero overhead when not used (<1ms directory check).

## User Project Structure

When users run `mloop init my-project`:
```
my-project/
├── .mloop/                # Project marker
├── mloop.yaml             # User configuration
├── datasets/              # Training data (train.csv, predict.csv)
├── models/
│   └── {model-name}/      # Per-model namespace
│       ├── staging/       # Experiments (exp-001, exp-002, ...)
│       └── production/    # Promoted model
└── predictions/           # Output files
```

## Testing

- 521+ tests across all projects
- Unit tests: Core logic in isolation
- Integration tests: Real filesystem operations
- Use `FluentAssertions` for assertions
- Use `xunit` as test framework

## Key Dependencies

| Package | Purpose |
|---------|---------|
| Microsoft.ML 5.0.0 | ML.NET core |
| Microsoft.ML.AutoML 0.23.0 | AutoML engine |
| System.CommandLine 2.0.2 | CLI framework |
| Spectre.Console 0.54.0 | Rich CLI output |
| FilePrepper 0.4.9 | CSV preprocessing |
| Microsoft.CodeAnalysis.CSharp 5.0.0 | Roslyn script compilation |

## CLI Commands Reference

```bash
mloop init <project> --task <type>   # Initialize project
mloop train <data> <label> [options] # Train with AutoML
mloop predict [model] [data]         # Run predictions
mloop list [--name <model>]          # View experiments
mloop promote <exp-id>               # Promote to production
mloop serve [--port 5000]            # Start REST API
mloop docker                         # Generate Docker files
mloop info <data>                    # Dataset profiling
mloop logs                           # View prediction logs
mloop feedback add/list/metrics      # Feedback collection
mloop sample create/stats            # Data sampling
mloop trigger check                  # Retraining trigger evaluation
mloop prep run [options]             # Run preprocessing pipeline
mloop validate                       # Validate mloop.yaml config
mloop compare <exp1> <exp2>          # Compare experiments
mloop update                         # Self-update to latest version
```

## Important Conventions

1. **Error Handling**: Extension failures should never break core AutoML functionality. Use graceful degradation with warnings.

2. **Filesystem State**: All state as JSON/JSONL files. Human-readable, Git-friendly. No databases.

3. **Multi-Model Support**: Models are namespaced under `models/{name}/`. Default model name is "default".

4. **Experiment IDs**: Sequential format `exp-001`, `exp-002`. Generated atomically with file locking.

5. **Encoding**: Always handle Korean text (CP949/EUC-KR → UTF-8 conversion).

6. **Versioning Policy**: 메이저 버전(X.0.0) 변경 절대 금지. Claude가 임의로 메이저 버전을 올리지 않는다. 마이너(0.X.0) 및 패치/빌드(0.0.X) 버전만 변경 가능. 메이저 버전 변경은 팀 승인 후 수동 작업으로만 진행한다.

## Dogfooding: iyulab 패키지 피드백

MLoop와 함께 사용되는 다음 패키지들은 iyulab 팀이 관리합니다:
- **ironhive-cli**: AI 에이전트 CLI (MCP 클라이언트)
- **mloop**: ML.NET AutoML CLI (이 프로젝트)
- **fileprepper**: CSV 전처리 라이브러리

**패키지별 처리 방식**:

| 패키지 | 처리 방식 |
|--------|-----------|
| **mloop** (현재 프로젝트) | 즉시 수정 가능하면 수정, 큰 작업은 이슈 작성 후 후속 페이즈에서 진행 |
| **ironhive-cli** | 이슈 작성하여 팀에 전달 |
| **fileprepper** | 이슈 작성하여 팀에 전달 |

**공통 원칙**:
1. 사용 중 발견되는 버그, 한계, 개선점을 즉시 기록
2. 사소해 보여도 기록 (작은 문제가 큰 문제로 발전)
3. 비판적 시각 유지, 개선 가능성 적극 탐색
4. 이슈 파일 위치: `claudedocs/issues/ISSUE-<YYYYMMDD>-<slug>.md`

**이슈 초안 형식**:
```markdown
# [제목]
**Status**: Open
**Package**: [ironhive-cli | mloop | fileprepper]
**Severity**: [Critical | High | Medium | Low]

## 문제
[문제 설명]

## 재현 방법
[재현 단계]

## 예상 동작
[기대 결과]

## 현재 동작
[실제 결과]

## 제안
[개선 방안]
```
