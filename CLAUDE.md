# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build
dotnet build MLoop.sln --configuration Release

# Run all tests (includes heavy tests - requires API keys for LLM tests)
dotnet test MLoop.sln --configuration Release

# Run CI tests only (excludes LLM, Slow, E2E, Database tests)
dotnet test MLoop.sln --filter "Category!=LLM&Category!=Slow&Category!=E2E&Category!=Database"

# Run specific test category
dotnet test MLoop.sln --filter "Category=LLM"

# Run single test
dotnet test MLoop.sln --filter "FullyQualifiedName~YourTestClassName.YourTestMethodName"

# Format check
dotnet format MLoop.sln --verify-no-changes

# Check vulnerable packages
dotnet list MLoop.sln package --vulnerable --include-transitive
```

## Architecture Overview

MLoop is a CLI-based ML.NET MLOps platform following **Multi-Process Casual** design - every command runs as an independent process that starts, executes, and exits cleanly. No daemon, no background service.

### Solution Structure

```
src/
├── MLoop.CLI          # Main CLI tool (System.CommandLine)
├── MLoop.Core         # ML.NET integration, data loading, contracts
├── MLoop.API          # REST API for model serving (mloop serve)
├── MLoop.AIAgent      # Multi-provider LLM agent integration (Ironbees)
└── MLoop.Extensibility # C# scripting and preprocessing hooks
```

### Key Architectural Patterns

**Filesystem-First MLOps**: All state persisted as files, no database required.
```
models/{modelName}/
├── staging/exp-XXX/    # Training experiments
└── production/         # Promoted model
```

**Multi-Model Support**: Each model name gets isolated experiment namespace.
- `IExperimentStore` - Per-model experiment management
- `IModelRegistry` - Per-model production slot management
- `IModelNameResolver` - Model name validation and resolution

**CLI Infrastructure** (`src/MLoop.CLI/Infrastructure/`):
- `Configuration/` - `MLoopConfig`, `ConfigLoader`, YAML-based config
- `FileSystem/` - `ExperimentStore`, `ModelRegistry`, `ProjectDiscovery`
- `ML/` - `TrainingEngine`, `PredictionEngine`, `EvaluationEngine`

### Core Interfaces

```
IExperimentStore     # Generate IDs, save/load experiments, list by model
IModelRegistry       # Promote, get production, auto-promote logic
ITrainingEngine      # ML.NET AutoML training orchestration
IDatasetDiscovery    # Locate train.csv, predict.csv by convention
```

### Configuration Format (mloop.yaml)

```yaml
project: my-project
models:
  default:           # Model name
    task: regression
    label: Price
    training:
      time_limit_seconds: 60
data:
  train: datasets/train.csv
```

## Test Categories

Use `[Trait("Category", "X")]` for test categorization:
- **LLM** - Requires API keys (OPENAI_API_KEY or AZURE_OPENAI_API_KEY)
- **Slow** - Long-running tests
- **E2E** - End-to-end integration tests
- **Database** - Database-dependent tests

CI excludes all heavy categories by default.

## Central Package Management

This project uses `Directory.Packages.props` for centralized version control. When adding packages:
1. Add `<PackageVersion>` to `Directory.Packages.props`
2. Reference in `.csproj` without version: `<PackageReference Include="PackageName" />`

## Key Technologies

- .NET 10.0, ML.NET 5.0, Microsoft.ML.AutoML 0.23.0
- System.CommandLine for CLI, YamlDotNet for config
- Spectre.Console for rich terminal output
- Microsoft.Extensions.AI for multi-provider LLM support
- Ironbees SDK for AI agent orchestration
