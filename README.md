# MLoop

A modern CLI tool for building, running, and managing ML.NET models with filesystem-based MLOps.

## Overview

MLoop fills the gap left by the discontinued ML.NET CLI, providing a simple yet powerful command-line interface for the entire machine learning lifecycle - from training to prediction to deployment.

## Philosophy: Excellent MLOps with Minimum Cost

**Core Mission**: Enable anyone to achieve production-quality ML models with minimal coding and ML expertise, while maintaining the flexibility for advanced customization.

### Design Principles

**1. Convention Over Configuration**
- **Filesystem-based contracts**: Just drop your CSV in `datasets/`, train, and predict
- **Zero configuration required**: Intelligent defaults make it work immediately
- **Git-friendly MLOps**: All experiment metadata tracked as files, not databases
- **No daemon, no servers**: Each command runs independently and exits cleanly

**2. AutoML-First, Minimal Coding**
- **One command to train**: `mloop train datasets/train.csv price --time 60`
- **Automatic model selection**: ML.NET AutoML finds the best algorithm for your data
- **No feature engineering required**: Optional FilePrepper integration for complex preprocessing
- **Production-ready in minutes**: From CSV to deployed model in 3 commands

**3. Extensibility Through Dynamic Scripting**
- **Optional customization**: Start simple, add complexity only when needed
- **Code-based hooks**: Inject custom logic at any pipeline stage (pre-train, post-train, etc.)
- **Custom metrics**: Define business-specific optimization objectives
- **C# scripting**: Full IDE support with IntelliSense and debugging
- **Zero overhead**: <1ms performance impact when extensions aren't used

**4. Minimum Cost, Maximum Value**
- **Development cost**: 3-command workflow vs traditional multi-week ML projects
- **Knowledge cost**: AutoML handles algorithm selection automatically
- **Operational cost**: Filesystem-based MLOps eliminates infrastructure complexity
- **Time cost**: From CSV to production model in minutes, not days

### What Makes MLoop Different

| Traditional ML Workflow | MLoop Workflow |
|------------------------|----------------|
| **Weeks**: Data prep → Feature engineering → Model selection → Training → Deployment | **Minutes**: `init` → `train` → `predict` |
| **Requires**: Python, Jupyter, scikit-learn, pandas, Docker, Kubernetes | **Requires**: .NET CLI only |
| **Expertise**: ML engineering, DevOps, data science | **Expertise**: Basic CSV understanding |
| **Cost**: Engineering team, infrastructure, training | **Cost**: Developer time only |
| **Result**: Custom solution, high maintenance | **Result**: Production-ready, git-trackable MLOps |

### Key Features

- **AutoML Training**: Automatic model selection with ML.NET AutoML
- **Multi-Model Projects**: Manage multiple models (churn, revenue, etc.) in one project
- **Smart Predictions**: Production model auto-discovery and batch processing
- **Filesystem MLOps**: Git-friendly experiment tracking (no database required)
- **Fast Preprocessing**: Integrated FilePrepper (20x faster than pandas)
- **Extensibility**: Code-based hooks and custom metrics
- **Zero Config**: Works immediately with intelligent defaults
- **Multi-CSV Support**: Auto-merge same-schema files with `--data` or `--auto-merge`
- **Encoding Detection**: Automatic CP949/EUC-KR to UTF-8 conversion for Korean text
- **Docker Deployment**: Generate production-ready containerization with `mloop docker`
- **Self-Update**: Update to latest version with `mloop update`

## Quick Start

### Installation

**Standalone Binary** (Recommended):

Download from [GitHub Releases](https://github.com/iyulab/MLoop/releases) and place in your PATH.

| Platform | Binary |
|----------|--------|
| Windows x64 | `mloop-win-x64.exe` |
| Linux x64 | `mloop-linux-x64` |
| macOS x64 | `mloop-osx-x64` |

```bash
# Self-update to latest version anytime
mloop update
```

### 60-Second Workflow

```bash
# 1. Initialize project
mloop init my-ml-project --task regression
cd my-ml-project

# 2. Add training data
cp ~/data/train.csv datasets/train.csv

# 3. Train model (60 seconds)
mloop train datasets/train.csv price --time 60

# 4. Make predictions
mloop predict
# ✅ Output: predictions/predictions-TIMESTAMP.csv
```

That's it! Your model is trained and ready to use.

## Core Commands

```bash
mloop init <project> --task <type>    # Initialize ML project
mloop train <data> <label> [options]  # Train with AutoML
mloop predict [model] [data]          # Run predictions
mloop list                             # View experiments
mloop promote <exp-id>                # Promote to production
mloop evaluate <model> <test> <label> # Evaluate performance
mloop info <data>                      # Dataset profiling with encoding detection
mloop validate                         # Validate mloop.yaml configuration
mloop prep run [options]               # Run preprocessing pipeline
mloop compare <exp1> <exp2>            # Compare experiment metrics
mloop serve                            # Start REST API server
mloop docker                           # Generate Docker deployment files
mloop update                           # Self-update to latest version
```

### Advanced Data Handling

```bash
# Multi-file training - auto-merge same-schema CSVs
mloop train --data file1.csv file2.csv --label Target --task regression

# Auto-discovery merge from datasets/ folder
mloop train --auto-merge --label Target --task regression

# Handle Korean/Chinese encoded files (CP949, EUC-KR auto-converted)
mloop train korean_data.csv label --task regression
# Output: [Info] Converted CP949 → UTF-8

# Drop missing label values (default for classification)
mloop train data.csv label --task binary-classification --drop-missing-labels
```

**Full documentation**: [docs/GUIDE.md](docs/GUIDE.md)

## Project Structure

MLoop uses **Convention over Configuration** - intelligent defaults that work out of the box.

```
my-ml-project/
├── .mloop/           # Project marker (like .git)
├── mloop.yaml        # Project configuration
├── datasets/         # Training data → train.csv, predict.csv
├── models/
│   └── {model-name}/ # Per-model namespace (default, churn, revenue, etc.)
│       ├── staging/      # Experiments (exp-001, exp-002, ...)
│       └── production/   # Promoted model
└── predictions/      # Outputs (timestamped CSVs)
```

### Workflow Example

```bash
# Train multiple experiments
mloop train datasets/train.csv price --time 60   # exp-001
mloop train datasets/train.csv price --time 120  # exp-002 (better)
mloop train datasets/train.csv price --time 180  # exp-003 (best!)

# Review experiments
mloop list

# Output:
# ┌─────────┬───────────┬─────────────┬──────────────┐
# │   ID    │  Status   │ Best Metric │    Stage     │
# ├─────────┼───────────┼─────────────┼──────────────┤
# │ exp-003 │ Completed │    0.9523   │ ★ Production │
# │ exp-002 │ Completed │    0.9401   │   Staging    │
# │ exp-001 │ Completed │    0.9278   │   Staging    │
# └─────────┴───────────┴─────────────┴──────────────┘

# Use production model
mloop predict  # Auto-uses exp-003
```

### Multi-Model Support

Manage multiple models within a single project - perfect for complex ML systems.

```bash
# Train different models for different targets
mloop train --name churn --label Churned --task binary-classification
mloop train --name revenue --label Revenue --task regression
mloop train --name ltv --label LifetimeValue --task regression

# Each model has independent experiments
mloop list --name churn     # Shows churn experiments only
mloop list --name revenue   # Shows revenue experiments only
mloop list                  # Shows all experiments across models

# Promote and predict per model
mloop promote exp-001 --name churn
mloop predict data.csv --name churn

# Serve multiple models via API
mloop serve
# GET /predict?name=churn
# GET /predict?name=revenue
```

**Directory Structure with Multiple Models:**
```
models/
├── churn/
│   ├── staging/exp-001/
│   └── production/
├── revenue/
│   ├── staging/exp-001/
│   └── production/
└── default/
    └── staging/exp-001/
```

## Documentation

### Getting Started
- **[User Guide](docs/GUIDE.md)** - Complete usage guide with concurrent job management
- **[AI Agents](docs/AI-AGENTS.md)** - Multi-provider LLM agents for interactive ML assistance
- **[Examples](examples/)** - Sample workflows
- **[Job Management Scripts](examples/scripts/)** - Sequential and parallel execution tools

### Project Philosophy & Roadmap
- **[Philosophy & Design](README.md#philosophy-excellent-mlops-with-minimum-cost)** - Core mission and design principles
- **[Roadmap](ROADMAP.md)** - Feature roadmap aligned with mission

### Technical Documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical decisions
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines and development workflow

## Why MLoop?

With Microsoft discontinuing ML.NET CLI and Model Builder updates, MLoop provides:

- ✅ **Active Development** - Modern tooling with latest ML.NET 5.0
- ✅ **Production Ready** - From prototyping to deployment
- ✅ **Git Friendly** - All state as files, no database
- ✅ **Extensible** - Hooks and custom metrics
- ✅ **Fast** - FilePrepper integration (20x speedup)

## Requirements

- .NET 10.0+
- ML.NET 5.0.0+

## License

MIT License - Built on [ML.NET](https://github.com/dotnet/machinelearning) by Microsoft.