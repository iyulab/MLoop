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

**3. AI Agent Assistance for Expert-Level Results**
- **Interactive ML guidance**: Multi-provider LLM agents (OpenAI, Anthropic, Google, etc.)
- **Intelligent optimization**: AI suggests hyperparameters, feature engineering, and model selection
- **No ML expertise required**: Agents guide you through the entire workflow
- **Learn as you go**: AI explains decisions and teaches ML concepts

**4. Extensibility Through Dynamic Scripting**
- **Optional customization**: Start simple, add complexity only when needed
- **Code-based hooks**: Inject custom logic at any pipeline stage (pre-train, post-train, etc.)
- **Custom metrics**: Define business-specific optimization objectives
- **C# scripting**: Full IDE support with IntelliSense and debugging
- **Zero overhead**: <1ms performance impact when extensions aren't used

**5. Minimum Cost, Maximum Value**
- **Development cost**: 3-command workflow vs traditional multi-week ML projects
- **Knowledge cost**: No ML degree required - AI agents provide expertise
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
- **AI Agents**: Interactive ML assistance with multi-provider LLM support
- **Smart Predictions**: Production model auto-discovery and batch processing
- **Filesystem MLOps**: Git-friendly experiment tracking (no database required)
- **Fast Preprocessing**: Integrated FilePrepper (20x faster than pandas)
- **Extensibility**: Code-based hooks and custom metrics
- **Zero Config**: Works immediately with intelligent defaults

## Quick Start

### Installation

```bash
dotnet tool install -g mloop
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
```

**Full documentation**: [docs/GUIDE.md](docs/GUIDE.md)

## Project Structure

MLoop uses **Convention over Configuration** - intelligent defaults that work out of the box.

```
my-ml-project/
├── .mloop/           # Project marker (like .git)
├── datasets/         # Training data → train.csv, predict.csv
├── models/
│   ├── staging/      # All experiments (exp-001, exp-002, ...)
│   └── production/   # Promoted model (symlink)
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

## Documentation

### Getting Started
- **[User Guide](docs/GUIDE.md)** - Complete usage guide with concurrent job management
- **[AI Agents](docs/AI-AGENTS.md)** - Multi-provider LLM agents for interactive ML assistance
- **[Examples](examples/)** - Sample workflows
- **[Job Management Scripts](examples/scripts/)** - Sequential and parallel execution tools

### Project Philosophy & Roadmap
- **[Philosophy & Design](README.md#philosophy-excellent-mlops-with-minimum-cost)** - Core mission and design principles
- **[Roadmap](ROADMAP.md)** - Feature roadmap aligned with mission (v0.1 → v1.0)
- **[Task Breakdown](docs/TASKS.md)** - Detailed, actionable tasks for current sprint

### Technical Documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical decisions
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines and development workflow

## Why MLoop?

With Microsoft discontinuing ML.NET CLI and Model Builder updates, MLoop provides:

- ✅ **Active Development** - Modern tooling with latest ML.NET 4.0
- ✅ **Production Ready** - From prototyping to deployment
- ✅ **Git Friendly** - All state as files, no database
- ✅ **Extensible** - Hooks and custom metrics
- ✅ **Fast** - FilePrepper integration (20x speedup)

## Requirements

- .NET 10.0+
- ML.NET 5.0.0+

## License

MIT License - Built on [ML.NET](https://github.com/dotnet/machinelearning) by Microsoft.