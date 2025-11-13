# MLoop

A modern CLI tool for building, running, and managing ML.NET models with filesystem-based MLOps.

## Overview

MLoop fills the gap left by the discontinued ML.NET CLI, providing a simple yet powerful command-line interface for the entire machine learning lifecycle - from training to prediction to deployment.

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

- **[User Guide](docs/GUIDE.md)** - Complete usage guide with concurrent job management
- **[AI Agents](docs/AI-AGENTS.md)** - Multi-provider LLM agents for interactive ML assistance
- **[Architecture](docs/ARCHITECTURE.md)** - Technical documentation
- **[Examples](examples/)** - Sample workflows
- **[Job Management Scripts](examples/scripts/)** - Sequential and parallel execution tools

## Why MLoop?

With Microsoft discontinuing ML.NET CLI and Model Builder updates, MLoop provides:

- ✅ **Active Development** - Modern tooling with latest ML.NET 4.0
- ✅ **Production Ready** - From prototyping to deployment
- ✅ **Git Friendly** - All state as files, no database
- ✅ **Extensible** - Hooks and custom metrics
- ✅ **Fast** - FilePrepper integration (20x speedup)

## Requirements

- .NET 9.0+
- ML.NET 5.0.0+

## License

MIT License - Built on [ML.NET](https://github.com/dotnet/machinelearning) by Microsoft.