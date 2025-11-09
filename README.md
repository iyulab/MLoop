# MLoop

A modern CLI tool for building, running, and managing ML.NET models with filesystem-based MLOps.

## Overview

MLoop fills the gap left by the discontinued ML.NET CLI, providing a simple yet powerful command-line interface for the entire machine learning lifecycle - from training to prediction to deployment.

## Key Features

- **Build Models**: Train ML.NET models using AutoML or custom configurations
- **Run Predictions**: Execute batch and single predictions with trained models
- **Filesystem-based MLOps**: Git-friendly experiment tracking and model versioning
- **Data Preprocessing**: Integrated FilePrepper for ML-focused data cleaning and normalization
- **Extensibility**: Code-based hooks and custom metrics for AutoML customization
- **Pipeline Automation**: Define and execute reproducible ML workflows
- **Model Registry**: Organize and manage model versions
- **API Serving**: Deploy models as REST APIs

## Installation

```bash
dotnet tool install -g mloop
```

## Quick Start

```bash
# 1. Initialize a new ML project
mloop init

# 2. Add your training data
# Place train.csv in datasets/ folder
# (Optional: add validation.csv, test.csv, predict.csv)

# 3. Train a model with AutoML
mloop train datasets/train.csv price --time 60

# 4. List all experiments
mloop list
# Shows: ID, Status, Metrics, Production status

# 5. Promote an experiment to production
mloop promote exp-001
# exp-001 is now the production model

# 6. Make predictions with production model
mloop predict
# Automatically uses: production model + datasets/predict.csv
# Outputs to: predictions/predictions-TIMESTAMP.csv

# Or specify custom paths
mloop predict models/staging/exp-001/model.zip datasets/predict.csv --output results.csv
```

## Core Commands

### Available Now
- **`init`** - Initialize a new ML project with filesystem structure
- **`train`** - Train models using ML.NET AutoML with automatic experiment tracking
- **`predict`** - Run predictions with auto-discovery of production models
- **`list`** - List all experiments with status and metrics
- **`promote`** - Manually promote experiments to production or staging
- **`info`** - Display dataset profiling information
- **`evaluate`** - Evaluate model performance on test data
- **`validate`** - Validate extensibility scripts (hooks and metrics)
- **`extensions`** - List all discovered hooks and metrics

### Coming Soon
- `serve` - Deploy models as REST APIs
- `pipeline` - Define and execute ML workflows

## Why MLoop?

With Microsoft discontinuing ML.NET CLI and Model Builder updates, MLoop provides:

- **Active Development**: Modern tooling with latest ML.NET features
- **Production Ready**: From experimentation to deployment
- **Version Control Friendly**: All configurations in YAML/JSON
- **Reproducible**: Complete experiment tracking and lineage
- **Extensible**: Plugin architecture for custom components

## Philosophy

MLoop follows a filesystem-first approach where experiments, models, and configurations are stored as files. This makes everything:

- Versionable with Git
- Shareable across teams
- Auditable and reproducible
- Independent of external services

## MLOps Workflow

MLoop implements **Convention over Configuration** - intelligent defaults that work out of the box.

### Project Structure

```
my-ml-project/
├── .mloop/                    # Project marker (like .git)
│   └── scripts/              # Extensibility scripts
│       ├── hooks/            # Pre/post-train hooks
│       └── metrics/          # Custom metrics
├── datasets/                  # Training and prediction data
│   ├── train.csv             # Required: training data
│   ├── validation.csv        # Optional: validation split
│   ├── test.csv              # Optional: test evaluation
│   └── predict.csv           # Optional: prediction input
├── models/
│   ├── staging/              # All trained experiments
│   │   ├── exp-001/          # Auto-generated IDs
│   │   │   ├── model.zip     # Trained model
│   │   │   └── metadata.json # Metrics and config
│   │   └── exp-002/
│   └── production/           # Promoted production model
│       └── current -> ../staging/exp-003/  # Symlink
└── predictions/              # Prediction outputs
    └── predictions-20241104-143022.csv
```

### Model Registry

Models progress through stages:
- **Staging**: All trained models (auto-saved as `exp-NNN`)
- **Production**: Promoted model (via symlink for zero-copy promotion)

### Auto-Discovery Patterns

**Training**: `mloop train datasets/train.csv label_column --time 60`
- Reads: `datasets/train.csv`
- Generates: Unique experiment ID (`exp-001`, `exp-002`, ...)
- Saves to: `models/staging/exp-NNN/`
- Auto-promotes: First successful model → production

**Prediction**: `mloop predict` (no arguments needed!)
- Auto-discovers: Production model from registry
- Auto-discovers: `datasets/predict.csv` if no data file specified
- Auto-generates: Timestamped output in `predictions/`

**Manual Override**: All paths can be explicitly specified when needed
```bash
mloop predict models/staging/exp-005/model.zip data/custom.csv --output results.csv
```

**Experiment Management**:
```bash
# List all experiments with status and metrics
mloop list
# Shows: ID, Timestamp, Status, Best Metric, Stage (Production/Staging)

# Manually promote a specific experiment to production
mloop promote exp-003
# Replaces current production model with exp-003

# Show all experiments including failed ones
mloop list --all
```

### Complete Workflow Example

```bash
# 1. Initialize project
mloop init my-ml-project --task regression
cd my-ml-project

# 2. Add training data
cp ~/data/train.csv datasets/train.csv

# 3. Train multiple experiments
mloop train datasets/train.csv price --time 60   # Creates exp-001
mloop train datasets/train.csv price --time 120  # Creates exp-002
mloop train datasets/train.csv price --time 180  # Creates exp-003

# 4. Review all experiments
mloop list
# ┌─────────┬──────────────────┬───────────┬─────────────┬──────────────┐
# │   ID    │    Timestamp     │  Status   │ Best Metric │    Stage     │
# ├─────────┼──────────────────┼───────────┼─────────────┼──────────────┤
# │ exp-003 │ 2025-11-04 10:30 │ Completed │    0.9523   │ ★ Production │
# │ exp-002 │ 2025-11-04 10:20 │ Completed │    0.9401   │   Staging    │
# │ exp-001 │ 2025-11-04 10:10 │ Completed │    0.9278   │   Staging    │
# └─────────┴──────────────────┴───────────┴─────────────┴──────────────┘

# 5. Make predictions with production model
mloop predict  # Auto-uses exp-003 (production)

# 6. Promote a different experiment if needed
mloop promote exp-002
# Now exp-002 is in production
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Data Tools Integration](docs/DATA_TOOLS_INTEGRATION.md) - FileFlux + FilePrepper pipeline
- [Extensibility Guide](docs/EXTENSIBILITY.md) - Hooks and custom metrics
- [FilePrepper Integration](docs/FILEPREPPER_INTEGRATION.md) - Detailed CSV preprocessing
- [Configuration Reference](docs/configuration.md)
- [Pipeline Guide](docs/pipelines.md)
- [API Reference](docs/api-reference.md)
- [Examples](examples/)

## Requirements

- .NET 9.0 or later
- ML.NET 4.0.3 or later
- ML.NET AutoML 0.22.3 or later

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of the excellent [ML.NET](https://github.com/dotnet/machinelearning) framework by Microsoft.