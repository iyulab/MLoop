# MLoop

A modern CLI tool for building, running, and managing ML.NET models with filesystem-based MLOps.

## Overview

MLoop fills the gap left by the discontinued ML.NET CLI, providing a simple yet powerful command-line interface for the entire machine learning lifecycle - from training to prediction to deployment.

## Key Features

- **Build Models**: Train ML.NET models using AutoML or custom configurations
- **Run Predictions**: Execute batch and single predictions with trained models
- **Filesystem-based MLOps**: Git-friendly experiment tracking and model versioning
- **Pipeline Automation**: Define and execute reproducible ML workflows
- **Model Registry**: Organize and manage model versions
- **API Serving**: Deploy models as REST APIs

## Installation

```bash
dotnet tool install -g mloop
```

## Quick Start

```bash
# Initialize a new ML project
mloop init my-project --task binary-classification

# Train a model
mloop train config.yaml

# Make predictions
mloop predict model.zip data.csv

# Serve model as API
mloop serve model.zip --port 5000
```

## Core Commands

- `init` - Initialize a new ML project
- `train` - Train models using AutoML or custom pipelines
- `predict` - Run predictions on new data
- `evaluate` - Evaluate model performance
- `serve` - Deploy models as REST APIs
- `pipeline` - Define and execute ML workflows
- `experiment` - Track and compare experiments
- `model` - Manage model registry

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

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration Reference](docs/configuration.md)
- [Pipeline Guide](docs/pipelines.md)
- [API Reference](docs/api-reference.md)
- [Examples](examples/)

## Requirements

- .NET 8.0 or later
- ML.NET 4.0 or later

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of the excellent [ML.NET](https://github.com/dotnet/machinelearning) framework by Microsoft.