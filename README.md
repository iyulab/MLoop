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
- **One command to train**: `mloop train datasets/train.csv --label price --time 60`
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

- **15 ML Task Types**: Full ML.NET coverage — classification, regression, clustering, ranking, forecasting, recommendation, anomaly detection, NLP, image, and more
- **AutoML Training**: Automatic model selection with ML.NET AutoML
- **Multi-Model Projects**: Manage multiple models (churn, revenue, etc.) in one project
- **On-Demand DL Runtimes**: `mloop runtime install torch` downloads native runtimes only when needed
- **Smart Predictions**: Production model auto-discovery and batch processing
- **Filesystem MLOps**: Git-friendly experiment tracking (no database required)
- **Fast Preprocessing**: Integrated FilePrepper (20x faster than pandas)
- **Extensibility**: Code-based hooks and custom metrics
- **Zero Config**: Works immediately with intelligent defaults
- **Encoding Detection**: Automatic CP949/EUC-KR to UTF-8 conversion for Korean text
- **Self-Update**: Update to latest version with `mloop update`

### Supported ML Tasks

| Category | Tasks | Runtime |
|----------|-------|:-------:|
| **Tabular** | Binary Classification, Multiclass Classification, Regression | Built-in |
| **Unsupervised** | Anomaly Detection (PCA), Clustering (K-Means) | Built-in |
| **Ranking** | Learning to Rank (LightGBM, FastTree) | Built-in |
| **Time Series** | Forecasting (SSA), Time Series Anomaly (SR-CNN) | Built-in |
| **Recommendation** | Matrix Factorization (collaborative filtering) | Built-in |
| **Deep Learning** | Image Classification, Object Detection, Text Classification, NER, Sentence Similarity, QA | On-demand (`mloop runtime install`) |

> Image Classification (directory loader → TensorFlow transfer learning, `tf` runtime) and Object
> Detection (COCO/YOLO loader → AutoFormerV2, `torch` runtime) are wired end-to-end through the runtime
> gate. Install the matching runtime to train. Both support the full `train → predict → evaluate` loop
> over image directories: image classification `evaluate` reports multiclass accuracy, object detection
> `predict` emits per-image detections (label + score + box) as JSON and `evaluate` reports mAP
> (`map_50` / `map_50_95`) via ML.NET's built-in object-detection evaluator.

## Quick Start

### Installation

**.NET Tool** (Recommended for .NET developers):

```bash
dotnet tool install --global mloop
```

The package ID on [NuGet.org](https://www.nuget.org/packages/mloop) is `mloop`. This is a
framework-dependent tool, so it reuses your installed .NET runtime — ideal for slim Docker images:

```dockerfile
RUN dotnet tool install --global mloop
ENV PATH="$PATH:/root/.dotnet/tools"
```

**Standalone Binary**:

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
mloop train datasets/train.csv --label price --time 60

# 4. Add the rows you want scored, then predict
cp ~/data/new.csv datasets/predict.csv
mloop predict
# ✅ Output: predictions/default-predictions-TIMESTAMP.csv
```

That's it! Your model is trained and ready to use.

## Core Commands

```bash
mloop init <project> --task <type>    # Initialize ML project
mloop train <data> --label <label> [options]  # Train with AutoML
mloop predict [model] [data]          # Run predictions
mloop list [--json]                    # View experiments
mloop promote [exp-id] [--latest|--best] [--json] [--decide-only]  # Promote to production (auto-select newest/best; --decide-only reports the pick without moving the pointer)
mloop detect <data> [--column <col>]   # One-shot TS-anomaly detection with SPC bounds (no training)
mloop evaluate <model> <test> <label> # Evaluate performance
mloop info <data>                      # Dataset profiling with encoding detection
mloop analyze profile <data>           # Column types, null %, cardinality, constant columns
mloop analyze correlation <data>       # High-correlation pairs and multicollinearity
mloop analyze importance <data> -l <col>  # Feature importance ranking (requires label)
mloop analyze outliers <data>          # Outlier count, rate, isolation-forest threshold
mloop analyze distribution <data>      # Skewness, kurtosis, normality tests
                                       # add --json to any for structured, LLM-consumable output
mloop validate                         # Validate mloop.yaml configuration
mloop prep plan --set <type[:method]> [--columns ...] [--json]  # Declare prep step (policy only, no data change)
mloop prep run [options]               # Run preprocessing pipeline
mloop features select --drop/--keep <cols> [--reset] [--json]   # Declare feature include/exclude (policy only)
mloop compare <exp1> <exp2>            # Compare experiment metrics
mloop compare --metrics-file <path> --json  # Provided-state: rank/select best from supplied metrics (no local .mloop/; distributed consumers)
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
mloop train korean_data.csv --label label --task regression
# Output: [Info] Converted CP949 → UTF-8

# Drop missing label values (default for classification)
mloop train data.csv --label label --task binary-classification --drop-missing-labels

# Machine-readable training: stdout becomes an NDJSON event stream
# (phase / trial / warning / result / error — one JSON object per line, flushed per event)
mloop train data.csv --label label --task binary-classification --time 60 --json
```

### Image Classification

Image classification reads a directory whose subfolders are class labels (folder name = label):

```bash
mloop init vision --task image-classification
cd vision

# Lay out images: datasets/images/<class>/<files>
#   datasets/images/OK/  img001.jpg ...
#   datasets/images/NG/  img101.jpg ...

mloop runtime install tf        # one-time TensorFlow CPU runtime (~182MB)
mloop train --task image-classification   # auto-detects datasets/images/
```

Supported extensions: `.jpg .jpeg .png .bmp .gif`. The trainer uses TensorFlow transfer learning,
so the `tf` runtime must be installed first. After training, `mloop predict` classifies new images
and `mloop evaluate exp-001 datasets/images` reports multiclass accuracy over a labelled directory.

### Object Detection

Object detection reads a directory holding a COCO-format annotations file plus the referenced images:

```bash
mloop init detector --task object-detection
cd detector

# Edit datasets/coco/annotations.json (COCO format), with images alongside it:
#   datasets/coco/
#   ├── annotations.json   # images[], annotations[] (bbox=[x,y,w,h]), categories[]
#   ├── img001.jpg
#   └── img002.jpg ...

mloop runtime install torch     # one-time TorchSharp (libtorch CPU) runtime
mloop train --task object-detection   # auto-detects datasets/coco/
```

The COCO `bbox` (`[x, y, width, height]`, absolute pixels) is converted to the `x0 y0 x1 y1` order
the AutoFormerV2 trainer expects. Conventional annotation file names — `annotations.json` and
`_annotations.coco.json` (Roboflow) — are auto-detected; `file_name` is resolved relative to the
annotations file. YOLO datasets (`images/` + `labels/*.txt`) are also auto-detected. The trainer
uses the `torch` runtime, which must be installed first.

After training, predict and evaluate run over the same directory layout:

```bash
mloop predict datasets/yolo      # → predictions/<model>-detections-<ts>.json (label + score + box per image)
mloop evaluate exp-001 datasets/yolo   # → mAP (map_50 / map_50_95)
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
mloop train datasets/train.csv --label price --time 60   # exp-001
mloop train datasets/train.csv --label price --time 120  # exp-002 (better)
mloop train datasets/train.csv --label price --time 180  # exp-003 (best!)

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
# All endpoints except /health require a JWT: mint one with `mloop token`
# (add --role admin for write endpoints) and send it as `Authorization: Bearer <token>`.
# A programmatic caller must issue/refresh the token itself — the server has no token endpoint.
export MLOOP_TOKEN=$(mloop token --quiet)
# GET /predict?name=churn   -H "Authorization: Bearer $MLOOP_TOKEN"
# GET /predict?name=revenue -H "Authorization: Bearer $MLOOP_TOKEN"
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
- **[Embedding MLoop.Core](docs/EMBEDDING.md)** - In-process consumers: DL-weight pruning recipe & security floor
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