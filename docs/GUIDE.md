# MLoop User Guide

Complete guide for using MLoop, the ML.NET CLI tool for building and managing machine learning models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Commands](#core-commands)
3. [Deployment Commands](#deployment-commands)
4. [Data Preprocessing](#data-preprocessing)
5. [Extensibility](#extensibility)
6. [Project Structure](#project-structure)
7. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
dotnet tool install -g mloop
```

### Initialize Project

```bash
# Create new ML project
mloop init my-ml-project --task regression
cd my-ml-project

# Project structure created:
# .mloop/               # Project marker
# datasets/             # Your training data goes here
# models/
#   ├── staging/        # All experiments
#   └── production/     # Promoted model
# predictions/          # Prediction outputs
```

### Train Your First Model

```bash
# 1. Add your training data
cp ~/data/train.csv datasets/train.csv

# 2. Train with AutoML (60 seconds)
mloop train datasets/train.csv price --time 60

# Output:
# ✅ Training complete: exp-001
# 📊 Best Trainer: FastTreeRegressor
# 📈 R²: 0.9523, RMSE: 12.45

# 3. Make predictions
mloop predict
# Auto-uses production model + datasets/predict.csv
# Output: predictions/predictions-20241110-143022.csv
```

---

## Core Commands

### `mloop init`

Initialize a new ML project with filesystem structure.

```bash
# Basic usage
mloop init <project-name> --task <regression|binary-classification|multiclass-classification>

# Examples
mloop init sales-forecast --task regression
mloop init fraud-detection --task binary-classification
mloop init sentiment-analysis --task multiclass-classification
```

**Created Structure**:
```
my-project/
├── .mloop/
│   └── scripts/
│       ├── hooks/       # Pre/post-train hooks
│       └── metrics/     # Custom metrics
├── datasets/
├── models/
│   ├── staging/
│   └── production/
└── predictions/
```

### `mloop train`

Train models using ML.NET AutoML with automatic experiment tracking.

```bash
# Basic training
mloop train <data-file> <label-column> [options]

# Options:
#   --time <seconds>     Training time budget (default: 60)
#   --metric <name>      Metric to optimize (default: task-dependent)
#   --test-split <0-1>   Test split fraction (default: 0.2)
#   --output <path>      Custom model output path

# Examples
mloop train datasets/train.csv price --time 120
mloop train data/sales.csv revenue --time 300 --metric r_squared
mloop train data/fraud.csv is_fraud --time 180 --test-split 0.3
```

**Output**:
- Model saved to `models/staging/exp-XXX/model.zip`
- Metadata saved to `models/staging/exp-XXX/metadata.json`
- First successful model auto-promoted to production

### `mloop predict`

Run predictions with trained models.

```bash
# Auto-discovery mode (uses production model + datasets/predict.csv)
mloop predict

# Custom paths
mloop predict <model-path> <data-file> [options]

# Options:
#   --output <path>      Output file path (default: predictions/predictions-TIMESTAMP.csv)

# Examples
mloop predict                                          # Auto mode
mloop predict models/staging/exp-003/model.zip data/test.csv
mloop predict --output results/forecast.csv
```

### `mloop list`

List all experiments with status and metrics.

```bash
# List all successful experiments
mloop list

# Show all experiments including failed
mloop list --all

# Output:
# ┌─────────┬──────────────────┬───────────┬─────────────┬──────────────┐
# │   ID    │    Timestamp     │  Status   │ Best Metric │    Stage     │
# ├─────────┼──────────────────┼───────────┼─────────────┼──────────────┤
# │ exp-003 │ 2025-11-10 10:30 │ Completed │    0.9523   │ ★ Production │
# │ exp-002 │ 2025-11-10 10:20 │ Completed │    0.9401   │   Staging    │
# │ exp-001 │ 2025-11-10 10:10 │ Completed │    0.9278   │   Staging    │
# └─────────┴──────────────────┴───────────┴─────────────┴──────────────┘
```

### `mloop promote`

Manually promote experiments to production or staging.

```bash
# Promote to production (default)
mloop promote <experiment-id>

# Promote to staging
mloop promote <experiment-id> --to staging

# Examples
mloop promote exp-002              # Promote to production
mloop promote exp-005 --to staging
```

### `mloop info`

Display dataset profiling information.

```bash
mloop info <data-file>

# Output:
# 📊 Dataset Information
# Rows: 1,000
# Columns: 15
#
# Column Details:
# - price (Numeric): Min=10.5, Max=999.9, Mean=245.3
# - category (Text): 5 unique values
# - date (DateTime): Range: 2024-01-01 to 2024-12-31
```

### `mloop evaluate`

Evaluate model performance on test data.

```bash
mloop evaluate <model-path> <test-data> <label-column>

# Example
mloop evaluate models/production/current/model.zip datasets/test.csv price

# Output:
# 📊 Evaluation Results
# R²: 0.9456
# RMSE: 13.21
# MAE: 9.87
```

### `mloop validate`

Validate extensibility scripts (hooks and metrics).

```bash
# Validate all scripts
mloop validate

# Validate specific type
mloop validate --type hooks
mloop validate --type metrics

# Output:
# ✅ Hook: PreTrainValidator (Valid)
# ✅ Metric: CustomF1Score (Valid)
# ❌ Hook: InvalidHook (Compilation failed)
```

### `mloop extensions`

List all discovered hooks and metrics.

```bash
mloop extensions

# Output:
# 🔌 Discovered Extensions
#
# Hooks (2):
# - PreTrainValidator (.mloop/scripts/hooks/01_validator.cs)
# - PostTrainLogger (.mloop/scripts/hooks/02_logger.cs)
#
# Metrics (1):
# - CustomF1Score (.mloop/scripts/metrics/f1_weighted.cs)
```

---

## Deployment Commands

### `mloop serve`

Start REST API server for model serving.

```bash
# Start server on default port (5000)
mloop serve

# Custom port and host
mloop serve --port 8080 --host 0.0.0.0

# Run in background
mloop serve --detach

# API Endpoints:
# GET  /health              - Health check
# GET  /info                - Production model information
# GET  /models              - List all models
# POST /predict             - Make predictions

# Example prediction request:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'

# Batch prediction:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 3.0, "feature2": 4.0}
  ]'
```

**Features**:
- Auto-loads production model from registry
- Swagger/OpenAPI documentation at `/swagger`
- Single and batch predictions
- JSON request/response format
- CORS enabled for web clients

### `mloop pipeline`

Execute ML workflow from YAML pipeline definition.

```bash
# Execute pipeline
mloop pipeline pipeline.yml

# Dry-run (validate without executing)
mloop pipeline pipeline.yml --dry-run

# Override variables
mloop pipeline pipeline.yml --vars '{"training_time": 300}'

# Save results to JSON
mloop pipeline pipeline.yml --save-result results.json
```

**Example Pipeline** (`pipeline.yml`):
```yaml
name: Complete ML Workflow
description: End-to-end ML pipeline

variables:
  data_dir: datasets
  training_time: 120

steps:
  - name: preprocess_data
    type: preprocess
    parameters:
      input_file: $data_dir/raw_data.csv
      output_file: $data_dir/train.csv
    continue_on_error: false

  - name: train_model
    type: train
    parameters:
      data_file: $data_dir/train.csv
      label_column: price
      training_time: $training_time
    continue_on_error: false

  - name: evaluate_model
    type: evaluate
    parameters:
      model: $train_model.experiment_id
      test_file: $data_dir/test.csv
    continue_on_error: false

  - name: promote_to_production
    type: promote
    parameters:
      experiment_id: $train_model.experiment_id
      stage: production
    continue_on_error: false
```

**Step Types**:
- `preprocess` - Data preprocessing
- `train` - Model training with AutoML
- `evaluate` - Model evaluation
- `predict` - Generate predictions
- `promote` - Promote model to production/staging

**Features**:
- Variable substitution with `$variable_name`
- Step output chaining with `$step_name.output_key`
- Conditional execution with `continue_on_error`
- Dry-run validation
- Result persistence to JSON

---

## Data Preprocessing

MLoop integrates with **FilePrepper** for high-performance CSV preprocessing.

### Quick Examples

```bash
# DateTime feature extraction
fileprepper datetime -i data.csv -o out.csv -c "OrderDate" -m features --header

# Expression calculations
fileprepper expression -i data.csv -o out.csv -e "total=price*quantity" --header

# Column selection
fileprepper select -i data.csv -o out.csv -c "id,price,quantity" --header

# Filtering
fileprepper filter -i data.csv -o out.csv -e "price>100" --header
```

### Performance

FilePrepper is **20x faster** than pandas for common preprocessing tasks:

| Operation | pandas | FilePrepper | Speedup |
|-----------|--------|-------------|---------|
| DateTime extraction | 2.5s | 0.12s | **20.8x** |
| Expression eval | 3.1s | 0.15s | **20.7x** |
| Filtering | 1.8s | 0.09s | **20.0x** |

### Preprocessing Pipeline

Create sequential preprocessing scripts:

```
.mloop/scripts/preprocess/
├── 01_datetime.cs
├── 02_calculate.cs
└── 03_normalize.cs
```

Each script receives the output of the previous script as input.

**Example Script** (`01_datetime.cs`):
```csharp
using MLoop.Extensibility;

public class DateTimePreprocessor : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var outputPath = Path.Combine(context.OutputDirectory, "01_datetime.csv");

        // Use FilePrepper for fast processing
        await context.Csv.ExtractDateTimeFeaturesAsync(
            inputPath: context.InputPath,
            outputPath: outputPath,
            columnName: "OrderDate",
            mode: DateTimeMode.Features
        );

        return outputPath;
    }
}
```

---

## Extensibility

MLoop supports code-based extensibility through hooks and custom metrics.

### Hooks

Execute custom logic before/after training.

**Script Location**: `.mloop/scripts/hooks/`

```csharp
using MLoop.Extensibility;

public class DataValidator : IMLoopHook
{
    public string Name => "Data Validator";

    public async Task<HookResult> ExecuteAsync(HookContext context)
    {
        // Pre-train validation
        var rowCount = context.DataView.GetRowCount();

        if (rowCount < 100)
        {
            return HookResult.Abort("Insufficient data: need at least 100 rows");
        }

        context.Logger.Info($"✅ Validated {rowCount} rows");
        return HookResult.Continue();
    }
}
```

### Custom Metrics

Add custom evaluation metrics.

**Script Location**: `.mloop/scripts/metrics/`

```csharp
using MLoop.Extensibility;

public class WeightedF1 : IMLoopMetric
{
    public string Name => "Weighted F1";

    public async Task<double> CalculateAsync(MetricContext context)
    {
        // Calculate custom metric
        var predictions = context.Predictions;
        var schema = predictions.Schema;

        // Your metric calculation logic
        double weightedF1 = CalculateWeightedF1Score(predictions);

        return weightedF1;
    }
}
```

### Validation

Validate scripts before using them:

```bash
mloop validate

# Output:
# ✅ Hook: DataValidator (Valid)
# ✅ Metric: WeightedF1 (Valid)
```

---

## Project Structure

### Convention over Configuration

MLoop uses intelligent defaults:

```
my-ml-project/
├── .mloop/                    # Project marker (like .git)
│   ├── config.json            # Optional project config
│   └── scripts/
│       ├── hooks/             # Pre/post-train hooks
│       ├── metrics/           # Custom metrics
│       └── preprocess/        # Preprocessing scripts
│
├── datasets/                  # Training and prediction data
│   ├── train.csv             # Required: training data
│   ├── validation.csv        # Optional: validation split
│   ├── test.csv              # Optional: test evaluation
│   └── predict.csv           # Optional: prediction input
│
├── models/
│   ├── staging/              # All trained experiments
│   │   ├── exp-001/
│   │   │   ├── model.zip     # Trained model
│   │   │   └── metadata.json # Metrics and config
│   │   └── exp-002/
│   └── production/           # Promoted production model
│       └── current -> ../staging/exp-003/  # Symlink
│
└── predictions/              # Prediction outputs
    └── predictions-20241110-143022.csv
```

### Auto-Discovery

**Training**:
- Reads: `datasets/train.csv`
- Generates: Unique experiment ID (`exp-001`, `exp-002`, ...)
- Saves to: `models/staging/exp-NNN/`
- Auto-promotes: First successful model → production

**Prediction**:
- Auto-discovers: Production model from registry
- Auto-discovers: `datasets/predict.csv` if no data file specified
- Auto-generates: Timestamped output in `predictions/`

**Manual Override**:
All paths can be explicitly specified when needed:
```bash
mloop predict models/staging/exp-005/model.zip data/custom.csv --output results.csv
```

---

## Best Practices

### Data Preparation

1. **Clean Data First**: MLoop expects preprocessed, clean CSV data
2. **Consistent Schema**: Ensure train/test/predict files have matching columns
3. **Label Column**: Must be numeric for regression, categorical for classification
4. **No Missing Values**: Handle missing data before training

### Training Workflow

1. **Start Small**: Begin with short training times (60s) to validate
2. **Iterate**: Gradually increase time budget for better models
3. **Experiment**: Try different time budgets and metrics
4. **Track**: Use `mloop list` to compare experiments
5. **Promote**: Manually promote best model with `mloop promote`

### Model Management

```bash
# Typical workflow
mloop train datasets/train.csv price --time 60   # Experiment
mloop train datasets/train.csv price --time 120  # Improve
mloop train datasets/train.csv price --time 180  # Optimize

# Review results
mloop list

# Promote best
mloop promote exp-003

# Validate
mloop evaluate models/production/current/model.zip datasets/test.csv price

# Deploy
mloop predict
```

### Performance Tips

1. **Test Split**: Use 0.2 for small datasets, 0.1 for large datasets
2. **Time Budget**: 60s for prototyping, 300-600s for production
3. **Preprocessing**: Use FilePrepper for 20x speedup on large CSVs
4. **Parallelization**: Train multiple experiments in parallel (different terminals)

### Git Integration

MLoop is designed for version control:

```bash
# .gitignore recommendations
models/staging/*/model.zip    # Large model files (optional)
predictions/                   # Generated outputs
.mloop/temp/                  # Temporary files

# DO commit
.mloop/scripts/               # Custom hooks and metrics
.mloop/config.json            # Project configuration
models/staging/*/metadata.json # Experiment metadata (small)
```

### Troubleshooting

**Training fails with "Label column not found"**:
```bash
# Check column names
mloop info datasets/train.csv

# Use exact column name (case-sensitive)
mloop train datasets/train.csv Price  # If column is 'Price'
```

**Low model accuracy**:
1. Increase training time: `--time 300`
2. Verify data quality with `mloop info`
3. Try different metrics: `--metric f1_score` for classification
4. Check test split: `--test-split 0.1` for more training data

**Prediction output is wrong**:
1. Verify model: `mloop list` to check production model
2. Check schema: Input file must match training schema
3. Evaluate first: `mloop evaluate` on test data

---

## Concurrent Job Management

MLoop follows the Unix philosophy: each command runs as an independent process that exits when complete. This design provides natural isolation but requires external tools for managing multiple concurrent training jobs.

### Problem: Resource Exhaustion

When running multiple training jobs simultaneously, system resources (CPU/memory) can be exhausted:

```bash
# ❌ All 3 jobs compete for CPU → system slowdown
$ mloop train dataset1.csv price --time 1800 &
$ mloop train dataset2.csv price --time 1800 &
$ mloop train dataset3.csv price --time 1800 &
```

### Solution 1: Sequential Wrapper Script (Recommended)

Use the provided `mloop-queue.sh` script for simple sequential execution:

```bash
# ✅ Jobs run one after another
$ examples/scripts/mloop-queue.sh \
    "train dataset1.csv price --time 1800" \
    "train dataset2.csv price --time 1800" \
    "train dataset3.csv price --time 1800"

# Output:
# [1/3] Running: mloop train dataset1.csv price --time 1800
# ✅ Job 1 complete (30 min)
# [2/3] Running: mloop train dataset2.csv price --time 1800
# ✅ Job 2 complete (30 min)
# [3/3] Running: mloop train dataset3.csv price --time 1800
# ✅ Job 3 complete (30 min)
# 🎉 All jobs complete: 3/3 succeeded
```

**Features**:
- Sequential execution (one at a time)
- Progress tracking
- Error handling
- Total time reporting

### Solution 2: GNU Parallel (Advanced)

For advanced control, use [GNU Parallel](https://www.gnu.org/software/parallel/):

```bash
# Install GNU Parallel
$ sudo apt-get install parallel  # Ubuntu/Debian
$ brew install parallel           # macOS

# Run jobs with control
$ cat jobs.txt | parallel -j 1 --progress mloop {}

# jobs.txt:
# train dataset1.csv price --time 1800
# train dataset2.csv price --time 1800
# train dataset3.csv price --time 1800
```

**Advanced Options**:
```bash
# Limit to 2 concurrent jobs (if you have 2 GPUs)
$ parallel -j 2 mloop train {} ::: dataset1.csv dataset2.csv dataset3.csv

# Resume on failure
$ parallel --resume --joblog jobs.log -j 1 mloop train {} ::: dataset*.csv

# Different time budgets per job
$ parallel --colsep ',' mloop train {1} price --time {2} :::: jobs.csv

# jobs.csv:
# dataset1.csv,1800
# dataset2.csv,3600
# dataset3.csv,900
```

See `examples/scripts/mloop-parallel.sh` for complete examples.

### Solution 3: OS Job Control

Use standard Unix tools for background job management:

```bash
# Schedule jobs with 'at'
$ echo "mloop train dataset1.csv price --time 1800" | at now
$ echo "mloop train dataset2.csv price --time 1800" | at now + 30 minutes
$ echo "mloop train dataset3.csv price --time 1800" | at now + 60 minutes

# Monitor with screen/tmux
$ screen -S training1
$ mloop train dataset1.csv price --time 1800
# Ctrl+A, D to detach

$ screen -S training2
$ mloop train dataset2.csv price --time 1800
# Ctrl+A, D to detach

# List sessions
$ screen -ls

# Reattach
$ screen -r training1
```

### Solution 4: CI/CD Pipeline Orchestration

For production workflows, integrate with existing orchestration tools:

**GitHub Actions**:
```yaml
# .github/workflows/train-models.yml
jobs:
  train:
    strategy:
      matrix:
        dataset: [dataset1, dataset2, dataset3]
      max-parallel: 1  # Sequential execution
    steps:
      - name: Train model
        run: mloop train ${{ matrix.dataset }}.csv price --time 1800
```

**Kubernetes**:
```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mloop-training
spec:
  parallelism: 1  # One job at a time
  completions: 3   # Total jobs
  template:
    spec:
      containers:
      - name: mloop
        image: mloop:latest
        command: ["mloop", "train"]
```

**Airflow**:
```python
# dags/mloop_training.py
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('mloop_training', schedule_interval='@daily')

train1 = BashOperator(
    task_id='train_dataset1',
    bash_command='mloop train dataset1.csv price --time 1800',
    dag=dag
)

train2 = BashOperator(
    task_id='train_dataset2',
    bash_command='mloop train dataset2.csv price --time 1800',
    dag=dag
)

train1 >> train2  # Sequential execution
```

### Resource Monitoring

Monitor resource usage during training:

```bash
# CPU/Memory monitoring
$ watch -n 5 'ps aux | grep mloop'

# System resources
$ htop

# GPU usage (if applicable)
$ watch -n 1 nvidia-smi

# Disk I/O
$ iotop
```

### Best Practices

1. **Estimate Resource Needs**: Run one job first to measure CPU/memory usage
2. **Use Sequential Execution**: Default to sequential for safety (wrapper script)
3. **Enable Parallel Only When Safe**: If you have multiple GPUs or low CPU usage
4. **Monitor First Job**: Check resource usage before starting batch jobs
5. **Use Absolute Paths**: Avoid issues with working directory in background jobs
6. **Log Output**: Redirect stdout/stderr for debugging: `mloop train ... > job1.log 2>&1`

### Performance Tips

```bash
# Reduce CPU usage with nice (lower priority)
$ nice -n 19 mloop train dataset.csv price --time 1800

# Pin to specific CPU cores (Linux)
$ taskset -c 0-3 mloop train dataset.csv price --time 1800

# Set memory limits (Linux)
$ systemd-run --scope -p MemoryMax=4G mloop train dataset.csv price --time 1800
```

---

## Next Steps

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- **Examples**: Check `examples/` directory for complete workflows
- **Concurrent Execution**: See `examples/scripts/` for job management scripts
- **Contributing**: Review development guidelines in [ARCHITECTURE.md](ARCHITECTURE.md)

---

**MLoop**: Clean Data In, Trained Model Out - That's It.
