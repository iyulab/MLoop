# Migration Guide: Multi-Model Support

This guide helps you migrate existing MLoop projects to the new multi-model configuration format.

## Overview

MLoop 0.2.0 introduces multi-model support, allowing multiple ML models (churn, revenue, ltv, etc.) within a single project. This requires an updated `mloop.yaml` format.

## Configuration Changes

### Before (Legacy Format)

```yaml
project_name: customer-churn
task: binary-classification
label_column: Churn

training:
  time_limit_seconds: 300
  metric: F1Score
  test_split: 0.2

data:
  train: datasets/train.csv
```

### After (Multi-Model Format)

```yaml
project: customer-churn

models:
  default:  # Model name - use 'default' for backward compatibility
    task: binary-classification
    label: Churn
    description: Customer churn prediction model
    training:
      time_limit_seconds: 300
      metric: F1Score
      test_split: 0.2

data:
  train: datasets/train.csv
```

## Migration Steps

### Step 1: Update mloop.yaml

1. Rename `project_name` to `project`
2. Rename `label_column` to `label`
3. Wrap model settings in `models.default:` block
4. Move `task` and `label` under the model definition

### Step 2: Directory Structure Migration

Your model files need to move to the new per-model directory structure:

```bash
# Old structure
models/
├── staging/exp-001/
└── production/

# New structure
models/
└── default/           # Model name directory
    ├── staging/exp-001/
    └── production/
```

**Automated migration:**
```bash
cd your-project

# Move staging experiments
mkdir -p models/default
mv models/staging models/default/staging

# Move production model (if exists)
if [ -d "models/production" ]; then
  mv models/production models/default/production
fi
```

**Windows PowerShell:**
```powershell
cd your-project

# Create model directory
New-Item -ItemType Directory -Path "models\default" -Force

# Move staging
Move-Item -Path "models\staging" -Destination "models\default\staging" -Force

# Move production (if exists)
if (Test-Path "models\production") {
    Move-Item -Path "models\production" -Destination "models\default\production" -Force
}
```

### Step 3: Verify Migration

```bash
# List experiments (should show your existing experiments)
mloop list --name default

# Check production model
mloop list --name default | grep Production
```

## CLI Command Changes

The `--name` option is now available on all model-related commands:

| Command | Usage |
|---------|-------|
| `train` | `mloop train --name default` (default if omitted) |
| `predict` | `mloop predict data.csv --name default` |
| `list` | `mloop list --name default` |
| `promote` | `mloop promote exp-001 --name default` |
| `evaluate` | `mloop evaluate exp-001 test.csv --name default` |

**Backward compatibility:** When `--name` is omitted, MLoop uses `default` as the model name.

## Adding Additional Models

After migration, you can add more models:

```yaml
project: customer-analytics

models:
  default:
    task: binary-classification
    label: Churn
    description: Customer churn prediction

  revenue:
    task: regression
    label: Revenue
    description: Revenue prediction model

  ltv:
    task: regression
    label: LifetimeValue
    description: Customer lifetime value

data:
  train: datasets/train.csv
```

```bash
# Train multiple models
mloop train --name default --label Churn
mloop train --name revenue --label Revenue
mloop train --name ltv --label LifetimeValue

# Each model has independent experiments
mloop list --name default
mloop list --name revenue
mloop list --name ltv
```

## API Changes

The REST API now supports multi-model queries:

```bash
# Health check (unchanged)
curl http://localhost:5000/health

# List all production models
curl -H "Authorization: Bearer TOKEN" http://localhost:5000/models

# List specific model
curl -H "Authorization: Bearer TOKEN" http://localhost:5000/models?name=churn

# Predict with specific model
curl -X POST \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": "value"}' \
  "http://localhost:5000/predict?name=churn"
```

## Troubleshooting

### "Model not found" errors

If you get model not found errors after migration:

1. Check directory structure:
   ```bash
   ls -la models/default/staging/
   ls -la models/default/production/
   ```

2. Verify mloop.yaml format:
   ```bash
   cat mloop.yaml
   ```

3. Ensure model name matches:
   ```bash
   mloop list --name default
   ```

### Experiments not showing

If `mloop list` shows empty results:

1. Check experiment metadata exists:
   ```bash
   ls models/default/staging/*/experiment.json
   ```

2. Re-run list with explicit model name:
   ```bash
   mloop list --name default
   ```

### Production model not working

If predictions fail after migration:

1. Check production model exists:
   ```bash
   ls models/default/production/model.zip
   ```

2. Re-promote if needed:
   ```bash
   mloop promote exp-001 --name default
   ```

## Support

For additional help:
- [GitHub Issues](https://github.com/iyulab/MLoop/issues)
- [Architecture Documentation](ARCHITECTURE.md)
