# ict-inspect

MLoop machine learning project for binary-classification.

## Quick Start

### 1. Prepare Your Data

Place your training data in `datasets/train.csv`:

```bash
# Your CSV should have:
# - One column for the label (target variable)
# - Other columns as features
```

### 2. Train a Model

```bash
# Train the default model
mloop train --label <your-label-column>

# Train with specific model name
mloop train --name default --label <your-label-column>
```

### 3. Make Predictions

```bash
# Predict using default model
mloop predict new-data.csv

# Predict using specific model
mloop predict new-data.csv --name default
```

## Multi-Model Support

MLoop supports multiple models per project. Each model has its own:
- Experiments (staging)
- Production deployment
- Metrics history

```bash
# Train different models for different targets
mloop train --name churn --label Churned --task binary-classification
mloop train --name revenue --label Revenue --task regression

# List experiments per model
mloop list --name churn
mloop list --name revenue

# Promote and predict
mloop promote exp-001 --name churn
mloop predict new-data.csv --name churn
```

## Project Structure

```
ict-inspect/
├── .mloop/              # Internal MLoop metadata (gitignored)
│   └── scripts/         # Hooks and custom metrics
├── mloop.yaml           # User configuration
├── datasets/            # Training data (train.csv)
└── models/
    └── default/
        ├── staging/     # Experimental models (exp-001, exp-002, ...)
        └── production/  # Promoted production model
```

## Configuration

Edit `mloop.yaml` to configure:
- Model definitions (task, label, training settings)
- Default data paths

## Commands

- `mloop init` - Initialize a new project
- `mloop train` - Train a model
- `mloop predict` - Make predictions
- `mloop list` - List experiments
- `mloop promote` - Promote experiment to production
- `mloop evaluate` - Evaluate model performance

## Documentation

For more information, see [MLoop documentation](https://github.com/iyulab/MLoop).
