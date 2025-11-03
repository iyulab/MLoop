# ml-test-project

MLoop machine learning project for binary-classification.

## Quick Start

### 1. Prepare Your Data

Place your training data in `data/processed/train.csv`:

```bash
# Your CSV should have:
# - One column for the label (target variable)
# - Other columns as features
```

### 2. Train a Model

```bash
mloop train data/processed/train.csv --label <your-label-column>
```

### 3. Make Predictions

```bash
mloop predict experiments/exp-001/model.zip data/new-data.csv
```

## Project Structure

```
ml-test-project/
├── .mloop/              # Internal MLoop metadata (gitignored)
├── mloop.yaml           # User configuration
├── data/
│   ├── processed/       # Your training/test data
│   └── predictions/     # Prediction outputs
├── experiments/         # Training experiments
├── models/
│   ├── staging/         # Staging models
│   └── production/      # Production models
└── README.md
```

## Configuration

Edit `mloop.yaml` to customize:
- Training time limit
- Evaluation metric
- Test split ratio

## Commands

- `mloop train` - Train a model
- `mloop predict` - Make predictions
- `mloop evaluate` - Evaluate model performance
- `mloop experiment list` - List all experiments

## Documentation

For more information, see [MLoop documentation](https://github.com/yourusername/mloop).
