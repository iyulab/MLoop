# Iris Classification Tutorial

**Difficulty**: Beginner
**Task Type**: Multiclass Classification
**Dataset**: Iris flowers (3 species)
**Time**: 5 minutes

## What You'll Learn

- Train your first ML.NET model with MLoop
- Understand multiclass classification
- Evaluate model performance with metrics
- Make predictions on new data

---

## The Problem

Given measurements of an iris flower (sepal length/width, petal length/width), predict which of 3 species it belongs to:
- **Setosa**
- **Versicolor**
- **Virginica**

This is a classic **multiclass classification** problem - predicting one of multiple categories.

---

## Dataset

**File**: `datasets/iris.csv` (30 samples)

**Features**:
- `SepalLength` - Sepal length in cm
- `SepalWidth` - Sepal width in cm
- `PetalLength` - Petal length in cm
- `PetalWidth` - Petal width in cm

**Label**: `Species` (Setosa, Versicolor, Virginica)

---

## Step 1: Initialize Project

```bash
# Navigate to tutorial directory
cd examples/tutorials/iris-classification

# Initialize MLoop project
mloop init

# Verify structure
ls -la .mloop/
```

You should see:
```
.mloop/
├── models/
└── temp/
```

---

## Step 2: Train Model

```bash
# Train with default configuration (from mloop.yaml)
mloop train

# Or specify parameters explicitly
mloop train datasets/iris.csv --label Species --task multiclass-classification --time 30
```

**What happens**:
1. MLoop loads your CSV data
2. ML.NET AutoML tries different algorithms (30 seconds)
3. Best model is automatically selected based on accuracy
4. Model is saved to `.mloop/models/default/staging/exp-XXXXX/`

**Expected output**:
```
Training Configuration
─────────────────────
Model         default
Task          multiclass-classification
Data File     datasets/iris.csv
Label Column  Species
Time Limit    30s
Metric        micro_accuracy

Trial 1: LightGbmMulti - micro_accuracy=0.9333
Trial 2: FastTreeOva - micro_accuracy=0.9000
...

Training Complete!
─────────────────
Model: default
Experiment ID: exp-20250107-143022
Best Trainer: LightGbmMulti
Training Time: 28.45s

METRIC              VALUE
MICRO ACCURACY      0.9333
MACRO ACCURACY      0.9286
LOG-LOSS            0.2145

Model promoted to production!
```

---

## Step 3: Understand Metrics

**Micro Accuracy** (0.9333):
- Overall accuracy across all 3 classes
- 93.33% of predictions are correct
- Good for balanced datasets like Iris

**Macro Accuracy** (0.9286):
- Average accuracy per class
- Treats all classes equally
- Useful when class sizes differ

**Log-Loss** (0.2145):
- Measures prediction confidence
- Lower is better (0 = perfect)
- Penalizes confident wrong predictions

---

## Step 4: Make Predictions

Create a test file `predict.csv`:

```csv
SepalLength,SepalWidth,PetalLength,PetalWidth
5.1,3.5,1.4,0.3
6.2,2.9,4.3,1.3
6.0,3.0,4.8,1.8
```

Predict:

```bash
mloop predict predict.csv
```

**Output**:
```
Predictions
───────────────────────────────
SepalLength  SepalWidth  PetalLength  PetalWidth  PredictedLabel  Score
5.1          3.5         1.4          0.3         Setosa          0.9987
6.2          2.9         4.3          1.3         Versicolor      0.8234
6.0          3.0         4.8          1.8         Virginica       0.7543

Results saved to: predict_predictions.csv
```

---

## Step 5: Experiment with Different Models

Try training with different time limits:

```bash
# Quick model (10 seconds)
mloop train --time 10

# Better model (60 seconds)
mloop train --time 60

# Compare experiments
mloop list
```

**Output**:
```
Experiments for model 'default'
────────────────────────────────────────────────────────────
EXP ID              TRAINER       MICRO_ACCURACY  TIME   STATUS
exp-20250107-143022 LightGbmMulti 0.9333          28.45s PRODUCTION
exp-20250107-143510 FastTreeOva   0.9000          9.87s  STAGING
```

---

## Step 6: Evaluate Model

```bash
# Evaluate on the same data (for demonstration)
mloop evaluate datasets/iris.csv

# In real projects, use separate test data:
# mloop evaluate datasets/test.csv
```

**Output**:
```
Evaluation Results
──────────────────
Model: default (production)
Data: datasets/iris.csv

METRIC              VALUE
MICRO ACCURACY      0.9333
MACRO ACCURACY      0.9286
LOG-LOSS            0.2145
LOG-LOSS REDUCTION  0.7855

Confusion Matrix:
              Setosa  Versicolor  Virginica
Setosa        10      0           0
Versicolor    0       9           1
Virginica     0       0           10
```

**Reading the confusion matrix**:
- Rows = actual species
- Columns = predicted species
- Diagonal = correct predictions
- One Versicolor was misclassified as Virginica

---

## Key Concepts Learned

### 1. **Multiclass Classification**
Predicting one of 3+ categories (vs binary = 2 categories)

### 2. **AutoML**
ML.NET automatically tries different algorithms and picks the best one

### 3. **Metrics**
- **Accuracy**: % of correct predictions
- **Log-Loss**: Confidence penalty
- Use `micro_accuracy` for balanced datasets

### 4. **Experiment Tracking**
Every training run creates a new experiment with:
- Unique ID
- Model file
- Metrics
- Training config

### 5. **Production Promotion**
Best model is automatically promoted to production slot

---

## Next Steps

### Try Different Configurations

**Optimize for log-loss instead of accuracy**:
```bash
mloop train --metric log_loss
```

**Train a specific model (default)**:
```bash
mloop train --name my-iris-model --label Species
```

**Analyze data quality**:
```bash
mloop train --analyze-data
```

### Explore MLoop Features

1. **Preprocessing**: Add feature engineering scripts
   ```bash
   mkdir -p .mloop/scripts/preprocess
   # Add custom preprocessing (see examples/preprocessing-scripts/)
   ```

2. **Custom Metrics**: Define business-specific metrics
   ```bash
   mkdir -p .mloop/scripts/metrics
   # Add custom metrics (see docs/examples/metrics/)
   ```

3. **Hooks**: Add lifecycle hooks for validation
   ```bash
   mkdir -p .mloop/scripts/hooks
   # Add hooks (see docs/examples/hooks/)
   ```

4. **AI Agent**: Get intelligent assistance
   ```bash
   mloop agent
   # Ask: "How can I improve my Iris model?"
   ```

### Learn More

- **Binary Classification**: See `examples/tutorials/sentiment-analysis/`
- **Regression**: See `examples/tutorials/housing-prices/`
- **Advanced Preprocessing**: See `examples/tutorials/multi-file-workflow/`
- **Complete Beginner Guide**: See `examples/tutorials/complete-beginner/`

---

## Troubleshooting

### Error: "Label column 'Species' not found"

**Cause**: CSV header doesn't match configuration
**Fix**: Check that CSV has `Species` column (case-sensitive)

### Warning: "Model saved to staging"

**Cause**: New model didn't beat production model
**Fix**: This is normal! Try:
- Increase `time_limit_seconds`
- Use different `metric`
- Add more training data

### Low Accuracy (<80%)

**Possible reasons**:
- Dataset too small (30 samples is minimal)
- Time limit too short
- Data quality issues

**Solutions**:
```bash
# Analyze data quality
mloop train --analyze-data

# Increase training time
mloop train --time 120

# Check for missing values or outliers
```

---

## Summary

**You just**:
✅ Trained a multiclass classification model
✅ Achieved 93% accuracy in 30 seconds
✅ Made predictions on new data
✅ Understood ML.NET evaluation metrics
✅ Learned MLoop experiment tracking

**Total code written**: 0 lines (just configuration!)

**Next**: Try `sentiment-analysis` tutorial for binary classification or `housing-prices` for regression.
