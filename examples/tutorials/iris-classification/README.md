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

**File**: `datasets/iris.csv` (150 samples: 50 of each species, Fisher's classic Iris data)

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

# Initialize MLoop project (keeps the tutorial's mloop.yaml and adds the .mloop/ marker)
mloop init .

# Verify structure
ls -la .mloop/
```

You should see the project marker with its configuration and script folders:
```
.mloop/
├── config.json
└── scripts/
```

Trained models go to `models/` next to it, one folder per model name.

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
2. MLoop's trainer search tries ML.NET trainers on cross-validation folds (30 seconds)
3. The best trial is fitted again on the whole training share and checked on the rows held out
4. The model is saved to `models/default/staging/exp-001/` (the project's `models/` folder)

**Output** (the end of the run — the search itself prints a progress line per trial before it):
```
── Training Complete! ──────────────────────────────────────────────────────────

> Model: default
> Experiment ID: exp-001
> Best Trainer: FastForestOva [The refit of SdcaMaximumEntropy scored 0.6917 on
the rows it learned, against 0.9667 validated; the model is the refit of
FastForestOva (0.9667 on the rows it learned, 0.9583 validated).]
> Training Time: 30.16s

┌────────────────────┬────────┐
│ Metric             │ Value  │
├────────────────────┼────────┤
│ MICRO ACCURACY     │ 0.9333 │
│ MACRO ACCURACY     │ 0.9333 │
│ MACRO F1           │ 0.9327 │
│ LOG LOSS REDUCTION │ 0.9215 │
│ LOG LOSS           │ 0.0862 │
│ ECE                │ 0.0627 │
└────────────────────┴────────┘

Model saved to: models/default/staging/exp-001/model.zip

Model promoted to production!
   First model for this project — nothing to compare against yet
```

This is one real run; the search is time-limited, so your numbers and the winning trainer will differ a little.
The bracketed note is MLoop checking its own work: the search's top trial was a linear model that scored 0.97 on the
validation folds, but when fitted again on all the training rows it scored only 0.69 on the very rows it had learned, so
it did not stand and the best trial of another trainer took its place. You will not always see it.

---

## Step 3: Understand Metrics

**Micro Accuracy** (0.9333):
- Overall accuracy across all 3 classes — here 28 of the 30 held-out rows
- Good for balanced datasets like Iris

**Macro Accuracy** (0.9333):
- Average accuracy per class
- Treats all classes equally; the same as micro here because the classes are the same size

**Log Loss** (0.0862) and **Log Loss Reduction** (0.9215):
- Log loss measures prediction confidence; lower is better (0 = perfect), and wrong answers given with high confidence are punished hardest
- Log loss reduction is how much of the loss a model that only knows the class shares would have, this model removes

**ECE** (0.0627):
- Expected calibration error: how far the confidence is from how often it is right (0 = perfectly calibrated)

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

**Output** (saved to `predictions/default-predictions-<timestamp>.csv`; the file holds your columns and these):
```
SepalLength,SepalWidth,PetalLength,PetalWidth,PredictedLabel,Score.Setosa,Score.Versicolor,Score.Virginica,Confidence
5.1,3.5,1.4,0.3,Setosa,0.989315,0.0047118654,0.005973176,0.989315
6.2,2.9,4.3,1.3,Versicolor,0.0049155485,0.98616785,0.008916599,0.986168
6.0,3.0,4.8,1.8,Virginica,0.009208006,0.169544,0.82124805,0.821248
```

`Confidence` is the probability of the predicted class; the third flower, the one nearest the border, is the least sure.

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

**Output** (`mloop list`, trimmed):
```
│ Model   │   ID    │ When     │  Status   │ Trainer     │ Metric (micro_accuracy) │ Stage │
│ default │ exp-002 │ just now │ Completed │ FastForestOva │ 0.9333 │   -   │
│ default │ exp-001 │ 1m ago   │ Completed │ FastForestOva │ 0.9333 │ prod  │
```
`prod` marks the experiment that serves predictions; a new experiment stays in staging until you promote it (`mloop promote exp-002`) or it beats production by the promotion rules.

---

## Step 6: Evaluate Model

```bash
# Evaluate on the same data (for demonstration)
mloop evaluate datasets/iris.csv

# In real projects, use separate test data:
# mloop evaluate datasets/test.csv
```

**Output** (trimmed):
```
── Evaluation Results ──────────────────────────────────────────────────────────

│ Metric             │ Training │ Test   │ Difference │
│ micro_accuracy     │ 0.9333   │ 0.9600 │ +0.0267    │
│ log_loss           │ 0.0862   │ 0.0803 │ -0.0059    │
│ ece                │ 0.0627   │ 0.0190 │ -0.0437    │

Confusion (rows: actual · columns: predicted)
│            │ Setosa │ Versicolor │ Virginica │
│ Setosa     │     50 │          0 │         0 │
│ Versicolor │      0 │         49 │         1 │
│ Virginica  │      0 │          5 │        45 │
```

The "Test" column here is the whole file, including the rows the model trained on, so it flatters the model — a real evaluation uses rows it has never seen.

**Reading the confusion matrix**:
- Rows = actual species
- Columns = predicted species
- Diagonal = correct predictions
- One Versicolor was called Virginica and five Virginica were called Versicolor — the two species that overlap

---

## Key Concepts Learned

### 1. **Multiclass Classification**
Predicting one of 3+ categories (vs binary = 2 categories)

### 2. **Trainer search**
MLoop tries several ML.NET trainers within the time limit, scores each on cross-validation folds, and keeps the best

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

4. **AI agents**: Drive MLoop from an AI assistant through mloop-mcp — see `docs/AI-AGENTS.md`

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
- Dataset too small (a few dozen rows leaves a held-out share of a handful of rows, so accuracy moves in big steps)
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
✅ Reached about 93% held-out accuracy in 30 seconds on 150 rows (numbers vary a little run to run)
✅ Made predictions on new data
✅ Understood ML.NET evaluation metrics
✅ Learned MLoop experiment tracking

**Total code written**: 0 lines (just configuration!)

**Next**: Try `sentiment-analysis` tutorial for binary classification or `housing-prices` for regression.
