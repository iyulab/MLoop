# Housing Price Prediction Tutorial

**Difficulty**: Beginner
**Task Type**: Regression
**Dataset**: House features and prices
**Time**: 10 minutes

## What You'll Learn

- Train a regression model (predict continuous values)
- Work with multiple feature types (numeric and categorical)
- Understand R², MAE, and RMSE metrics
- Make price predictions for new houses

---

## The Problem

Given house features (square feet, bedrooms, location, etc.), predict the **selling price**.

This is a **regression** problem - predicting a continuous numerical value (vs classification which predicts categories).

**Real-world applications**:
- Real estate price estimation
- Sales forecasting
- Demand prediction
- Risk assessment

---

## Dataset

**File**: `datasets/houses.csv` (20 samples)

**Features**:
- `SquareFeet` - Total living area (numeric)
- `Bedrooms` - Number of bedrooms (numeric)
- `Bathrooms` - Number of bathrooms (numeric)
- `YearBuilt` - Construction year (numeric)
- `Neighborhood` - Location (categorical: Downtown, Suburbs, Rural)

**Label**: `Price` - Selling price in USD

**Example**:
- 1500 sqft, 3 bed, 2 bath, built 2010, Downtown → $425,000
- 900 sqft, 1 bed, 1 bath, built 2019, Rural → $195,000

---

## Step 1: Initialize Project

```bash
cd examples/tutorials/housing-prices
mloop init
```

---

## Step 2: Train Model

```bash
mloop train
```

**What happens**:
1. MLoop encodes the categorical column ("Neighborhood") and puts numeric features on a common scale for the linear trainers
2. Its trainer search tries regression trainers (FastTree, FastForest, LightGBM where its native library is present, SDCA, L-BFGS Poisson) on cross-validation folds
3. Optimizes for R² (coefficient of determination)
4. The best trial is fitted again on the whole training share and scored once on the rows held out

**Output** (the end of the run — the search prints a progress line per trial before it):
```
── Training Complete! ──────────────────────────────────────────────────────────

> Model: default
> Experiment ID: exp-001
> Best Trainer: LbfgsPoissonRegression
> Training Time: 45.10s

┌────────────────────────┬────────────────┐
│ Metric                 │ Value          │
├────────────────────────┼────────────────┤
│ RMSE                   │ 12892.3692     │
│ MAE                    │ 11192.7917     │
│ R SQUARED              │ 0.9943         │
│ ...                    │                │
└────────────────────────┴────────────────┘

Model saved to: models/default/staging/exp-001/model.zip
Model promoted to production!
```

This is one real run on the 20-row sample, whose held-out share is only 4 houses, so the numbers are indicative and
move from run to run (and the winning trainer with them). Use real data of a few hundred rows or more before trusting them.

---

## Step 3: Understand Regression Metrics

### R² - R-Squared (0.9943)
- **Interpretation**: Model explains 99.4% of price variation on the 4 held-out houses
- **Range**: 0 (random) to 1 (perfect)
- **Rule of thumb**:
  - R² > 0.9: Excellent
  - R² > 0.7: Good
  - R² > 0.5: Moderate
  - R² < 0.5: Poor

**Example**: If houses range $100K-$800K:
- R² = 0.9: Model predictions are very close to actual prices
- R² = 0.5: Model has large errors

### MAE - Mean Absolute Error (11,193)
- **Interpretation**: Average prediction error is $11,193
- **In real terms**: Predictions are typically off by about ±$11K
- **Lower is better** (0 = perfect)

**Example**:
- Actual: $425,000
- Predicted: $436,000
- Error: $11,000 (about the MAE)

### RMSE - Root Mean Squared Error (12,892)
- **Interpretation**: Similar to MAE but penalizes large errors more
- **Always ≥ MAE** (because it squares errors)
- **Use when**: Large errors are particularly bad

**RMSE > MAE indicates**: Some predictions have large errors
**RMSE ≈ MAE indicates**: Errors are consistent

---

## Step 4: Make Predictions

Create `predict.csv`:

```csv
SquareFeet,Bedrooms,Bathrooms,YearBuilt,Neighborhood
1650,3,2,2015,Suburbs
2300,4,3,2011,Downtown
1100,2,1,2018,Rural
```

Predict:

```bash
mloop predict predict.csv
```

**Output** (saved to `predictions/default-predictions-<timestamp>.csv`; the file holds your columns and these):
```
SquareFeet,Bedrooms,Bathrooms,YearBuilt,Neighborhood,Score,ScoreLowerBound,ScoreUpperBound,Confidence
1650,3,2,2015,Suburbs,384113.03,368010.22,400215.84,0.58366
2300,4,3,2011,Downtown,602270.6,586167.8,618373.44,0.58366
1100,2,1,2018,Rural,228897.19,212794.38,245000,0.58366
```

`Score` is the predicted price; `ScoreLowerBound` / `ScoreUpperBound` bracket it by the model's held-out error.

**Understanding predictions**:
- $384,113: 1650 sqft in Suburbs → typical price for that area
- $602,271: Larger house in expensive Downtown area
- $228,897: Small house in affordable Rural area

---

## Step 5: Analyze Feature Importance

Which features matter most for price? `mloop explain` shuffles one column at a time on a labeled file and reports how much the score falls:

```bash
mloop explain --data datasets/houses.csv
```

**Output** (this run):
```
exp-001 on houses.csv: 20 rows, r_squared 0.9930 (higher is better), 3 shuffle(s) per column
│ Column       │ Fall in r_squared │   ± sd │
│ SquareFeet   │           +0.6752 │ 0.1457 │
│ Neighborhood │           +0.3477 │ 0.0843 │
│ Bedrooms     │           +0.0705 │ 0.0137 │
│ YearBuilt    │           +0.0015 │ 0.0015 │
│ Bathrooms    │           +0.0004 │ 0.0004 │
```

**Reading it**: the model leans on `SquareFeet` and `Neighborhood` almost entirely; shuffling `Bathrooms` or `YearBuilt`
costs nothing. "Relies on" is not "causes": columns that move together (more bedrooms, more bathrooms, more square feet)
share one another's credit, and 20 rows is far too few to say anything about a real market.

---

## Step 6: Evaluate Model

```bash
mloop evaluate datasets/houses.csv
```

**Output** (trimmed):
```
── Evaluation Results ──────────────────────────────────────────────────────────

│ Metric    │ Training       │ Test           │ Difference     │
│ r_squared │ 0.9943         │ 0.9930         │ -0.0014        │
│ rmse      │ 12892.3692     │ 15540.5461     │ +2648.1770     │
│ mae       │ 11192.7917     │ 12173.5812     │ +980.7896      │
```

The "Test" column here is the whole file, rows the model trained on included — a real evaluation uses houses it has never seen.

---

## Key Concepts Learned

### 1. **Regression vs Classification**
| Regression | Classification |
|------------|----------------|
| Predict number (price, temperature) | Predict category (spam/not spam) |
| Continuous output | Discrete output |
| Metrics: R², MAE, RMSE | Metrics: Accuracy, F1, AUC |
| Example: $425,000 | Example: "Expensive" category |

### 2. **Feature Engineering**
ML.NET automatically handles:
- **Categorical encoding**: "Downtown" → [1,0,0], "Suburbs" → [0,1,0]
- **Normalization**: Scale features to similar ranges
- **Missing values**: Imputation (if present)

### 3. **Evaluation Strategies**
```bash
# Train on full data (small dataset)
mloop train datasets/houses.csv

# With train/test split (larger datasets)
mloop train datasets/houses.csv --test-split 0.2

# Cross-validation (most reliable, slower)
# Coming soon: --cross-validation flag
```

### 4. **Overfitting vs Underfitting**
- **Overfitting**: R² = 0.99 on training, 0.60 on test (memorized training data)
- **Underfitting**: R² = 0.50 on both (too simple model)
- **Good fit**: R² = 0.90 on training, 0.85 on test (generalizes well)

---

## Common Issues & Solutions

### Issue: Low R² (<0.6)

**Causes**:
1. **Insufficient features** (missing important predictors)
2. **Non-linear relationships** (price doesn't scale linearly with sqft)
3. **Outliers** (luxury penthouses skewing model)

**Solutions**:
```bash
# Check data quality
mloop train --analyze-data

# Try different algorithms
mloop train --time 120  # More time = more algorithms tested

# Add feature engineering
# Create .mloop/scripts/preprocess/01_feature_engineering.cs
```

### Issue: Large MAE/RMSE

**Example**: MAE = $150,000 (way too high!)

**Causes**:
- Price range is huge ($100K - $2M)
- Missing important features (school district, condition)
- Outliers (foreclosures, luxury homes)

**Solutions**:
```bash
# Remove outliers
# .mloop/scripts/preprocess/02_outlier_removal.cs

# Log-transform prices (helps with wide ranges)
# Price → log(Price), then predict, then exp(prediction)

# Separate models for different price ranges
mloop train --name budget-homes    # < $300K
mloop train --name luxury-homes    # > $700K
```

### Issue: RMSE >> MAE

**Example**: MAE = $30K, RMSE = $80K

**Meaning**: Some predictions have very large errors

**Investigation**:
```bash
# See where the error sits: the metrics again for each neighborhood
mloop evaluate datasets/houses.csv --slice-by Neighborhood

# Look for patterns:
# - Specific neighborhoods consistently wrong?
# - New construction (YearBuilt > 2018) poorly predicted?
# - Large homes (>3000 sqft) underestimated?
```

---

## Advanced: Feature Engineering

Add preprocessing script (`.mloop/scripts/preprocess/01_feature_engineering.cs`):

```csharp
using MLoop.Extensibility.Preprocessing;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class FeatureEngineeringScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        var lines = await File.ReadAllLinesAsync(ctx.InputPath);
        var header = lines[0] + ",Age,PricePerSqFt";

        var enhanced = new List<string> { header };

        foreach (var line in lines.Skip(1))
        {
            var parts = line.Split(',');
            var sqft = double.Parse(parts[0]);
            var yearBuilt = int.Parse(parts[3]);
            var price = double.Parse(parts[5]);

            var age = 2025 - yearBuilt;
            var pricePerSqFt = price / sqft;

            enhanced.Add($"{line},{age},{pricePerSqFt:F2}");
        }

        var outputPath = Path.Combine(ctx.OutputDirectory, "01_engineered.csv");
        await File.WriteAllLinesAsync(outputPath, enhanced);

        return outputPath;
    }
}
```

**New features**:
- `Age` = 2025 - YearBuilt (easier for model than year)
- `PricePerSqFt` = Price / SquareFeet (market indicator)

---

## Next Steps

### 1. Improve Model

**Collect more data**:
- At least 100+ houses for reliable predictions
- Cover diverse price ranges
- Include seasonal variations

**Add features**:
- School district rating
- Distance to amenities
- Lot size
- Garage spaces
- Recent renovations

### 2. Deploy Prediction API

```bash
# Start prediction server
mloop serve

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SquareFeet": 1800,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "YearBuilt": 2015,
    "Neighborhood": "Suburbs"
  }'

# Response:
# {"Score": 412500}
```

### 3. Monitor Predictions

```bash
# Add hook to track prediction drift
# .mloop/scripts/hooks/03_prediction_monitoring.cs
```

### 4. Try Related Tutorials

- **Classification**: `examples/tutorials/sentiment-analysis/` - Predict categories
- **Advanced Preprocessing**: `examples/tutorials/multi-file-workflow/` - Complex data prep
- **AI Agent**: `examples/tutorials/complete-beginner/` - Get intelligent assistance

---

## Summary

**You just**:
✅ Trained a regression model (R² = 0.91)
✅ Learned regression evaluation metrics
✅ Understood MAE vs RMSE trade-offs
✅ Made continuous value predictions

**Prediction accuracy**: Within ±6.2% of actual prices

**Real-world value**: Same approach works for:
- Sales forecasting
- Demand prediction
- Risk scoring
- Resource estimation

**Lines of code written**: 0 (pure configuration!)
