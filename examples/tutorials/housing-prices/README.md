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
1. ML.NET automatically handles categorical features (one-hot encoding for "Neighborhood")
2. Normalizes numeric features (different scales: sqft vs year)
3. Tries regression algorithms (FastTree, LightGBM, SDCA)
4. Optimizes for R² (coefficient of determination)

**Expected output**:
```
Training Configuration
─────────────────────
Model         default
Task          regression
Data File     datasets/houses.csv
Label Column  Price
Time Limit    45s
Metric        r_squared

Trial 1: FastTreeRegression - r_squared=0.8542
Trial 2: LightGbmRegression - r_squared=0.9123
Trial 3: SdcaRegression - r_squared=0.7834
...

Training Complete!
─────────────────
Best Trainer: LightGbmRegression
Training Time: 42.18s

METRIC              VALUE
R_SQUARED           0.9123
ABSOLUTE_LOSS       28,450
SQUARED_LOSS        1.24e9
RMS_LOSS            35,214

Model promoted to production!
```

---

## Step 3: Understand Regression Metrics

### R² - R-Squared (0.9123)
- **Interpretation**: Model explains 91.23% of price variation
- **Range**: 0 (random) to 1 (perfect)
- **Rule of thumb**:
  - R² > 0.9: Excellent
  - R² > 0.7: Good
  - R² > 0.5: Moderate
  - R² < 0.5: Poor

**Example**: If houses range $100K-$800K:
- R² = 0.9: Model predictions are very close to actual prices
- R² = 0.5: Model has large errors

### MAE - Mean Absolute Error (28,450)
- **Interpretation**: Average prediction error is $28,450
- **In real terms**: Predictions are typically off by ±$28K
- **Lower is better** (0 = perfect)

**Example**:
- Actual: $425,000
- Predicted: $453,000
- Error: $28,000 (within MAE)

### RMSE - Root Mean Squared Error (35,214)
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

**Output**:
```
Predictions
────────────────────────────────────────────────────────────────
SquareFeet  Bedrooms  Bathrooms  YearBuilt  Neighborhood  Score
1650        3         2          2015       Suburbs       $382,500
2300        4         3          2011       Downtown      $598,200
1100        2         1          2018       Rural         $238,100

Results saved to: predict_predictions.csv
```

**Understanding predictions**:
- $382,500: 1650 sqft in Suburbs → typical price for that area
- $598,200: Larger house in expensive Downtown area
- $238,100: Small house in affordable Rural area

---

## Step 5: Analyze Feature Importance

Which features matter most for price?

**Common patterns** (from our training):
1. **SquareFeet**: +1000 sqft → ~+$180K (strongest correlation)
2. **Neighborhood**: Downtown > Suburbs > Rural (~$150K difference)
3. **Bathrooms**: +1 bathroom → ~+$75K
4. **Bedrooms**: +1 bedroom → ~+$50K (less than bathrooms!)
5. **YearBuilt**: Newer (+5 years) → ~+$15K

**Surprising insight**: Bathrooms have more impact than bedrooms!
- **Why?**: Master bathrooms indicate luxury/size
- **Lesson**: Domain knowledge helps interpret models

---

## Step 6: Evaluate Model

```bash
mloop evaluate datasets/houses.csv
```

**Output**:
```
Evaluation Results
──────────────────
R_SQUARED           0.9123
ABSOLUTE_LOSS       28,450
SQUARED_LOSS        1.24e9
RMS_LOSS            35,214

Prediction Errors:
Actual      Predicted   Error       % Error
$425,000    $442,300    +$17,300    +4.1%
$315,000    $298,750    -$16,250    -5.2%
$585,000    $612,400    +$27,400    +4.7%
...

Mean Absolute Percentage Error (MAPE): 6.2%
```

**What is MAPE?**
- Average error as percentage of actual price
- 6.2% MAPE = predictions within ±6.2% on average
- For $400K house: ±$24,800 typical error

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
# Find worst predictions
mloop evaluate datasets/houses.csv --verbose

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

public class FeatureEngineeringScript : IPreprocessingScript
{
    public async Task<PreprocessingResult> ExecuteAsync(PreprocessingContext context)
    {
        var lines = await File.ReadAllLinesAsync(context.InputPath);
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

        var outputPath = context.GetTempPath("engineered.csv");
        await File.WriteAllLinesAsync(outputPath, enhanced);

        return new PreprocessingResult
        {
            OutputPath = outputPath,
            Success = true,
            Message = "Added Age and PricePerSqFt features"
        };
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
