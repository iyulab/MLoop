# Equipment Anomaly Detection - Complete MLOps Workflow

This document demonstrates the **complete MLOps pipeline** using MLoop, FilePrepper, and Ironbees agents for real-world equipment anomaly detection.

## 🎯 Project Overview

**Business Problem**: Detect equipment anomalies in manufacturing process before failures occur

**Data Sources**:
- **Sensor Data**: Temperature and Current measurements from 5공정 (Process 5)
- **Error Logs**: Historical error lot lists identifying failed processes
- **Time Period**: September-October 2021
- **Update Frequency**: 5-second intervals

**ML Task**: Binary Classification (Normal vs Anomaly)

## 📊 Data Characteristics

### Sensor Measurements
- **Features**:
  - `Temp`: Temperature in Celsius
  - `Current`: Electrical current in Amperes
  - `Process`: Process/Equipment ID (categorical)
  - `Time`: Timestamp
  - `Date`: Date of measurement

### Challenges
- **Time-series nature**: Sequential measurements with temporal dependencies
- **Class imbalance**: Anomalies are rare (<10% of data)
- **Multiple files**: 50+ daily CSV files need merging
- **Feature engineering**: Need to extract temporal patterns

## 🔄 Complete MLOps Pipeline

### Phase 1: Data Preparation (FilePrepper)

**Goal**: Consolidate multiple CSV files and create labeled dataset

```bash
# Run data preparation script
cd examples/equipment-anomaly-detection
./scripts/prepare-data.sh "../../ML-Resource/014-장비이상 조기탐지/Dataset/data/5공정_180sec"
```

**What happens**:
1. **File Consolidation**:
   - Merge 50+ daily sensor CSVs
   - Handle encoding issues (UTF-8 BOM)
   - Validate column consistency

2. **Labeling Logic**:
   ```
   For each sensor measurement:
     IF (Process, Date) IN Error_Lot_List:
       IsError = 1  (Anomaly)
     ELSE:
       IsError = 0  (Normal)
   ```

3. **Feature Engineering**:
   - Extract time features: Hour, Minute, Day of Week
   - Calculate rolling statistics: Mean, Std over 10-minute windows
   - Add lag features: Previous 3 measurements

4. **Data Splitting**:
   - Train: 60% (earlier dates)
   - Validation: 20% (middle dates)
   - Test: 20% (recent dates)
   - **Important**: Split by date, not random (time-series)

**FilePrepper Capabilities Demonstrated**:
- Multi-file CSV merging
- Column type inference
- Missing value detection
- Data validation
- Schema consistency checks

**Output**: `datasets/train.csv`, `datasets/validation.csv`, `datasets/test.csv`

---

### Phase 2: Data Analysis (Ironbees AI Agent)

**Goal**: Understand dataset characteristics and guide preprocessing

```bash
# Start MLoop agent in interactive mode
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj agent --interactive

# Switch to data-analyst agent
/switch data-analyst

# Ask for analysis
Analyze the dataset at datasets/train.csv. Focus on:
1. Class imbalance in IsError
2. Temperature and Current distributions
3. Missing value patterns
4. Feature correlations
5. Temporal patterns
```

**Expected Agent Response**:

```
📊 Dataset Overview
- Rows: 45,230
- Columns: 7 (Process, Temp, Current, Date, Hour, Minute, IsError)
- Memory: ~2.5 MB
- Task: Binary Classification (IsError)

🎯 Target Variable Analysis
- Class Distribution:
  * Normal (0): 92.3% (41,747 samples)
  * Anomaly (1): 7.7% (3,483 samples)
- **Severe class imbalance detected**
- Recommendation: Use F1-Score or AUC, not Accuracy

🔍 Feature Insights

Numerical Features:
- **Temp** (Temperature):
  * Range: 72.5°C to 82.4°C
  * Mean: 76.8°C, Std: 2.1°C
  * Distribution: Slightly right-skewed
  * Outliers: 23 values >80°C (mostly in anomaly class)

- **Current**:
  * Range: 1.45A to 1.85A
  * Mean: 1.62A, Std: 0.08A
  * Distribution: Nearly normal
  * Outliers: 15 values >1.75A

- **Hour**: 8-18 (working hours)
- **Minute**: 0-59 (uniform distribution)

Categorical Features:
- **Process**: 42 unique values (1-42)
- High cardinality → Use hash encoding or target encoding

⚠️ Data Quality Issues
- Missing values: None (excellent!)
- Duplicates: 127 exact duplicates (0.3%) - investigate
- Data drift potential: Temperature range increases over time

🔗 Feature Correlations
- Temp & Current: 0.34 (moderate positive correlation)
- Temp & IsError: 0.28 (anomalies tend to have higher temp)
- Current & IsError: 0.21 (anomalies tend to have higher current)
- Process & IsError: 0.15 (certain processes more prone to errors)

💡 Recommendations

Must-Do Preprocessing:
1. **Handle class imbalance**:
   - Use F1-Score metric
   - Consider SMOTE or class weighting
   - Stratified splitting

2. **Encode Process**:
   - Use target encoding (high cardinality)
   - Preserve process-specific patterns

3. **Normalize features**:
   - Min-max scaling for Temp and Current
   - Keep same scale for time features

4. **Feature engineering**:
   - Add rolling statistics (mean, std over 10-min windows)
   - Create lag features (previous measurements)
   - Compute rate of change (delta temp, delta current)
   - Add interaction: Temp × Current

Suggested Model Types:
- **LightGBM**: Best for tabular data with complex patterns
- **FastTree**: Good interpretability, feature importance
- **FastForest**: Robust to outliers

Next Steps:
1. Generate preprocessing script with these recommendations
2. Train model with 600-second budget (F1-Score optimization)
3. Evaluate on temporal test set (recent dates)
4. Monitor for data drift in production
```

---

### Phase 3: Preprocessing Script Generation (Ironbees AI Agent)

**Goal**: Generate production-ready C# preprocessing script

```bash
# Still in interactive mode, switch agent
/switch preprocessing-expert

# Request preprocessing script
Based on the data analysis, generate a comprehensive preprocessing script that:
1. Handles the class imbalance
2. Encodes the Process feature with target encoding
3. Normalizes Temp and Current
4. Creates rolling statistics and lag features
5. Adds interaction features
```

**Expected Agent Response**:

```
📋 Preprocessing Strategy

Based on your analysis, I'll create a script with:
1. **Class Imbalance**: Use ML.NET's built-in class weighting
2. **Process Encoding**: Target encoding based on error rates
3. **Normalization**: Min-max scaling for sensors
4. **Time-Series Features**: Rolling stats and lags
5. **Interactions**: Temp × Current

💻 Generated Script

[Complete C# code will be generated - see PREPROCESSING_SCRIPT.cs]

📝 Usage Instructions
1. Save to: .mloop/scripts/preprocessing/equipment_preprocessing.cs
2. The script will be auto-discovered by MLoop
3. Run: mloop preprocess
4. Verify: Check datasets/train_processed.csv

⚠️ Important Notes
- Target encoding may leak information if not done carefully
- Rolling stats need sufficient history (handled with padding)
- Lag features reduce dataset size slightly (first N rows dropped)
- Script is production-ready and handles new data
```

**Generated Script Location**: `.mloop/scripts/preprocessing/equipment_preprocessing.cs`

---

### Phase 4: Model Training (Ironbees AI Agent + MLoop)

**Goal**: Get model recommendations and train optimal model

```bash
# Switch to model architect agent
/switch model-architect

# Ask for model recommendation
We have:
- Binary classification (anomaly detection)
- Class imbalance (92% normal, 8% anomaly)
- 45K training samples
- Time-series nature with engineered features
- Need good F1-score (>0.75)
- Production latency <100ms

What model and training strategy do you recommend?
```

**Expected Agent Response**:

```
🎯 Problem Summary
- Binary classification with severe imbalance
- Sufficient data (45K samples)
- Need balance between precision and recall (F1-score)
- Time-series with temporal dependencies
- Production latency constraint

🏗️ Recommended Architecture

**Primary Recommendation: LightGBM**

Reasoning:
1. ✅ Excellent performance on imbalanced data
2. ✅ Handles categorical features natively
3. ✅ Fast inference (<50ms latency)
4. ✅ Feature importance for interpretability
5. ✅ Robust to outliers in sensor data

Expected Performance:
- F1-Score: 0.78-0.85
- Precision: 0.75-0.82
- Recall: 0.80-0.88
- Training time: 8-12 minutes

**Alternative Options**:
- **FastTree**: Faster training (5 min), F1~0.75-0.80
- **FastForest**: More robust, F1~0.73-0.78

⚙️ Training Configuration

Recommended Settings:
```bash
mloop train \
  --time 600 \
  --metric F1Score \
  --test-split 0.3
```

Why these settings:
- 600 seconds: Enough for LightGBM to converge
- F1Score: Optimal for imbalanced anomaly detection
- 0.3 test split: Larger validation for time-series

AutoML Trainers to Explore:
```yaml
trainers:
  - LightGBM      # Primary
  - FastTree      # Backup
  - FastForest    # Robustness
```

📊 Performance Expectations

Realistic Targets:
- **F1-Score**: 0.78-0.85 (good for anomaly detection)
- **Precision**: 0.75-0.82 (75-82% of predicted anomalies are real)
- **Recall**: 0.80-0.88 (catch 80-88% of real anomalies)
- **AUC**: 0.88-0.92

⚠️ Risk Factors:
1. **Class imbalance**: May need threshold tuning (default 0.5 → 0.3)
2. **Temporal dependencies**: Test set must be chronologically later
3. **Feature engineering quality**: Critical for performance
4. **Overfitting risk**: Monitor validation metrics

🔧 Next Steps

1. **Train Model**:
   ```bash
   mloop train --time 600 --metric F1Score
   ```

2. **Monitor Progress**:
   - Watch F1-Score convergence
   - Check for overfitting (train vs validation)
   - Note best trainer and hyperparameters

3. **Evaluate Results**:
   ```bash
   mloop evaluate --experiment <exp-id>
   mloop list  # Compare experiments
   ```

4. **Tune if Needed**:
   - If Precision too low: Increase decision threshold
   - If Recall too low: Decrease threshold or add more features
   - If overfitting: Reduce training time or add regularization
```

**Execute Training**:

```bash
# Run the training as recommended
dotnet run --project ../../src/MLoop.CLI/MLoop.CLI.csproj train \
  --time 600 \
  --metric F1Score \
  --test-split 0.3
```

**Training Output**:
```
🚀 MLoop Training Started
Task: Binary Classification
Metric: F1Score
Time Budget: 600 seconds

⏱️  Training Progress:
[00:30] FastTree     - F1: 0.7234 ✓
[01:45] LightGBM     - F1: 0.8123 ✓✓✓ (Best so far!)
[03:20] FastForest   - F1: 0.7456 ✓
[05:40] LightGBM v2  - F1: 0.8267 ✓✓✓✓ (New best!)
...

✅ Training Complete!
Best Trainer: LightGBM
Best F1-Score: 0.8267
Experiment ID: exp-20241111-001

Model saved to: models/staging/exp-20241111-001/
```

---

### Phase 5: Model Evaluation (MLoop + Ironbees AI Agent)

**Goal**: Thoroughly evaluate model performance

```bash
# Evaluate the best model
mloop evaluate --experiment exp-20241111-001

# List all experiments for comparison
mloop list
```

**Evaluation Output**:
```
📊 Model Evaluation Results
Experiment: exp-20241111-001
Trainer: LightGBM

Performance Metrics:
├─ F1-Score:    0.8267 ✅
├─ Precision:   0.7891 ✅
├─ Recall:      0.8701 ✅
├─ Accuracy:    0.9623 (⚠️ misleading due to imbalance)
└─ AUC:         0.9034 ✅

Confusion Matrix:
                Predicted
              Normal  Anomaly
Actual Normal  12,456     187
      Anomaly     142     958

Key Insights:
✅ High recall (87%): Catches most anomalies
✅ Good precision (79%): Low false alarm rate
✅ F1-Score >0.80: Excellent for anomaly detection
⚠️ 142 missed anomalies: Monitor in production

Feature Importance:
1. Temp_Rolling_Mean:     0.245
2. Current_Lag_1:         0.198
3. Temp:                  0.167
4. Process_Encoded:       0.134
5. Temp_x_Current:        0.102
```

**Ask AI Agent for Interpretation**:

```bash
# Switch to mlops-manager
/switch mlops-manager

# Ask for evaluation interpretation
I just evaluated my model with these results:
- F1-Score: 0.8267
- Precision: 0.7891
- Recall: 0.8701
- 142 missed anomalies out of 1,100 total

Is this good enough for production? What should I do next?
```

**Expected Agent Response**:

```
📋 Current Status Assessment

Your model performance is **excellent for anomaly detection**:

✅ Strengths:
- F1-Score 0.83: Well above 0.75 target
- Recall 0.87: Catches 87% of anomalies (critical for safety)
- Precision 0.79: Reasonable false alarm rate
- AUC 0.90: Strong discrimination ability

⚠️ Considerations:
- 142 missed anomalies: 13% miss rate
- 187 false alarms: May cause alert fatigue

🎯 Recommended Next Steps

**Immediate Actions**:

1. **Threshold Tuning** (Optional):
   Current threshold: 0.5 (default)

   Experiment with:
   - Threshold 0.3: Higher recall (fewer misses), lower precision (more false alarms)
   - Threshold 0.7: Higher precision, lower recall

   Business question: What's more costly - missed anomaly or false alarm?

2. **Production Readiness Check**:
   ```bash
   # Check prediction latency
   mloop predict \
     --input datasets/test.csv \
     --output predictions.csv \
     --benchmark

   # Verify: Latency <100ms ✓
   ```

3. **Model Promotion**:
   ```bash
   # Promote to production
   mloop promote exp-20241111-001

   # This copies model to models/production/
   ```

**Deployment Strategy**:

```yaml
Stage 1 - Shadow Deployment (Week 1-2):
  - Run model in parallel with existing system
  - Log predictions but don't act on them
  - Compare with actual failures
  - Tune threshold if needed

Stage 2 - Canary Deployment (Week 3-4):
  - Use model for 10% of equipment
  - Monitor false alarm rate
  - Collect feedback from operators

Stage 3 - Full Deployment (Week 5+):
  - Roll out to all equipment
  - Set up monitoring dashboard
  - Define retraining triggers
```

**Production Monitoring**:

```yaml
Metrics to Track:
  model_performance:
    - F1-Score (weekly)
    - Precision/Recall trend
    - Alert response time

  data_quality:
    - Sensor value ranges (detect drift)
    - Missing value rate
    - Prediction distribution

  business_impact:
    - Prevented failures
    - False alarm reduction
    - Maintenance cost savings
```

**Retraining Triggers**:
1. F1-Score drops below 0.75
2. Data drift detected (sensor ranges change)
3. New equipment types added
4. Monthly scheduled retraining

🔧 Production Setup Commands:

```bash
# 1. Promote model
mloop promote exp-20241111-001

# 2. Set up API serving (if needed)
mloop serve --port 8080

# 3. Or batch prediction pipeline
# Create scheduled task:
# - Daily: mloop predict --input daily-sensors.csv
# - Alert if anomaly detected
```

**Success Criteria**:
✅ Model deployed and running
✅ Monitoring dashboard live
✅ Alert system integrated
✅ Feedback loop established
✅ Retraining pipeline automated

Your model is **production-ready**! 🚀
```

---

### Phase 6: Production Deployment & Monitoring

**Implementation**:

1. **Model Serving Options**:

   **Option A: Real-time API**
   ```bash
   mloop serve --port 8080

   # API endpoints:
   # POST /predict - Single prediction
   # POST /predict/batch - Batch predictions
   # GET /health - Health check
   # GET /metrics - Model metrics
   ```

   **Option B: Batch Processing**
   ```bash
   # Daily batch prediction script
   #!/bin/bash
   DAILY_DATA="sensors/$(date +%Y-%m-%d).csv"
   OUTPUT="predictions/$(date +%Y-%m-%d).csv"

   mloop predict \
     --input $DAILY_DATA \
     --output $OUTPUT \
     --threshold 0.5

   # Send alerts for detected anomalies
   python alert_system.py --predictions $OUTPUT
   ```

2. **Monitoring Dashboard** (pseudo-code):
   ```python
   # monitor.py - Production monitoring
   from mloop_monitoring import MetricsCollector

   metrics = MetricsCollector()

   # Track predictions
   metrics.log_prediction(
       timestamp=now(),
       features=sensor_data,
       prediction=model_output,
       confidence=probability
   )

   # Detect data drift
   if metrics.detect_drift(window='7days'):
       alert_ops_team('Data drift detected!')

   # Performance degradation
   if metrics.f1_score(window='7days') < 0.75:
       trigger_retraining()
   ```

3. **Automated Retraining Pipeline**:
   ```yaml
   # .github/workflows/retrain.yml
   name: Weekly Model Retraining

   on:
     schedule:
       - cron: '0 2 * * 0'  # Sunday 2 AM

   jobs:
     retrain:
       steps:
         - name: Prepare latest data
           run: ./scripts/prepare-data.sh

         - name: Train model
           run: mloop train --time 600 --metric F1Score

         - name: Evaluate model
           run: |
             NEW_F1=$(mloop evaluate --latest | grep F1Score)
             PROD_F1=$(cat production/metrics.txt | grep F1Score)

             if [ $NEW_F1 > $PROD_F1 ]; then
               mloop promote --latest
             fi

         - name: Notify team
           run: send_slack_notification "Model retrained: F1=$NEW_F1"
   ```

---

## 📈 Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                           │
└─────────────────────────────────────────────────────────────┘

1. Data Preparation (FilePrepper)
   ├─ Merge 50+ CSV files
   ├─ Join with Error Lot List
   ├─ Feature engineering
   └─ Train/Val/Test split
          ↓
2. Data Analysis (Ironbees AI - data-analyst)
   ├─ Understand distributions
   ├─ Detect imbalance
   ├─ Find correlations
   └─ Generate insights
          ↓
3. Preprocessing (Ironbees AI - preprocessing-expert)
   ├─ Generate C# script
   ├─ Handle imbalance
   ├─ Encode categories
   ├─ Normalize features
   └─ Engineer features
          ↓
4. Model Selection (Ironbees AI - model-architect)
   ├─ Recommend trainers
   ├─ Set training config
   └─ Predict performance
          ↓
5. Training (MLoop AutoML)
   ├─ Explore multiple trainers
   ├─ Optimize F1-Score
   ├─ Track experiments
   └─ Save best model
          ↓
6. Evaluation (MLoop + mlops-manager)
   ├─ Compute metrics
   ├─ Analyze results
   ├─ Tune threshold
   └─ Validate readiness
          ↓
7. Deployment (MLoop + Monitoring)
   ├─ Promote to production
   ├─ Set up serving/batch
   ├─ Configure monitoring
   └─ Automate retraining
```

## 🎓 Key Learnings

### 1. FilePrepper Integration
- **Multi-file merging**: Essential for time-series data spread across files
- **Data validation**: Catch encoding and schema issues early
- **Labeling logic**: Domain knowledge (Error Lot List) creates labels

### 2. AI-Powered Data Analysis
- **data-analyst agent**: Provides deep insights beyond basic statistics
- **Class imbalance detection**: Guides metric selection (F1 vs Accuracy)
- **Feature correlation**: Informs feature engineering decisions

### 3. Automated Preprocessing
- **preprocessing-expert agent**: Generates production-ready C# code
- **Time-series awareness**: Rolling stats, lag features, temporal splits
- **Reproducibility**: Scripts apply same logic to new data

### 4. Intelligent Model Selection
- **model-architect agent**: Recommends based on data characteristics
- **Performance prediction**: Sets realistic expectations
- **Training optimization**: Balances time, accuracy, and interpretability

### 5. MLOps Best Practices
- **mlops-manager agent**: Orchestrates complete workflow
- **Experiment tracking**: Compare and select best model
- **Monitoring**: Detect drift and performance degradation
- **Automation**: Retraining pipelines for continuous improvement

## 🚀 Production Deployment Checklist

- [ ] Data pipeline automated (FilePrepper + scheduling)
- [ ] Preprocessing script tested on new data
- [ ] Model achieves F1-Score >0.75
- [ ] Prediction latency <100ms
- [ ] Serving API or batch system deployed
- [ ] Monitoring dashboard configured
- [ ] Alert thresholds set
- [ ] Retraining pipeline automated
- [ ] Rollback procedure documented
- [ ] Team trained on system operation

## 📚 References

- MLoop Documentation: `../../README.md`
- FilePrepper: Data preparation and validation
- Ironbees Agents: AI-powered MLOps assistance
- ML.NET AutoML: Automated model training
- Dataset: ML-Resource/014-장비이상 조기탐지

---

**Next**: See `README.md` for quick start guide and command reference.
