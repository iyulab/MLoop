# MLoop Recipe Library

Comprehensive collection of patterns, examples, and templates for common ML workflows.

---

## 📚 Quick Navigation

| Category | Count | Start Here |
|----------|-------|------------|
| [Tutorials](#tutorials) | 3 | `iris-classification/` |
| [Preprocessing](#preprocessing-recipes) | 7 | `04_data_cleaning.cs` |
| [Hooks](#lifecycle-hooks) | 4 | `01_data_validation.cs` |
| [Custom Metrics](#custom-metrics) | 3 | `01_profit_maximization.cs` |
| [AI Agents](#ai-agents) | 5 | `data-analyst/` |

---

## 🎓 Tutorials

End-to-end beginner guides with sample datasets and step-by-step instructions.

### Iris Classification
**Path**: `tutorials/iris-classification/`
**Difficulty**: ⭐ Beginner
**Task**: Multiclass Classification
**Time**: 5 minutes

Learn:
- Train your first ML model
- Understand multiclass classification
- Interpret accuracy metrics

**Features**: 4 numeric features → 3 species categories
**Accuracy**: 93% with 30 seconds training

```bash
cd examples/tutorials/iris-classification
mloop init && mloop train
```

---

### Sentiment Analysis
**Path**: `tutorials/sentiment-analysis/`
**Difficulty**: ⭐ Beginner
**Task**: Binary Classification (Text)
**Time**: 10 minutes

Learn:
- Text classification with ML.NET
- Precision vs Recall trade-offs
- Work with natural language data

**Features**: Movie review text → Positive/Negative
**Accuracy**: 96% with 60 seconds training

```bash
cd examples/tutorials/sentiment-analysis
mloop init && mloop train
```

---

### Housing Price Prediction
**Path**: `tutorials/housing-prices/`
**Difficulty**: ⭐ Beginner
**Task**: Regression
**Time**: 10 minutes

Learn:
- Predict continuous values
- Understand R², MAE, RMSE metrics
- Handle mixed feature types (numeric + categorical)

**Features**: 5 features (sqft, beds, baths, year, location) → Price
**R-Squared**: 0.91 with 45 seconds training

```bash
cd examples/tutorials/housing-prices
mloop init && mloop train
```

---

## 🔧 Preprocessing Recipes

Ready-to-use scripts for common data preparation tasks.

**Path**: `examples/preprocessing-scripts/`
**Usage**: Copy to `.mloop/scripts/preprocess/` in your project

### Data Transformation Patterns

| Script | Purpose | When to Use | Complexity |
|--------|---------|-------------|------------|
| `01_datetime_features.cs` | Extract time components | Temporal data | ⭐⭐ |
| `02_unpivot_shipments.cs` | Wide-to-long transformation | Multi-column data | ⭐⭐⭐ |
| `03_feature_engineering.cs` | Calculate derived features | Domain-specific metrics | ⭐⭐ |

### Data Quality Patterns

| Script | Purpose | When to Use | Complexity |
|--------|---------|-------------|------------|
| `04_data_cleaning.cs` | Remove duplicates, trim whitespace | Messy CSV files | ⭐ |
| `05_encoding_normalization.cs` | Convert to UTF-8 | Encoding issues | ⭐ |
| `06_missing_value_imputation.cs` | Fill missing values | Incomplete data | ⭐⭐ |
| `07_outlier_detection.cs` | IQR-based outlier removal | Extreme values | ⭐⭐ |

### Usage Example

```bash
# 1. Copy script to your project
cp examples/preprocessing-scripts/04_data_cleaning.cs .mloop/scripts/preprocess/

# 2. Train (preprocessing runs automatically)
mloop train

# 3. Or run preprocessing separately
mloop preprocess --input raw.csv --output cleaned.csv
```

---

## 🪝 Lifecycle Hooks

Execute custom logic at key pipeline stages.

**Path**: `docs/examples/hooks/`
**Usage**: Copy to `.mloop/scripts/hooks/` in your project

### Available Hooks

| Hook | Trigger Point | Use Cases | Complexity |
|------|---------------|-----------|------------|
| `01_data_validation.cs` | Pre-train | Validate minimum rows, class balance | ⭐ |
| `02_mlflow_logging.cs` | Post-train | Log metrics to MLflow | ⭐⭐ |
| `03_performance_gate.cs` | Post-train | Abort if accuracy < threshold | ⭐ |
| `04_auto_deployment.cs` | Post-train | Trigger deployment on success | ⭐⭐⭐ |

### Usage Example

```bash
# 1. Copy hook to your project
cp docs/examples/hooks/01_data_validation.cs .mloop/scripts/hooks/

# 2. Train (hooks execute automatically)
mloop train

# Output:
# Pre-Train Hook: data-validation
#   ✓ Minimum 50 rows required: 150 found
#   ✓ Class balance acceptable: 33.3% / 33.3% / 33.3%
```

### Hook Pattern

```csharp
using MLoop.Extensibility.Hooks;

public class MyHook : IMLoopHook
{
    public async Task<HookResult> ExecuteAsync(HookContext context)
    {
        // Access training data
        var dataView = context.DataView;
        var labelColumn = context.Metadata["LabelColumn"];

        // Your validation logic here
        if (someCondition)
        {
            return HookResult.Continue();  // Proceed with training
        }
        else
        {
            return HookResult.Abort("Validation failed");  // Stop training
        }
    }
}
```

---

## 📊 Custom Metrics

Business-specific evaluation metrics beyond standard ML metrics.

**Path**: `docs/examples/metrics/`
**Usage**: Copy to `.mloop/scripts/metrics/` in your project

### Available Metrics

| Metric | Business Problem | Formula | Complexity |
|--------|------------------|---------|------------|
| `01_profit_maximization.cs` | Fraud detection | TP×Gain - FP×Cost | ⭐⭐ |
| `02_churn_prevention.cs` | Customer retention | Prevented LTV - Intervention Cost | ⭐⭐ |
| `03_roi_optimization.cs` | Marketing campaigns | (Revenue - Cost) / Cost | ⭐⭐ |

### Usage Example

```bash
# 1. Copy metric to your project
cp docs/examples/metrics/01_profit_maximization.cs .mloop/scripts/metrics/

# 2. Evaluate with custom metric
mloop evaluate test.csv

# Output:
# Standard Metrics:
#   Accuracy: 0.9200
#   F1 Score: 0.8950
#
# Custom Metrics:
#   Profit Per Transaction: $42.50
#   Total Profit (1000 transactions): $42,500
```

### Metric Pattern

```csharp
using MLoop.Extensibility.Metrics;

public class ProfitMetric : IMLoopMetric
{
    public async Task<MetricResult> CalculateAsync(MetricContext context)
    {
        var tp = context.TruePositives;
        var fp = context.FalsePositives;

        var gain = 100.0;  // Revenue per correct detection
        var cost = 20.0;   // Cost per false alarm

        var profit = (tp * gain) - (fp * cost);

        return new MetricResult
        {
            Name = "Profit",
            Value = profit,
            Description = $"Net profit: ${profit:F2}"
        };
    }
}
```

---

## 🤖 AI Agents

Intelligent assistants for ML guidance and automation.

**Path**: `examples/mloop-agents/.mloop/agents/`
**Usage**: Pre-installed, use `mloop agent` to interact

### Available Agents

| Agent | Purpose | Capabilities | When to Use |
|-------|---------|--------------|-------------|
| `data-analyst` | Dataset analysis | Quality issues, preprocessing recommendations | Before training |
| `model-architect` | Configuration | Time budgets, metric selection | Planning phase |
| `preprocessing-expert` | Feature engineering | Datetime features, encoding strategies | Data preparation |
| `experiment-explainer` | Result interpretation | Algorithm explanation, metric analysis | After training |
| `ml-tutor` | Learning assistant | ML concepts, Q&A, tutorials | Learning ML |

### Usage Example

```bash
# Start agent conversation
mloop agent

# Ask for help
> How can I improve my model accuracy?

# Agent provides specific recommendations based on:
# - Your dataset characteristics
# - Current metrics
# - Training configuration
# - Domain best practices
```

### Agent Conversation Flows

**Data Analysis Flow**:
```
User: "Analyze my dataset"
Agent (data-analyst):
  - Detects 15% missing values in "Age" column
  - Identifies class imbalance (80/20 split)
  - Recommends: Median imputation + SMOTE oversampling
```

**Model Tuning Flow**:
```
User: "My F1 score is only 0.65"
Agent (model-architect):
  - Analyzes feature count (5) vs dataset size (100 rows)
  - Recommends: Increase time_limit to 120s
  - Suggests: Try --metric f1_score instead of accuracy
```

---

## 🗂️ Recipe Categories

### By Difficulty

**⭐ Beginner** (No ML knowledge required):
- All 3 tutorials
- Data cleaning scripts (04-07)
- Basic hooks (01, 03)

**⭐⭐ Intermediate** (Some ML understanding):
- Feature engineering scripts (01-03)
- MLflow integration hook
- Custom metrics (all)

**⭐⭐⭐ Advanced** (ML expertise):
- Unpivot transformations
- Auto-deployment hooks
- Multi-model ensembles (coming soon)

### By Use Case

**Data Quality Issues**:
1. Check: `mloop train --analyze-data`
2. Fix: Copy relevant recipe from `04_*` - `07_*`
3. Generate: `mloop train --generate-script auto_fix.cs`

**Performance Optimization**:
1. Analyze: Use `data-analyst` agent
2. Tune: Follow `model-architect` recommendations
3. Monitor: Add `02_mlflow_logging.cs` hook

**Production Deployment**:
1. Validate: Add `03_performance_gate.cs` hook
2. Deploy: Add `04_auto_deployment.cs` hook
3. Monitor: Implement custom metrics

---

## 🔍 Finding the Right Recipe

### Decision Tree

**Q: What do you need?**

→ **Learn ML basics**: Start with `tutorials/`
  - Classification: `iris` or `sentiment-analysis`
  - Regression: `housing-prices`

→ **Fix data issues**: Use `preprocessing-scripts/`
  - Encoding errors: `05_encoding_normalization.cs`
  - Missing values: `06_missing_value_imputation.cs`
  - Outliers: `07_outlier_detection.cs`

→ **Add custom logic**: Use `hooks/`
  - Pre-training validation: `01_data_validation.cs`
  - Post-training actions: `02_mlflow_logging.cs`

→ **Business metrics**: Use `metrics/`
  - Financial impact: `01_profit_maximization.cs`
  - Customer value: `02_churn_prevention.cs`

→ **Get guidance**: Use AI agents
  - Data questions: `data-analyst`
  - Model questions: `model-architect`
  - Learning: `ml-tutor`

---

## 💡 Best Practices

### Starting a New Project

```bash
# 1. Initialize
mloop init

# 2. Analyze data quality
mloop train --analyze-data

# 3. Fix issues (if needed)
cp examples/preprocessing-scripts/XX_*.cs .mloop/scripts/preprocess/

# 4. Train
mloop train

# 5. Evaluate
mloop evaluate test.csv

# 6. Deploy
mloop serve
```

### Organizing Recipes in Your Project

```
my-project/
├── .mloop/
│   ├── scripts/
│   │   ├── preprocess/
│   │   │   ├── 01_encoding.cs        # From recipe 05
│   │   │   ├── 02_missing_values.cs  # From recipe 06
│   │   │   └── 03_outliers.cs        # From recipe 07
│   │   ├── hooks/
│   │   │   ├── 01_validation.cs      # From hooks/01
│   │   │   └── 02_mlflow.cs          # From hooks/02
│   │   └── metrics/
│   │       └── 01_profit.cs          # From metrics/01
│   └── models/
└── mloop.yaml
```

### Testing Recipes

```bash
# Test preprocessing
mloop preprocess --input test.csv --validate

# Test hooks (dry run)
# Coming soon: mloop hooks --dry-run

# Test metrics
mloop evaluate test.csv
```

---

## 📖 Additional Resources

### Documentation
- **Getting Started**: `docs/GUIDE.md`
- **Preprocessing Guide**: `examples/preprocessing-scripts/README.md`
- **Hooks Guide**: `docs/examples/hooks/README.md`
- **Metrics Guide**: `docs/examples/metrics/README.md`
- **AI Agent Usage**: `docs/AI-AGENT-USAGE.md`

### Community Recipes

Submit your own recipes via GitHub Pull Request:
1. Follow existing pattern structure
2. Include README with use case
3. Add tests if applicable
4. Update this index

---

## ⚡ Quick Reference

### Most Common Recipes

**Data Cleaning** → `04_data_cleaning.cs`
**Missing Values** → `06_missing_value_imputation.cs`
**Pre-train Validation** → `01_data_validation.cs`
**Custom Business Metric** → `01_profit_maximization.cs`

### Recipe Counts

- **Total Recipes**: 22
- **Preprocessing**: 7
- **Hooks**: 4
- **Metrics**: 3
- **Tutorials**: 3
- **AI Agents**: 5

---

**Last Updated**: January 7, 2026
**MLoop Version**: 0.2.0-draft
