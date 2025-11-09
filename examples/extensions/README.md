# MLoop Extension Examples

This directory contains real-world examples of MLoop extensions (Hooks and Custom Metrics).

## 📁 Directory Structure

```
examples/extensions/
├── README.md (this file)
├── hooks/
│   ├── DataValidationHook.cs        # Pre-train data validation
│   ├── MLflowLoggingHook.cs         # Post-train MLflow integration
│   └── ModelPerformanceGateHook.cs  # Post-train performance gate
└── metrics/
    ├── ProfitMetric.cs              # Business profit optimization
    └── ChurnCostMetric.cs           # Customer churn cost minimization
```

## 🚀 Quick Start

### Step 1: Copy Examples to Your Project

```bash
# Initialize your MLoop project
mloop init my-project
cd my-project

# Create scripts directory
mkdir -p .mloop/scripts/hooks
mkdir -p .mloop/scripts/metrics

# Copy examples (choose what you need)
cp path/to/examples/extensions/hooks/DataValidationHook.cs .mloop/scripts/hooks/pre-train.cs
cp path/to/examples/extensions/metrics/ProfitMetric.cs .mloop/scripts/metrics/profit-metric.cs
```

### Step 2: Customize for Your Domain

Edit the copied files to match your business logic:
- Adjust validation thresholds
- Update business parameters (profit, cost, etc.)
- Modify logging endpoints

### Step 3: Run Training

```bash
# Extensions auto-discovered
mloop train data.csv --label target

# Or explicitly specify metric
mloop train data.csv --label target --metric profit-metric.cs
```

## 📚 Examples Overview

### Hooks

#### DataValidationHook.cs
**Purpose**: Validate data quality before training
**Hook Point**: `pre-train`
**Use Case**: Prevent training on insufficient or poor-quality data

**Features:**
- Minimum row count validation
- Class imbalance detection
- Missing value analysis

**When to use:**
- Automated training pipelines
- CI/CD integration
- Production environments

---

#### MLflowLoggingHook.cs
**Purpose**: Log training results to MLflow
**Hook Point**: `post-train`
**Use Case**: Experiment tracking and model registry

**Features:**
- Metrics logging (accuracy, F1, AUC)
- Parameter logging (trainer, config)
- Model artifact upload

**Prerequisites:**
- MLflow server running
- `MLFLOW_TRACKING_URI` environment variable set

**When to use:**
- Team collaboration
- Experiment comparison
- Model versioning

---

#### ModelPerformanceGateHook.cs
**Purpose**: Enforce minimum performance thresholds
**Hook Point**: `post-train`
**Use Case**: Quality gates for production deployment

**Features:**
- Configurable metric thresholds
- Automatic deployment trigger
- Performance comparison with baseline

**When to use:**
- Automated deployment pipelines
- Quality assurance
- Model governance

---

### Metrics

#### ProfitMetric.cs
**Purpose**: Optimize for expected profit instead of accuracy
**Use Case**: Business-driven model selection

**Business Parameters:**
- `PROFIT_PER_TP`: Revenue per true positive prediction
- `LOSS_PER_FP`: Cost per false positive prediction
- `LOSS_PER_FN`: Opportunity cost per false negative

**When to use:**
- Marketing campaigns (response modeling)
- Fraud detection (balance cost vs. catch rate)
- Recommendation systems (conversion optimization)

**Example domains:**
- E-commerce: Product recommendations
- Finance: Credit risk assessment
- Healthcare: Treatment recommendations

---

#### ChurnCostMetric.cs
**Purpose**: Minimize customer retention campaign costs
**Use Case**: Customer churn prevention optimization

**Business Parameters:**
- `CAMPAIGN_COST`: Cost per retention campaign
- `CHURN_LOSS`: Customer lifetime value lost
- `CAMPAIGN_SUCCESS`: Retention campaign success rate

**When to use:**
- SaaS subscription businesses
- Telecom customer retention
- Any subscription-based model

**Optimization**: Balances campaign costs vs. lost customer value

---

## 🔧 Customization Guide

### Adjusting Hook Behavior

```csharp
// Example: Change validation threshold
public async Task<HookResult> ExecuteAsync(HookContext ctx)
{
    var rowCount = ctx.DataView.Preview().RowView.Length;

    // ⬇️ Adjust this threshold for your needs
    if (rowCount < 500)  // Changed from 100
    {
        return HookResult.Abort("Need at least 500 rows");
    }

    return HookResult.Continue();
}
```

### Adjusting Business Metrics

```csharp
// Example: Update profit parameters for your business
public class ProfitMetric : IMLoopMetric
{
    // ⬇️ Update these based on your business model
    private const double PROFIT_PER_TP = 150.0;  // Your avg. conversion value
    private const double LOSS_PER_FP = -25.0;    // Your campaign cost
    private const double LOSS_PER_FN = -50.0;    // Your opportunity cost

    // ...
}
```

## 📖 Learning Resources

### Documentation
- [EXTENSIBILITY.md](../../docs/EXTENSIBILITY.md) - Complete extensibility guide
- [EXTENSIBILITY_ROADMAP.md](../../docs/EXTENSIBILITY_ROADMAP.md) - Implementation roadmap
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md#14-extensibility-system) - Architecture details

### API Reference
- `IMLoopHook` - Hook interface
- `IMLoopMetric` - Metric interface
- `HookContext` - Hook execution context
- `MetricContext` - Metric execution context
- `HookResult` - Hook return value

## 🎯 Best Practices

### Hook Design
1. **Single Responsibility**: One hook = one task
2. **Fail Gracefully**: Use try-catch, return Continue on non-critical errors
3. **Clear Logging**: Always log what the hook is doing
4. **Fast Execution**: Avoid expensive operations (< 100ms ideal)

### Metric Design
1. **Clear Business Logic**: Document all business parameters
2. **Transparent Calculation**: Log metric components for debugging
3. **Realistic Parameters**: Base on actual business data
4. **Test Sensitivity**: Verify metric responds to model quality

### Performance
1. **Sample for Validation**: Use `Preview(maxRows: 100)` not full dataset
2. **Cache Computations**: Don't recalculate same values
3. **Async Operations**: Use async/await for I/O operations
4. **Error Handling**: Don't let extension failures break training

## 🐛 Troubleshooting

### Extension Not Loading
```bash
# Check if extension exists
ls .mloop/scripts/hooks/
ls .mloop/scripts/metrics/

# Validate extension compiles
mloop validate .mloop/scripts/hooks/pre-train.cs
```

### Compilation Errors
```bash
# Check error message
mloop train data.csv --label target
# Look for: "⚠️  Compilation failed: ..."

# Common issues:
# - Missing using statements
# - Incorrect interface implementation
# - Syntax errors
```

### Hook Not Executing
```bash
# Verify hook discovery
mloop extensions list

# Check hook naming convention
# pre-train.cs, post-train.cs, pre-predict.cs, post-evaluate.cs

# Enable debug logging
mloop train data.csv --label target --verbose
```

### Metric Not Optimizing
```bash
# Verify metric is being used
mloop train data.csv --label target --metric my-metric.cs

# Check metric output
# Should see: "🎯 Optimization metric: [Your Metric Name]"

# Verify metric values are reasonable
# Check logs for metric calculation output
```

## 💡 Tips

1. **Start Simple**: Begin with DataValidationHook, then add complexity
2. **Test Locally**: Validate extensions work before deploying
3. **Version Control**: Commit extension scripts to Git
4. **Document Assumptions**: Comment business parameters and thresholds
5. **Monitor Performance**: Track extension execution time
6. **Iterate**: Refine based on actual training results

## 🤝 Contributing

Have a useful extension? Consider contributing!

1. Test thoroughly with multiple datasets
2. Document business use case
3. Add inline comments
4. Follow coding standards
5. Submit PR to MLoop repository

---

**Questions?** See [EXTENSIBILITY.md](../../docs/EXTENSIBILITY.md) for complete documentation.
