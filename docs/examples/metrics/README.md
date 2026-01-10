# Custom Business Metrics Examples

This directory contains production-ready examples of custom business metrics for MLoop.

## Overview

Custom metrics translate ML performance into business outcomes by calculating domain-specific values from model predictions. Unlike standard ML metrics (accuracy, F1, RMSE), business metrics measure financial impact, ROI, and real-world value.

## Example Metrics

### 1. Profit Maximization (`01_profit_maximization.cs`)

**Business Context**: Marketing campaign optimization, fraud detection with intervention costs

**Formula**: `Profit = (TP × Revenue) - (FP × Cost)`

**Configuration**:
```csharp
private const double REVENUE_PER_TRUE_POSITIVE = 100.0;
private const double COST_PER_FALSE_POSITIVE = 20.0;
```

**Use Case**: Optimize model to maximize expected profit by balancing revenue from correct predictions against costs of false positives.

### 2. Churn Prevention Value (`02_churn_prevention_value.cs`)

**Business Context**: SaaS customer retention, telecom churn prediction

**Formula**: `Value = (Prevented Churns × LTV) - (Interventions × Cost)`

**Configuration**:
```csharp
private const double CUSTOMER_LIFETIME_VALUE = 5000.0;
private const double INTERVENTION_COST = 50.0;
private const double INTERVENTION_SUCCESS_RATE = 0.70;
```

**Use Case**: Calculate business value from preventing customer churn, considering intervention costs and success rates.

### 3. ROI Optimization (`03_roi_optimization.cs`)

**Business Context**: Email marketing, paid advertising, sales lead prioritization

**Formula**: `ROI = ((Revenue - Cost) / Cost) × 100`

**Configuration**:
```csharp
private const double CONVERSION_VALUE = 200.0;
private const double CAMPAIGN_COST_PER_CONTACT = 5.0;
```

**Use Case**: Optimize marketing campaigns by maximizing return on investment.

## How to Use

### 1. Copy to Your Project

```bash
# Copy example to your project's metrics directory
cp docs/examples/metrics/01_profit_maximization.cs .mloop/scripts/metrics/

# Or create a new metric from template
# Adjust the configuration constants based on your business model
```

### 2. Customize Business Parameters

Edit the configuration constants at the top of each metric:

```csharp
// Adjust these values based on YOUR business model
private const double REVENUE_PER_TRUE_POSITIVE = 100.0;  // Your revenue
private const double COST_PER_FALSE_POSITIVE = 20.0;     // Your cost
```

### 3. Run Evaluation

Custom metrics are automatically calculated during model evaluation:

```bash
# Evaluate production model (auto-runs custom metrics)
mloop evaluate

# Evaluate specific experiment
mloop evaluate exp-001

# Custom metrics will be displayed after standard ML metrics
```

## Metric Output

Custom metrics display detailed business analysis:

```
📊 Calculating metric: Expected Profit
📊 Confusion Matrix:
   TP:  150 | FN:   20
   FP:   30 | TN:  800

💰 Financial Impact:
   Revenue:        $15,000.00
   Cost:             -$600.00
   Missed Revenue:  $2,000.00
   Net Profit:     $14,400.00

Expected Profit: 14400.00 ($14,400.00 profit (150 TPs × $100 - 30 FPs × $20))
```

## Creating Custom Metrics

### Step 1: Implement `IMLoopMetric`

```csharp
using MLoop.Extensibility.Metrics;

public class MyBusinessMetric : IMLoopMetric
{
    public string Name => "My Metric Name";
    public string Description => "Brief description";

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        // 1. Validate task type
        if (ctx.TaskType != "BinaryClassification")
        {
            throw new InvalidOperationException("Requires binary classification");
        }

        // 2. Extract predictions and labels
        var predictions = (bool[])ctx.Predictions;
        var labels = (bool[])ctx.Labels;

        // 3. Calculate your business metric
        double metricValue = CalculateBusinessValue(predictions, labels);

        // 4. Return result
        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = metricValue,
            Description = $"Human-readable result: ${metricValue:F2}"
        });
    }
}
```

### Step 2: Add Configuration

```csharp
// Business-specific constants
private const double BUSINESS_CONSTANT_1 = 100.0;
private const double BUSINESS_CONSTANT_2 = 20.0;

// Document what each constant means
// BUSINESS_CONSTANT_1: Revenue per successful prediction
// BUSINESS_CONSTANT_2: Cost per failed prediction
```

### Step 3: Add Logging

```csharp
ctx.Logger.Info($"📊 Business Analysis:");
ctx.Logger.Info($"   Key Metric 1: {value1}");
ctx.Logger.Info($"   Key Metric 2: {value2}");
ctx.Logger.Info($"💰 Financial Impact: ${total:F2}");
```

### Step 4: Include Details

```csharp
return Task.FromResult(new MetricResult
{
    Name = Name,
    Value = metricValue,
    Description = "Summary",
    Details = new Dictionary<string, object>
    {
        ["SubMetric1"] = value1,
        ["SubMetric2"] = value2,
        ["Explanation"] = "Additional context"
    }
});
```

## Task Type Reference

Custom metrics must validate and handle different ML task types:

### Binary Classification
```csharp
var predictions = (bool[])ctx.Predictions;  // True/False predictions
var labels = (bool[])ctx.Labels;            // True/False ground truth
```

### Multiclass Classification
```csharp
var predictions = (int[])ctx.Predictions;  // Class indices (0, 1, 2, ...)
var labels = (int[])ctx.Labels;            // Class indices ground truth
```

### Regression
```csharp
var predictions = (float[])ctx.Predictions;  // Predicted values
var labels = (float[])ctx.Labels;            // Actual values
```

## Best Practices

### 1. Clear Business Context

Document what each metric measures and why it matters:

```csharp
/// <summary>
/// Churn Prevention Value Metric
///
/// Business Context:
/// - Predicting which customers will churn
/// - Intervention campaigns have 70% success rate
/// - Average customer LTV is $5,000
/// </summary>
```

### 2. Configurable Constants

Make business parameters easy to adjust:

```csharp
// ❌ Bad: Hardcoded magic numbers
double profit = truePositives * 100 - falsePositives * 20;

// ✅ Good: Named configuration constants
private const double REVENUE_PER_TP = 100.0;
private const double COST_PER_FP = 20.0;
double profit = truePositives * REVENUE_PER_TP - falsePositives * COST_PER_FP;
```

### 3. Transparent Calculations

Log intermediate values for verification:

```csharp
ctx.Logger.Info($"Intermediate calculations:");
ctx.Logger.Info($"   TPs: {truePositives} × ${REVENUE} = ${revenue}");
ctx.Logger.Info($"   FPs: {falsePositives} × ${COST} = ${cost}");
ctx.Logger.Info($"   Profit: ${revenue} - ${cost} = ${profit}");
```

### 4. Graceful Error Handling

Validate inputs and provide clear error messages:

```csharp
if (predictions.Length != labels.Length)
{
    throw new InvalidOperationException(
        $"Prediction count ({predictions.Length}) != Label count ({labels.Length})");
}
```

### 5. Include Opportunity Analysis

Show what value was left on the table:

```csharp
double missedRevenue = falseNegatives * REVENUE_PER_TP;
ctx.Logger.Info($"Missed opportunities: ${missedRevenue:F2}");
```

## Integration with MLoop Workflow

Custom metrics fit seamlessly into the standard MLoop workflow:

```bash
# 1. Train model (uses ML metrics for optimization)
mloop train

# 2. Evaluate model (runs both ML and business metrics)
mloop evaluate

# 3. Compare experiments (shows business impact)
mloop compare exp-001 exp-002

# 4. Promote model (decision based on business value)
mloop promote exp-001
```

## Philosophy Alignment

Custom metrics embody MLoop's "Excellent MLOps with Minimum Cost" philosophy:

- **Zero-Overhead**: Metrics only run during evaluation, no training impact
- **Zero-Knowledge**: Examples work out-of-box, just adjust constants
- **Business-Focused**: Translate ML performance into business decisions
- **Expert-Friendly**: Full customization for domain-specific needs

## Common Use Cases

| Industry | Metric | Key Question |
|----------|--------|--------------|
| E-commerce | Profit Maximization | "What's the expected revenue?" |
| SaaS | Churn Prevention | "How much LTV are we saving?" |
| Marketing | ROI Optimization | "What's the campaign ROI?" |
| Healthcare | Cost-Benefit | "Treatment cost vs. benefit?" |
| Finance | Risk-Adjusted Return | "Risk-adjusted profit?" |

## Troubleshooting

### Metric Not Running

Check that the metric script is in `.mloop/scripts/metrics/`:

```bash
ls .mloop/scripts/metrics/
# Should show: 01_profit_maximization.cs
```

### Type Mismatch Errors

Ensure you're casting to the correct type based on task:

```csharp
// Binary classification: bool[]
var predictions = (bool[])ctx.Predictions;

// Multiclass: int[]
var predictions = (int[])ctx.Predictions;

// Regression: float[]
var predictions = (float[])ctx.Predictions;
```

### No Output Displayed

Check that the metric returns a valid `MetricResult`:

```csharp
return Task.FromResult(new MetricResult
{
    Name = Name,  // Required
    Value = metricValue,  // Required
    Description = "Result summary"  // Recommended
});
```

## Next Steps

- Copy example metrics to your project
- Adjust business constants to match your domain
- Run `mloop evaluate` to see business impact
- Create custom metrics for your specific use cases
- Document your business assumptions in metric comments

## Resources

- [MLoop Extensibility Guide](../../GUIDE.md) - Complete extensibility documentation
- [Hook Examples](../hooks/) - Lifecycle hooks for validation and automation
- [Preprocessing Examples](../../../examples/preprocessing-scripts/) - Data transformation scripts
