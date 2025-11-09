# MLoop Extensibility System

**Version**: 0.2.0-alpha
**Status**: Design Specification
**Phase**: 1 (Hooks & Metrics)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Extension Types](#3-extension-types)
4. [Quick Start](#4-quick-start)
5. [Architecture](#5-architecture)
6. [API Reference](#6-api-reference)
7. [Real-World Examples](#7-real-world-examples)
8. [Performance](#8-performance)
9. [Testing](#9-testing)
10. [Best Practices](#10-best-practices)

---

## 1. Overview

### 1.1 Purpose

The MLoop Extensibility System enables **optional, code-based customization** of the AutoML pipeline while maintaining the core principle of simplicity.

**Core Value Proposition:**
- **Base Layer**: AutoML works perfectly without any extensions
- **Enhancement Layer**: Add domain knowledge through C# scripts for better model accuracy
- **Zero Overhead**: No performance cost when extensions are not used

### 1.2 What You Can Extend

| Extension Type | Purpose | Phase |
|----------------|---------|-------|
| **Hooks** | Lifecycle events (validation, logging, integration) | Phase 1 |
| **Custom Metrics** | Business-aligned model evaluation | Phase 1 |
| **Transforms** | Feature engineering | Phase 2 |
| **Pipelines** | Complete workflow control | Phase 2 |

### 1.3 Key Principles

1. **Completely Optional** - Extensions never required for basic operation
2. **Progressive Enhancement** - Add complexity only when needed
3. **Zero-Overhead** - No cost when not used (< 1ms check)
4. **Graceful Degradation** - Extension errors don't break AutoML
5. **Type-Safe** - Full C# type system, IDE support
6. **Convention-Based** - Automatic discovery via filesystem

---

## 2. Design Philosophy

### 2.1 AutoML First, Extensions When Needed

```bash
# ✅ Level 1: Pure AutoML (Default)
mloop train data.csv --label target
# → Works perfectly, no extensions needed

# ✅ Level 2: AutoML + Validation
.mloop/scripts/hooks/pre-train.cs  # Add data validation
mloop train data.csv --label target
# → Auto-discovered, enhances AutoML

# ✅ Level 3: AutoML + Business Metrics
.mloop/scripts/metrics/profit-metric.cs
mloop train data.csv --label target --metric profit-metric.cs
# → AutoML optimizes for business value
```

### 2.2 Why Extensions?

**Problem**: Pure AutoML limitations
- Generic metrics (accuracy, F1) don't align with business goals
- Cannot incorporate domain knowledge
- No integration with existing MLOps tools

**Solution**: Optional extensions
- **Hooks**: Add validation, logging, external integrations
- **Custom Metrics**: Optimize for business outcomes (profit, cost, ROI)
- **No Breaking Changes**: Base AutoML always works

### 2.3 Design Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| **Backward Compatible** | Existing commands work unchanged |
| **Zero-Overhead** | < 1ms check when extensions disabled |
| **Fail-Safe** | Extension errors → AutoML continues |
| **Type-Safe** | Compile-time checking via C# |
| **IDE Support** | IntelliSense, debugging, refactoring |

---

## 3. Extension Types

### 3.1 Hooks (Phase 1)

**Purpose**: Execute custom logic at specific lifecycle points

**Hook Points:**
```
mloop train data.csv
    ↓
[pre-train hook]  ← Data validation, preprocessing checks
    ↓
AutoML Training
    ↓
[post-train hook] ← Model validation, deployment, logging
    ↓
Save Results
```

**Use Cases:**
- Data quality validation
- MLflow/W&B integration
- Model performance checks
- Automated deployment triggers

### 3.2 Custom Metrics (Phase 1)

**Purpose**: Define business-specific optimization objectives

**Standard Metrics** (AutoML default):
```csharp
// Built-in: Accuracy, F1, AUC, Precision, Recall
mloop train data.csv --label target --metric accuracy
```

**Custom Business Metrics**:
```csharp
// Optimize for profit, not accuracy
public class ProfitMetric : IMLoopMetric
{
    public double Calculate(IDataView predictions)
    {
        // True Positive: +$100 profit
        // False Positive: -$50 cost
        // False Negative: -$30 opportunity cost
        return expectedProfit;
    }
}
```

**AutoML will optimize for your business metric instead of generic accuracy.**

### 3.3 Future Extensions (Phase 2+)

```
Phase 2:
- Transforms: Feature engineering scripts
- Pipelines: Complete workflow control

Not Planned:
- Marketplace: Too early, community-driven later
```

---

## 4. Quick Start

### 4.1 Using AutoML Without Extensions (Default)

```bash
# Initialize project
mloop init my-project

# Train with AutoML (no extensions)
mloop train data.csv --label target --time 300

# ✅ Works perfectly
```

### 4.2 Adding Your First Hook (5 minutes)

**Step 1: Generate Hook Template**
```bash
mloop new hook --name DataValidation --type pre-train
# Created: .mloop/scripts/hooks/pre-train.cs
```

**Step 2: Implement Hook**
```csharp
// .mloop/scripts/hooks/pre-train.cs
using MLoop.Extensibility;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Validation";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var rowCount = ctx.DataView.Preview().RowView.Length;

        if (rowCount < 100)
        {
            ctx.Logger.Error($"Insufficient data: {rowCount} rows");
            return HookResult.Abort("Need at least 100 rows");
        }

        ctx.Logger.Info($"✅ Validation passed: {rowCount} rows");
        return HookResult.Continue();
    }
}
```

**Step 3: Run Training (Auto-Discovery)**
```bash
mloop train data.csv --label target

# Output:
🔍 Discovering extensions...
   ✅ Found hook: pre-train.cs (Data Validation)

📊 Executing hook: Data Validation
   ✅ Validation passed: 1,234 rows

🚀 AutoML training...
   [████████████████████] 100%

✅ Training completed
```

### 4.3 Adding Custom Metric

**Step 1: Generate Metric Template**
```bash
mloop new metric --name ProfitMetric
# Created: .mloop/scripts/metrics/profit-metric.cs
```

**Step 2: Implement Business Logic**
```csharp
// .mloop/scripts/metrics/profit-metric.cs
using MLoop.Extensibility;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public bool HigherIsBetter => true;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        const double profitPerTP = 100.0;
        const double lossPerFP = -50.0;

        var metrics = ctx.MLContext.BinaryClassification
            .Evaluate(ctx.Predictions);

        return (metrics.PositiveRecall * profitPerTP) +
               (metrics.FalsePositiveRate * lossPerFP);
    }
}
```

**Step 3: Use Metric**
```bash
mloop train data.csv --label target --metric profit-metric.cs

# Output:
🎯 Optimization metric: Expected Profit (higher is better)

⏱️  AutoML searching...
   Trial 1: LightGbm → $45.32
   Trial 2: FastTree → $48.91 ⭐
   Trial 3: SdcaLogistic → $43.17

✅ Best model: FastTree ($48.91 expected profit)
```

---

## 5. Architecture

### 5.1 Directory Structure

```
my-project/
├── .mloop/
│   ├── scripts/                     # Extension scripts ⭐
│   │   ├── hooks/                   # Lifecycle hooks
│   │   │   ├── pre-train.cs
│   │   │   ├── post-train.cs
│   │   │   ├── pre-predict.cs
│   │   │   └── post-evaluate.cs
│   │   └── metrics/                 # Custom metrics
│   │       ├── profit-metric.cs
│   │       └── churn-cost.cs
│   ├── .cache/                      # Compiled DLLs (auto-generated)
│   │   └── scripts/
│   │       ├── hooks.pre-train.dll
│   │       └── metrics.profit-metric.dll
│   └── config.json
├── mloop.yaml
└── experiments/
```

### 5.2 Extension Discovery Flow

```
1. User runs: mloop train data.csv --label target
       ↓
2. Extension Check:
   - .mloop/scripts/ exists? → Yes
   - config.yaml: extensions.enabled? → Check (default: true if dir exists)
       ↓
3. Script Discovery:
   - Scan .mloop/scripts/hooks/*.cs
   - Scan .mloop/scripts/metrics/*.cs
       ↓
4. Hybrid Compilation:
   - Check .cache/*.dll (cached?)
   - If cached & up-to-date → Load DLL (fast: ~50ms)
   - If not → Compile .cs → Cache DLL (first time: ~500ms)
       ↓
5. Validation:
   - Implements required interface?
   - No compilation errors?
   - On failure → Warning + Continue with AutoML
       ↓
6. Execution:
   - Hook: Execute at lifecycle point
   - Metric: Pass to AutoML optimizer
       ↓
7. AutoML Training (always runs)
```

### 5.3 Hybrid Compilation Strategy

**Problem**: Pure Roslyn scripting
- ❌ Slow compilation on every run
- ❌ Limited IDE support
- ❌ No debugging

**Solution**: Hybrid approach
- ✅ Develop in .cs files (full IDE support)
- ✅ First run: Compile → Cache DLL
- ✅ Subsequent runs: Load cached DLL (fast)
- ✅ Auto-recompile on file change

**Implementation:**
```csharp
public class ScriptLoader
{
    public async Task<T?> LoadScriptAsync<T>(string scriptPath)
    {
        var dllPath = GetCachedDllPath(scriptPath);

        // Cached DLL up-to-date?
        if (File.Exists(dllPath) &&
            File.GetLastWriteTime(dllPath) > File.GetLastWriteTime(scriptPath))
        {
            return LoadFromDll<T>(dllPath);  // Fast path
        }

        // Compile .cs → DLL
        var assembly = await CompileScriptAsync(scriptPath);
        await SaveAssemblyAsync(assembly, dllPath);

        return LoadFromDll<T>(dllPath);
    }
}
```

**Performance:**
```
Extension Check (no scripts):  < 1ms
First Run (compile):           ~500ms
Cached Runs:                   ~50ms
AutoML Training:               ~300s (unchanged)
```

---

## 6. API Reference

### 6.1 Hook Interface

```csharp
namespace MLoop.Extensibility;

/// <summary>
/// Lifecycle hook for custom logic at specific points
/// </summary>
public interface IMLoopHook
{
    /// <summary>Hook display name</summary>
    string Name { get; }

    /// <summary>Execute hook logic</summary>
    /// <param name="context">Execution context with data access</param>
    /// <returns>Continue or abort training</returns>
    Task<HookResult> ExecuteAsync(HookContext context);
}
```

### 6.2 Hook Context

```csharp
/// <summary>
/// Context provided to hooks during execution
/// </summary>
public class HookContext
{
    /// <summary>ML.NET context</summary>
    public MLContext MLContext { get; }

    /// <summary>Current data view</summary>
    public IDataView DataView { get; }

    /// <summary>Logger for output</summary>
    public ILogger Logger { get; }

    /// <summary>Read-only metadata (experiment ID, config, etc.)</summary>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>Set custom metadata for next hooks</summary>
    public void SetMetadata(string key, object value);

    /// <summary>Get data view helper</summary>
    public IDataView GetDataView() => DataView;
}
```

### 6.3 Hook Result

```csharp
/// <summary>
/// Result of hook execution
/// </summary>
public class HookResult
{
    /// <summary>Should training continue?</summary>
    public bool ShouldContinue { get; init; }

    /// <summary>Message (reason for abort)</summary>
    public string? Message { get; init; }

    /// <summary>Continue training</summary>
    public static HookResult Continue() =>
        new() { ShouldContinue = true };

    /// <summary>Abort training with reason</summary>
    public static HookResult Abort(string message) =>
        new() { ShouldContinue = false, Message = message };
}
```

### 6.4 Metric Interface

```csharp
/// <summary>
/// Custom evaluation metric for AutoML optimization
/// </summary>
public interface IMLoopMetric
{
    /// <summary>Metric display name</summary>
    string Name { get; }

    /// <summary>Optimization direction</summary>
    bool HigherIsBetter { get; }

    /// <summary>Calculate metric value</summary>
    /// <param name="context">Evaluation context with predictions</param>
    /// <returns>Metric value for optimization</returns>
    Task<double> CalculateAsync(MetricContext context);
}
```

### 6.5 Metric Context

```csharp
/// <summary>
/// Context provided to metrics during evaluation
/// </summary>
public class MetricContext
{
    /// <summary>ML.NET context</summary>
    public MLContext MLContext { get; }

    /// <summary>Predictions from model</summary>
    public IDataView Predictions { get; }

    /// <summary>Label column name</summary>
    public string LabelColumn { get; }

    /// <summary>Score column name</summary>
    public string ScoreColumn { get; }

    /// <summary>Logger for output</summary>
    public ILogger Logger { get; }
}
```

---

## 7. Real-World Examples

### 7.1 Data Quality Validation Hook

```csharp
// .mloop/scripts/hooks/pre-train.cs
using MLoop.Extensibility;

public class DataQualityHook : IMLoopHook
{
    public string Name => "Data Quality Check";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var df = ctx.DataView;
        var preview = df.Preview(maxRows: 1000);

        // 1. Minimum row check
        var rowCount = preview.RowView.Length;
        if (rowCount < 100)
        {
            return HookResult.Abort(
                $"Insufficient data: {rowCount} < 100 rows");
        }

        // 2. Class imbalance check
        var labelCol = ctx.Metadata["LabelColumn"] as string;
        var distribution = AnalyzeClassBalance(df, labelCol);

        if (distribution.ImbalanceRatio > 20)
        {
            ctx.Logger.Warning(
                $"⚠️  Severe class imbalance: {distribution.ImbalanceRatio:F1}:1");
            ctx.Logger.Warning(
                "Consider: SMOTE, class weights, or collecting more data");
        }

        // 3. Missing values check
        var missingStats = AnalyzeMissingValues(preview);
        if (missingStats.HasColumnsOver(30))
        {
            ctx.Logger.Warning(
                "⚠️  Some columns have >30% missing values:");
            foreach (var col in missingStats.ProblematicColumns)
            {
                ctx.Logger.Warning($"   - {col.Name}: {col.MissingPercent:F1}%");
            }
        }

        ctx.Logger.Info($"✅ Data quality check passed: {rowCount} rows");
        return HookResult.Continue();
    }

    private ClassDistribution AnalyzeClassBalance(IDataView df, string labelCol)
    {
        // Implementation: Count class occurrences
        // Return ratio of majority/minority
    }

    private MissingValueStats AnalyzeMissingValues(DataDebuggerPreview preview)
    {
        // Implementation: Count missing values per column
    }
}
```

### 7.2 MLflow Integration Hook

```csharp
// .mloop/scripts/hooks/post-train.cs
using MLoop.Extensibility;
using MLflow.NET;

public class MLflowHook : IMLoopHook
{
    public string Name => "MLflow Logging";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var experimentId = ctx.Metadata["ExperimentId"] as string;
        var metrics = ctx.Metadata["Metrics"] as BinaryClassificationMetrics;
        var modelPath = ctx.Metadata["ModelPath"] as string;

        var mlflowUri = Environment.GetEnvironmentVariable("MLFLOW_TRACKING_URI")
                       ?? "http://localhost:5000";

        using var client = new MlflowClient(mlflowUri);

        try
        {
            // 1. Create MLflow run
            var runId = await client.CreateRunAsync(experimentId);

            // 2. Log metrics
            await client.LogMetricsAsync(runId, new Dictionary<string, double>
            {
                ["accuracy"] = metrics.Accuracy,
                ["f1-score"] = metrics.F1Score,
                ["auc"] = metrics.AreaUnderRocCurve,
                ["precision"] = metrics.PositivePrecision,
                ["recall"] = metrics.PositiveRecall
            });

            // 3. Log parameters
            var trainerName = ctx.Metadata["BestTrainer"] as string;
            await client.LogParameterAsync(runId, "trainer", trainerName);
            await client.LogParameterAsync(runId, "mloop_version", "0.2.0");

            // 4. Log model artifact
            await client.LogArtifactAsync(runId, modelPath);

            ctx.Logger.Info($"✅ Logged to MLflow: {mlflowUri}/#/runs/{runId}");

            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Warning($"⚠️  MLflow logging failed: {ex.Message}");
            // Don't fail training because of logging issue
            return HookResult.Continue();
        }
    }
}
```

**Usage:**
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mloop train data.csv --label target

# Output:
📊 Executing hook: MLflow Logging
✅ Logged to MLflow: http://localhost:5000/#/runs/abc123
```

### 7.3 Churn Prevention Cost Metric

```csharp
// .mloop/scripts/metrics/churn-cost.cs
using MLoop.Extensibility;

public class ChurnCostMetric : IMLoopMetric
{
    public string Name => "Churn Prevention Cost";
    public bool HigherIsBetter => false;  // Minimize cost

    private const double CAMPAIGN_COST = 20.0;      // Per customer
    private const double CHURN_LOSS = 500.0;        // Lost customer value
    private const double CAMPAIGN_SUCCESS = 0.4;    // 40% success rate

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification
            .Evaluate(ctx.Predictions, labelColumnName: ctx.LabelColumn);

        // Simulate on 1000 customers
        const int totalCustomers = 1000;

        // Predictions
        var truePositives = metrics.PositiveRecall * totalCustomers;
        var falsePositives = metrics.FalsePositiveRate * totalCustomers;
        var falseNegatives = (1 - metrics.PositiveRecall) * totalCustomers;

        // Cost calculation
        var campaignCost = (truePositives + falsePositives) * CAMPAIGN_COST;
        var preventedChurns = truePositives * CAMPAIGN_SUCCESS;
        var missedChurns = falseNegatives;

        var totalCost = campaignCost + (missedChurns * CHURN_LOSS);
        var costPerCustomer = totalCost / totalCustomers;

        // Log breakdown
        ctx.Logger.Info($"Expected cost: ${costPerCustomer:F2} per customer");
        ctx.Logger.Info($"  Campaign cost: ${campaignCost:F2}");
        ctx.Logger.Info($"  Prevented churns: {preventedChurns:F0}");
        ctx.Logger.Info($"  Missed churns: {missedChurns:F0}");

        return costPerCustomer;
    }
}
```

**Usage:**
```bash
mloop train churn-data.csv --label will_churn --metric churn-cost.cs

# Output:
🎯 Optimization metric: Churn Prevention Cost (lower is better)

⏱️  AutoML searching...
   Trial 1: LightGbm → $245.30 per customer
   Trial 2: FastTree → $198.45 per customer ⭐
   Trial 3: SdcaLogistic → $267.12 per customer

✅ Best model: FastTree
   Expected cost: $198.45 per customer
   Campaign cost: $18,400
   Prevented churns: 368
   Missed churns: 52
```

### 7.4 Complete Integration Workflow

This section demonstrates the complete end-to-end workflow with hooks and metrics integrated into AutoML training.

#### Step 1: Project Initialization

```bash
# Create new project with extensibility support
mloop init fraud-detection --task binary-classification

# Output:
✓ Project 'fraud-detection' created successfully!

Folder structure:
  datasets/       # Training data (train.csv, validation.csv, test.csv)
  models/staging/ # Experimental models
  models/production/ # Auto-promoted best model
  .mloop/scripts/hooks/    # Custom hooks ⭐
  .mloop/scripts/metrics/  # Custom metrics ⭐
```

#### Step 2: Create Data Validation Hook

```bash
cd fraud-detection

# Create hook script
cat > .mloop/scripts/hooks/DataValidationHook.cs << 'EOF'
using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Fraud Data Validation";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var preview = ctx.DataView.Preview(maxRows: 100);
        var rowCount = preview.RowView.Length;

        if (rowCount < 100)
        {
            return HookResult.Abort($"Insufficient data: {rowCount} rows, need at least 100");
        }

        ctx.Logger.Info($"✅ Data validation passed: {rowCount} rows");
        return HookResult.Continue();
    }
}
EOF
```

#### Step 3: Create Business Metric

```bash
# Create profit-focused metric
cat > .mloop/scripts/metrics/FraudCostMetric.cs << 'EOF'
using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility;

public class FraudCostMetric : IMLoopMetric
{
    public string Name => "Fraud Detection Cost";
    public bool HigherIsBetter => false;  // Minimize cost

    private const double INVESTIGATION_COST = 15.0;     // Cost to investigate flagged transaction
    private const double FRAUD_LOSS = 250.0;            // Average loss from missed fraud

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(ctx.Predictions);

        // Calculate business cost
        var investigationCost = (metrics.PositiveRecall + (1 - metrics.NegativeRecall)) * INVESTIGATION_COST;
        var fraudLoss = (1 - metrics.PositiveRecall) * FRAUD_LOSS;

        var totalCost = investigationCost + fraudLoss;

        ctx.Logger.Info($"Total cost: ${totalCost:F2} per 100 transactions");
        ctx.Logger.Info($"  Investigation: ${investigationCost:F2}");
        ctx.Logger.Info($"  Missed fraud: ${fraudLoss:F2}");

        return totalCost;
    }
}
EOF
```

#### Step 4: Validate Scripts

```bash
# Validate both scripts compile correctly
mloop validate

# Output:
Validating MLoop extensibility scripts...

📋 Validating hooks...
  ✓ DataValidationHook.cs → Fraud Data Validation

📊 Validating metrics...
  ✓ FraudCostMetric.cs → Fraud Detection Cost (↓ Lower is better)

✓ Validation successful: All 2 scripts compiled successfully
```

#### Step 5: List Discovered Extensions

```bash
mloop extensions list

# Output:
MLoop Extensibility Scripts

📋 Hooks
┌──────────────────────────┬─────────────────────────┐
│ Name                     │ Type                    │
├──────────────────────────┼─────────────────────────┤
│ Fraud Data Validation    │ DataValidationHook      │
└──────────────────────────┴─────────────────────────┘
  Total: 1 hook(s)

📊 Metrics
┌──────────────────────────┬─────────────────────┬─────────────────────┐
│ Name                     │ Type                │ Direction           │
├──────────────────────────┼─────────────────────┼─────────────────────┤
│ Fraud Detection Cost     │ FraudCostMetric     │ ↓ Lower is better   │
└──────────────────────────┴─────────────────────┴─────────────────────┘
  Total: 1 metric(s)

✓ Found 2 extension(s) total

Use 'mloop validate' to check for compilation errors
```

#### Step 6: Train with Extensions

```bash
# Add training data
cp ~/fraud-transactions.csv datasets/train.csv

# Train with auto-discovered extensions
mloop train datasets/train.csv --label is_fraud --time 300

# Output:
🔍 Discovering extensions...
   ✅ Found hook: DataValidationHook.cs (Fraud Data Validation)
   ✅ Found metric: FraudCostMetric.cs (Fraud Detection Cost)

📋 Executing hook: Fraud Data Validation
   ✅ Data validation passed: 10,450 rows

🚀 AutoML training with custom metric: Fraud Detection Cost
   Optimizing for: Lower cost

⏱️  AutoML experiment progress:
   Trial 1: LightGbm → $45.20/100 transactions
   Trial 2: FastTree → $38.75/100 transactions ⭐
   Trial 3: SdcaLogisticRegression → $52.30/100 transactions
   Trial 4: FastForest → $39.10/100 transactions
   ...

✅ Best model: FastTree
   Custom metric - Fraud Detection Cost: $38.75/100 transactions
      Investigation: $22.50
      Missed fraud: $16.25

   Standard metrics:
      Accuracy: 0.9245
      F1 Score: 0.7823
      AUC: 0.9567

Model saved: experiments/exp-001/model.zip
```

#### Step 7: Production Workflow

```bash
# The model optimized for business cost (not just accuracy) is now ready

# Make predictions
mloop predict experiments/exp-001/model.zip new-transactions.csv

# Evaluate on test set
mloop evaluate experiments/exp-001/model.zip test-set.csv

# Output:
📊 Evaluation Results

Standard Metrics:
  Accuracy: 0.9201
  F1 Score: 0.7645
  AUC: 0.9523

Custom Metrics:
  custom_fraud_detection_cost: $39.12/100 transactions
     Investigation: $23.00
     Missed fraud: $16.12

# Promote to production if metrics are acceptable
mloop promote exp-001

# Output:
✓ Promoted exp-001 to production
  Model: models/production/model.zip
  Metrics: models/production/metrics.json
```

#### Key Takeaways from Integration Example

1. **Zero Configuration**: Extensions auto-discovered from `.mloop/scripts/` directory
2. **Compilation Check**: `mloop validate` catches errors before training
3. **List Discovery**: `mloop extensions list` shows what will be executed
4. **Business Alignment**: AutoML optimizes for fraud detection cost, not generic accuracy
5. **Graceful Degradation**: If scripts fail, AutoML continues with standard metrics
6. **Production Ready**: Model optimized for business outcomes, not technical metrics

---

## 8. Performance

### 8.1 Zero-Overhead Guarantee

**When extensions not used:**
```
Overhead: < 1ms (directory existence check only)
Training time: Unchanged
Memory: No additional allocation
```

**Benchmark:**
```csharp
// BenchmarkDotNet results
|              Method |      Mean |    Error |
|-------------------- |----------:|---------:|
| TrainWithoutScripts |  45.23 ms | 0.12 ms |  ← baseline
| TrainWithScriptsCheck | 45.31 ms | 0.15 ms |  ← +0.08ms
```

### 8.2 Extension Loading Performance

```
First Run (compile .cs → .dll):  ~500ms
Cached Runs (load .dll):         ~50ms
Hook Execution:                  < 100ms (user code dependent)
Total Overhead:                  < 600ms on first run, < 150ms cached
```

### 8.3 Compilation Caching

```bash
# First run: Compile
$ time mloop train data.csv --label target
Extension loading: 450ms (compile + cache)
Training: 45,000ms
Total: 45.45s

# Second run: Cached
$ time mloop train data.csv --label target
Extension loading: 35ms (load DLL) ← Cached!
Training: 45,000ms
Total: 45.04s
```

---

## 9. Testing

### 9.1 Extension Unit Tests

```csharp
// tests/Extensions/DataValidationHookTests.cs
public class DataValidationHookTests
{
    [Fact]
    public async Task ExecuteAsync_SufficientData_ReturnsContinu()
    {
        // Arrange
        var hook = new DataValidationHook();
        var context = CreateMockContext(rowCount: 1000);

        // Act
        var result = await hook.ExecuteAsync(context);

        // Assert
        Assert.True(result.ShouldContinue);
    }

    [Fact]
    public async Task ExecuteAsync_InsufficientData_ReturnsAbort()
    {
        // Arrange
        var hook = new DataValidationHook();
        var context = CreateMockContext(rowCount: 50);

        // Act
        var result = await hook.ExecuteAsync(context);

        // Assert
        Assert.False(result.ShouldContinue);
        Assert.Contains("Insufficient data", result.Message);
    }

    private HookContext CreateMockContext(int rowCount)
    {
        var mlContext = new MLContext();
        var dataView = CreateMockDataView(mlContext, rowCount);

        return new HookContext
        {
            MLContext = mlContext,
            DataView = dataView,
            Logger = Mock.Of<ILogger>(),
            Metadata = new Dictionary<string, object>
            {
                ["LabelColumn"] = "target"
            }
        };
    }
}
```

### 9.2 Integration Tests

```csharp
// tests/Integration/ExtensionIntegrationTests.cs
public class ExtensionIntegrationTests
{
    [Fact]
    public async Task Train_WithValidHook_ExecutesHook()
    {
        // Arrange
        using var testProject = new TempProject();
        await testProject.AddHookScript("pre-train.cs", PreTrainHookCode);

        // Act
        var exitCode = await RunCommandAsync(
            testProject.Path,
            "train data.csv --label target"
        );

        // Assert
        Assert.Equal(0, exitCode);
        Assert.True(testProject.HookWasExecuted("pre-train"));
    }

    [Fact]
    public async Task Train_WithoutExtensions_StillWorks()
    {
        // Arrange
        using var testProject = new TempProject();
        // No extension scripts added

        // Act
        var exitCode = await RunCommandAsync(
            testProject.Path,
            "train data.csv --label target"
        );

        // Assert
        Assert.Equal(0, exitCode);  // Training succeeded without extensions
    }

    [Fact]
    public async Task Train_WithCompilationError_ContinuesWithAutoML()
    {
        // Arrange
        using var testProject = new TempProject();
        await testProject.AddHookScript("pre-train.cs", InvalidCSharpCode);

        // Act
        var exitCode = await RunCommandAsync(
            testProject.Path,
            "train data.csv --label target"
        );

        // Assert
        Assert.Equal(0, exitCode);  // AutoML continued despite hook failure
        Assert.Contains("Warning: Compilation failed", testProject.Output);
    }
}
```

---

## 10. Best Practices

### 10.1 Hook Design

**✅ Good: Single Responsibility**
```csharp
// Focused on one task
public class DataValidationHook : IMLoopHook
{
    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        // Only validate data
        if (InvalidData())
            return HookResult.Abort("Data invalid");

        return HookResult.Continue();
    }
}
```

**❌ Bad: Too Many Responsibilities**
```csharp
public class DoEverythingHook : IMLoopHook
{
    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        ValidateData();         // Validation
        PreprocessData();       // Preprocessing
        LogToMLflow();          // Logging
        DeployModel();          // Deployment
        SendEmail();            // Notification
        // Too much! Split into multiple hooks
    }
}
```

### 10.2 Error Handling

**✅ Good: Graceful Degradation**
```csharp
public async Task<HookResult> ExecuteAsync(HookContext ctx)
{
    try
    {
        await LogToExternalService();
        return HookResult.Continue();
    }
    catch (Exception ex)
    {
        ctx.Logger.Warning($"Logging failed: {ex.Message}");
        // Don't fail training due to logging issue
        return HookResult.Continue();
    }
}
```

**❌ Bad: Unhandled Exceptions**
```csharp
public async Task<HookResult> ExecuteAsync(HookContext ctx)
{
    await LogToExternalService();  // Might throw!
    return HookResult.Continue();
    // If external service fails, entire training fails
}
```

### 10.3 Metric Design

**✅ Good: Clear Business Logic**
```csharp
public class ProfitMetric : IMLoopMetric
{
    // Clear business parameters
    private const double PROFIT_PER_TP = 100.0;
    private const double LOSS_PER_FP = -50.0;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(...);

        var profit = (metrics.PositiveRecall * PROFIT_PER_TP) +
                     (metrics.FalsePositiveRate * LOSS_PER_FP);

        // Log for transparency
        ctx.Logger.Info($"Expected profit: ${profit:F2}");

        return profit;
    }
}
```

**❌ Bad: Magic Numbers**
```csharp
public class WeirdMetric : IMLoopMetric
{
    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(...);

        // What do these numbers mean?
        return metrics.Accuracy * 123.45 - 67.89;
    }
}
```

### 10.4 Performance Optimization

**✅ Good: Efficient Data Access**
```csharp
public async Task<HookResult> ExecuteAsync(HookContext ctx)
{
    // Sample for validation (fast)
    var preview = ctx.DataView.Preview(maxRows: 100);

    var stats = CalculateStats(preview);
    return HookResult.Continue();
}
```

**❌ Bad: Full Data Loading**
```csharp
public async Task<HookResult> ExecuteAsync(HookContext ctx)
{
    // Loads entire dataset into memory!
    var allData = ctx.DataView.Preview(maxRows: 1000000);

    var stats = CalculateStats(allData);
    return HookResult.Continue();
}
```

---

## Appendix A: CLI Reference

### Extension Management Commands

```bash
# Create new extension
mloop new hook --name MyHook --type pre-train
mloop new metric --name MyMetric

# Validate extension
mloop validate .mloop/scripts/hooks/pre-train.cs
# ✅ Compilation successful
# ✅ Implements IMLoopHook
# ✅ No warnings

# List extensions
mloop extensions list
# Hooks:
#   ✅ pre-train.cs (Data Validation)
#   ✅ post-train.cs (MLflow Logging)
# Metrics:
#   ✅ profit-metric.cs (Expected Profit)

# Extension info
mloop extensions info pre-train.cs
# Name: Data Validation
# Type: Hook (pre-train)
# Compiled: .mloop/.cache/scripts/hooks.pre-train.dll
# Last modified: 2024-01-15 10:30:45

# Clean cache
mloop extensions clean
# Removed 5 cached DLLs
```

### Training with Extensions

```bash
# Auto-discovery (if .mloop/scripts/ exists)
mloop train data.csv --label target

# Force disable extensions
mloop train data.csv --label target --no-extensions

# Force enable extensions
mloop train data.csv --label target --use-extensions

# Use specific scripts
mloop train data.csv --label target \
  --hook .mloop/scripts/hooks/custom.cs \
  --metric .mloop/scripts/metrics/profit.cs
```

---

## Appendix B: Configuration

### mloop.yaml

```yaml
# Extension configuration
extensions:
  enabled: true  # Set false to disable all extensions

  hooks:
    enabled: true
    only:  # Optional: only run these hooks
      - pre-train
      - post-train

  metrics:
    enabled: true
```

---

## Appendix C: NuGet Package

```
MLoop.Extensibility v1.0.0

Interfaces:
- IMLoopHook
- IMLoopMetric

Context Classes:
- HookContext
- MetricContext
- HookResult

Dependencies:
- Microsoft.ML >= 4.0.0
- .NET 8.0+
```

**Installation:**
```bash
dotnet new classlib -n MyMLoopExtensions
cd MyMLoopExtensions
dotnet add package MLoop.Extensibility
```

---

**Version**: 0.2.0-alpha
**Last Updated**: 2025-01-09
**Status**: Design Specification (Phase 1)
