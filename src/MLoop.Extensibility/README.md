# MLoop.Extensibility

Optional extensibility API for MLoop - enables code-based customization of AutoML pipeline.

## Overview

MLoop.Extensibility provides interfaces for extending MLoop's AutoML capabilities through:
- **Hooks**: Execute custom logic at lifecycle points (pre-train, post-train, etc.)
- **Custom Metrics**: Optimize for business outcomes instead of generic accuracy

## Features

- ✅ **Completely Optional**: Works without any extensions
- ✅ **Zero-Overhead**: < 1ms impact when not used
- ✅ **Type-Safe**: Full C# type system with IDE support
- ✅ **Auto-Discovery**: Convention-based discovery from filesystem

## Quick Start

### Install Package

```bash
dotnet add package MLoop.Extensibility
```

### Create a Hook

```csharp
using MLoop.Extensibility;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Validation";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var rowCount = ctx.DataView.Preview().RowView.Length;

        if (rowCount < 100)
        {
            return HookResult.Abort("Need at least 100 rows");
        }

        ctx.Logger.Info($"✅ Validation passed: {rowCount} rows");
        return HookResult.Continue();
    }
}
```

### Create a Custom Metric

```csharp
using MLoop.Extensibility;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public bool HigherIsBetter => true;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(ctx.Predictions);

        // Optimize for profit instead of accuracy
        return (metrics.PositiveRecall * 100.0) +  // $100 per True Positive
               (metrics.FalsePositiveRate * -50.0); // -$50 per False Positive
    }
}
```

## API Reference

### Interfaces

- `IMLoopHook` - Lifecycle hook interface
- `IMLoopMetric` - Custom metric interface

### Context Classes

- `HookContext` - Provides data and ML context to hooks
- `MetricContext` - Provides predictions for metric calculation
- `HookResult` - Return value for hooks (Continue or Abort)

### Logger

- `ILogger` - Logging interface for hook output

## Documentation

For complete documentation, see:
- [EXTENSIBILITY.md](https://github.com/iyulab/MLoop/blob/main/docs/EXTENSIBILITY.md)
- [API Reference](https://github.com/iyulab/MLoop/blob/main/docs/API_REFERENCE.md)

## License

MIT
