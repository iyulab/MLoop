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

A hook lives in `.mloop/scripts/hooks/<stage>/` (`pre-train`, `post-train`, `pre-predict`, `post-evaluate`).

```csharp
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Validation";

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        // The pre-train stage reads a file: its path is in the hook's metadata.
        var dataFile = ctx.GetMetadata<string>("DataFile", "");
        var rowCount = File.Exists(dataFile)
            ? File.ReadLines(dataFile).Count(line => !string.IsNullOrWhiteSpace(line)) - 1   // minus the header
            : 0;

        if (rowCount < 100)
            return Task.FromResult(HookResult.Abort($"Insufficient data: {rowCount} rows, need at least 100"));

        ctx.Logger.Info($"Data validation passed: {rowCount} rows");
        return Task.FromResult(HookResult.Continue());
    }
}
```

### Create a Custom Metric

A metric lives in `.mloop/scripts/metrics/`. It is computed over the held-out rows after training and recorded with the run's metrics — a report, not an optimization target.

```csharp
using System;
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public string Description => "100 per true positive, -50 per false positive";

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        var predicted = (bool[])ctx.Predictions;   // binary-classification: bool[]
        var actual = (bool[])ctx.Labels;
        double profit = 0;
        for (var i = 0; i < predicted.Length; i++)
            if (predicted[i]) profit += actual[i] ? 100 : -50;
        return Task.FromResult(new MetricResult { Name = Name, Value = profit });
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
- [User guide — hooks, metrics and preprocessing scripts](https://github.com/iyulab/MLoop/blob/main/docs/GUIDE.md)
- [Examples](https://github.com/iyulab/MLoop/tree/main/examples/extensions)

## License

MIT
