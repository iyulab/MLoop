# MLoop Extension Examples

Working examples of the two kinds of extension script: a **hook** (runs at a stage of the pipeline and can stop it) and a
**custom metric** (a business number computed after training). Both are plain C# files that MLoop compiles when it needs them.

```
examples/extensions/
├── hooks/
│   └── DataValidationHook.cs   # pre-train: refuse a file that is too small, warn on empty columns and one-class labels
└── metrics/
    ├── ProfitMetric.cs         # expected profit of the model's answers (per true / false positive / negative)
    └── ChurnCostMetric.cs      # what a retention programme would cost, per 1,000 customers (lower is better)
```

## Try them

```bash
mloop init my-project --task binary-classification
cd my-project

# A hook goes in the folder of its stage: pre-train/, post-train/, pre-predict/ or post-evaluate/
mkdir -p .mloop/scripts/hooks/pre-train .mloop/scripts/metrics
cp path/to/examples/extensions/hooks/DataValidationHook.cs .mloop/scripts/hooks/pre-train/
cp path/to/examples/extensions/metrics/*.cs .mloop/scripts/metrics/

# Scripts are discovered by location; nothing to switch on
mloop train data.csv --label target
```

A hook that is not in a stage folder (for example `.mloop/scripts/hooks/pre-train.cs`) does not run — `mloop train` and
`mloop validate` say so. `mloop validate -v` compiles every script and reports what failed.

What you see: the hook's lines before training (`DataValidationHook.cs` — `Dataset size: …`, `Validation passed`, or the
abort reason), and after training each metric's value in the results table next to the built-in ones and in the run's
`metrics.json`.

**A custom metric is a report, not an optimization target.** It is computed over the held-out rows after the model is chosen;
the search still optimizes the project's `metric:` in `mloop.yaml` (`--metric` takes a built-in name such as `auc` or
`f1_score`, not a script).

## The contract

```csharp
// Hook — public class with a parameterless constructor
using MLoop.Extensibility.Hooks;
public class MyHook : IMLoopHook
{
    public string Name => "My hook";
    public Task<HookResult> ExecuteAsync(HookContext ctx) => Task.FromResult(HookResult.Continue()); // or HookResult.Abort("why")
}

// Metric
using MLoop.Extensibility.Metrics;
public class MyMetric : IMLoopMetric
{
    public string Name => "My metric";
    public string Description => "what the number means";
    public Task<MetricResult> CalculateAsync(MetricContext ctx) => Task.FromResult(new MetricResult { Name = Name, Value = 0 });
}
```

- A hook reads what the stage has through the context: `ctx.GetMetadata<string>("DataFile", "")` (pre-train: the training file),
  `"LabelColumn"`, the metrics after training, `ctx.Logger` for output.
- A metric gets `ctx.Predictions` and `ctx.Labels`: `bool[]` for binary classification, `int[]` class indices for
  multiclass classification, `float[]` for regression (`ctx.TaskType` names the task).
- Scripts have no implicit `using`s: write `using System; using System.Linq; using System.Threading.Tasks;` as the files here do.

## Adjusting them

Each file keeps its business parameters as constants at the top (`MinRows`, `ProfitPerTruePositive`, `CampaignCost`, …) —
change those to match your domain.

## More

- [User guide — hooks, metrics and preprocessing scripts](../../docs/GUIDE.md)
- [Architecture — the extensibility system](../../docs/ARCHITECTURE.md#14-extensibility-system)
- `mloop new hook <name> <stage>` writes a hook scaffold that compiles as it is.
