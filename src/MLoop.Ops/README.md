# MLoop.Ops

MLOps automation for model comparison, retraining triggers, and promotion.

## Features

- **Model Comparison**: Compare experiments by metrics
- **Time-Based Triggers**: Retraining based on time elapsed
- **Feedback-Based Triggers**: Retraining based on accuracy drop or feedback volume
- **Best Model Discovery**: Find best experiment by specified criteria

## Installation

```bash
dotnet add package MLoop.Ops
```

## Usage

### Model Comparison

```csharp
using MLoop.Ops;

var comparer = new FileModelComparer(projectRoot);

// Compare two experiments
var comparison = await comparer.CompareAsync("exp-001", "exp-002");

// Find best experiment by metric
var best = await comparer.FindBestAsync("Accuracy", higherIsBetter: true);
```

### Retraining Triggers

```csharp
using MLoop.Ops;

// Time-based trigger
var timeTrigger = new TimeBasedTrigger(projectRoot);
var shouldRetrain = await timeTrigger.EvaluateAsync("model", new[]
{
    new RetrainingCondition(ConditionType.TimeBased, "days", 7)
});

// Feedback-based trigger
var feedbackTrigger = new FeedbackBasedTrigger(feedbackCollector);
var result = await feedbackTrigger.EvaluateAsync("model", new[]
{
    new RetrainingCondition(ConditionType.AccuracyDrop, "threshold", 0.8),
    new RetrainingCondition(ConditionType.FeedbackVolume, "count", 100)
});

if (result.ShouldRetrain)
{
    // Trigger retraining
}
```

## Requirements

- .NET 10.0+
- MLoop.Core
- MLoop.DataStore

## License

MIT License
