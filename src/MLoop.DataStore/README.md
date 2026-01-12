# MLoop.DataStore

Prediction logging, feedback collection, and data sampling for MLOps.

## Features

- **Prediction Logging**: JSONL-based prediction history
- **Feedback Collection**: Ground truth recording for model monitoring
- **Data Sampling**: Create retraining datasets from predictions + feedback
- **Metrics Calculation**: Accuracy metrics from feedback data

## Installation

```bash
dotnet add package MLoop.DataStore
```

## Usage

```csharp
using MLoop.DataStore;

// Prediction Logging
var logger = new FilePredictionLogger(projectRoot);
await logger.LogAsync(new PredictionLogEntry
{
    ModelName = "churn-model",
    Input = inputData,
    Prediction = "Churned",
    Confidence = 0.87
});

// Feedback Collection
var collector = new FileFeedbackCollector(projectRoot);
await collector.RecordFeedbackAsync(new FeedbackEntry
{
    PredictionId = predictionId,
    ActualValue = "NotChurned"
});

// Data Sampling
var sampler = new FileDataSampler(projectRoot, collector, logger);
var samples = await sampler.CreateSampleAsync("churn-model", new SamplingConfig
{
    MaxSize = 1000,
    Strategy = SamplingStrategy.FeedbackPriority
});
```

## Requirements

- .NET 10.0+
- MLoop.Core

## License

MIT License
