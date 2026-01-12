# MLoop.Core

ML.NET AutoML engine for training, prediction, and model management.

## Features

- **AutoML Training**: Automated machine learning with ML.NET AutoML
- **Prediction Engine**: Batch and single predictions
- **Encoding Detection**: Automatic charset detection (UTF-8, CP949, EUC-KR)
- **Preprocessing**: Script-based data preprocessing
- **Model Management**: Experiment tracking and promotion

## Installation

```bash
dotnet add package MLoop.Core
```

## Usage

```csharp
using MLoop.Core;

// Training
var engine = new TrainingEngine(projectRoot);
var result = await engine.TrainAsync(new TrainingConfig
{
    DataPath = "data.csv",
    LabelColumn = "Target",
    TaskType = TaskType.BinaryClassification,
    TrainingTime = TimeSpan.FromMinutes(5)
});

// Prediction
var predictor = new PredictionEngine(projectRoot, "default");
var predictions = await predictor.PredictBatchAsync("test.csv");
```

## Requirements

- .NET 10.0+
- ML.NET 5.0+

## License

MIT License
