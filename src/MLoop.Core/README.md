# MLoop.Core

The in-process library behind the `mloop` CLI: trainer-search training on ML.NET, prediction from a saved
model, data loading with Korean-encoding detection, and the evaluation rules (quality gate, metric
names and directions) the CLI applies.

Most users want the CLI (`dotnet tool install -g mloop`) or its HTTP server (`mloop serve`): they
keep MLoop in its own process and add the project layout, experiment tracking and promotion. Use
this package when a host must call the engine in process and work with typed results. It is
versioned with the CLI.

## Installation

```bash
dotnet add package MLoop.Core
```

## Usage

```csharp
using Microsoft.ML;
using MLoop.Core.Training;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

var ml = new MLContext(seed: 42);

// Train: trainer search within the time budget, evaluated on a held-out split.
var runner = new TrainingRunner(ml, new CsvDataLoader(ml));
var result = await runner.RunAsync(new TrainingConfig
{
    ModelName = "default",
    DataFile = "train.csv",
    LabelColumn = "Target",
    Task = "binary-classification",
    TimeLimitSeconds = 60,
});
Console.WriteLine($"{result.BestTrainer}: {string.Join(", ", result.Metrics)}");

// Save, then predict rows against the saved model and the schema training captured.
ml.Model.Save(result.Model, inputSchema: null, "model.zip");
var predictions = new PredictionService(ml).Predict(
    [new Dictionary<string, object> { ["Age"] = 42f, ["Income"] = 51000f }],
    result.Schema!,
    "model.zip",
    "binary-classification",
    labelColumn: "Target");

foreach (var row in predictions.Rows)
    Console.WriteLine($"{row.PredictedLabel} {row.Score}");
```

`TrainingRunner` reports progress through `IProgress<TrainingProgress>` and takes a
`CancellationToken`. Its optional `warningSink` and `narrationSink` receive the warnings and progress text
the CLI prints.

## Requirements

- .NET 10.0+
- ML.NET 5.0+
- Time-series tasks (forecasting, time-series anomaly detection) need ML.NET's MKL native library,
  which has no osx-arm64 build; `TimeSeriesNativeSupport.IsAvailable` says whether it loaded.

## License

MIT License
