using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// The loader merges a file's numeric columns into the model's feature vector by position. A file with the same columns in
/// another order - an export from another system, the label moved to the front - fed every slot the wrong column, and the
/// model answered without a word (116 of 120 regression answers differed from the same rows in the training order). These
/// tests train on a table where the label depends on one column and require the file predict, the evaluation and the serving
/// path to answer a column-reversed copy exactly as they answer the original.
/// </summary>
[Collection("FileSystem")]
public class ColumnOrderSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-column-order-seam-tests", Guid.NewGuid().ToString("N"));

    public ColumnOrderSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task Every_reader_answers_a_file_whose_columns_are_in_another_order_as_it_answers_the_original()
    {
        var original = Path.Combine(_tempDir, "original.csv");
        var rnd = new Random(11);
        var lines = new List<string> { "a,b,c,y" };
        for (var i = 0; i < 300; i++)
        {
            var a = rnd.NextDouble() * 4 - 2;
            var b = rnd.NextDouble() * 4 - 2;
            var c = rnd.NextDouble() * 4 - 2;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{a:F3},{b:F3},{c:F3},{3 * a - 2 * b + 0.5 * c:F3}"));
        }
        File.WriteAllLines(original, lines);
        var reversed = Path.Combine(_tempDir, "reversed.csv");
        File.WriteAllLines(reversed, lines.Select(l => string.Join(',', l.Split(',').Reverse())));
        var (modelPath, schema) = await TrainAsync(original);

        // mloop predict on a file
        List<string> Filed(string csv, string name)
        {
            var output = Path.Combine(_tempDir, name);
            new PredictionEngine().PredictAsync(modelPath, csv, output, schema, CategoricalMapper.UnknownValueStrategy.Auto,
                cancellationToken: default, labelColumnOverride: "y", preserveColumns: null, interval: null, taskType: "regression").GetAwaiter().GetResult();
            var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
            var scoreIndex = Array.IndexOf(header, "Score");
            return [.. File.ReadLines(output).Skip(1).Select(l => l.Split(',')[scoreIndex])];
        }
        var fromOriginal = Filed(original, "o.csv");
        var fromReversed = Filed(reversed, "r.csv");
        Assert.Equal(fromOriginal, fromReversed);

        // mloop evaluate
        var evaluate = (await new EvaluationEngine().EvaluateAsync(modelPath, original, "y", "regression", trainedSchema: schema)).Metrics;
        var evaluateReversed = (await new EvaluationEngine().EvaluateAsync(modelPath, reversed, "y", "regression", trainedSchema: schema)).Metrics;
        Assert.Equal(evaluate["r_squared"], evaluateReversed["r_squared"], 9);

        // The serving path reads by name already; it must agree with the file path on the reversed file.
        var rows = PredictionEngine.LoadCsvAsRows(reversed, schema, "y");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "regression", "y");
        var servedScores = served.Rows.Select(r => r.Score!.Value).ToList();
        Assert.Equal(fromReversed.Select(s => double.Parse(s, CultureInfo.InvariantCulture)), servedScores, new ScoreComparer());
    }

    private sealed class ScoreComparer : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= 1e-4 * Math.Max(1, Math.Abs(x));
        public int GetHashCode(double value) => 0;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv)
    {
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
        var originalOut = Console.Out;
        Console.SetOut(new StringWriter());
        TrainingResult result;
        try
        {
            result = await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = "order",
                DataFile = csv,
                LabelColumn = "y",
                Task = "regression",
                TimeLimitSeconds = 30,
                MaxTrials = 4,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("order", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
