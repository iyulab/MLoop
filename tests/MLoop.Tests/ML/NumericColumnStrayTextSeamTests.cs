using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A numeric column that holds a few stray strings (<c>N/A</c>, <c>--</c>) in the file being scored is read as
/// text by the layout inference, which leaves one numeric column — and a group of one keeps its header name
/// instead of forming a vector. A model trained on two numeric features then found no <c>Features</c> column and
/// <c>mloop evaluate</c>, <c>mloop explain</c> and <c>mloop predict</c> on a file failed with
/// <c>Could not find input column 'Features'</c>, while the same file on rows (<c>--json</c>, the serving path)
/// scored. A larger file hid it: inference types a column from its first 10,000 lines, so strings further down
/// were never seen. These tests score one small file through every reader and require the same answers.
/// </summary>
[Collection("FileSystem")]
public class NumericColumnStrayTextSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-stray-text-seam-tests", Guid.NewGuid().ToString("N"));

    public NumericColumnStrayTextSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task Every_reader_scores_a_small_file_whose_numeric_column_holds_stray_text()
    {
        var (modelPath, schema) = await TrainAsync(WriteTable("train.csv", 300, strayEvery: 0));
        var stray = WriteTable("stray.csv", 120, strayEvery: 12);

        // mloop evaluate
        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, stray, "y", "regression", trainedSchema: schema);
        Assert.True(evaluation.Metrics["r_squared"] > 0.5, $"evaluate r_squared {evaluation.Metrics["r_squared"]:F3}");

        // mloop predict on a file
        var output = Path.Combine(_tempDir, "out.csv");
        await new PredictionEngine().PredictAsync(modelPath, stray, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "y", preserveColumns: null, interval: null, taskType: "regression");
        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        var scoreIndex = Array.IndexOf(header, "Score");
        Assert.True(scoreIndex >= 0, string.Join(",", header));
        var filed = File.ReadLines(output).Skip(1)
            .Select(l => double.Parse(l.Split(',')[scoreIndex], CultureInfo.InvariantCulture)).ToList();

        // the serving path, on rows
        var rows = PredictionEngine.LoadCsvAsRows(stray, schema, "y");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "regression", "y");
        var servedScores = served.Rows.Select(r => r.Score ?? double.NaN).ToList();

        // One file, two readers, one answer: a stray string is a missing value in both.
        Assert.Equal(120, filed.Count);
        Assert.Equal(filed.Count, servedScores.Count);
        var worst = filed.Zip(servedScores).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < 1e-3, $"file predict and serve differ by up to {worst:G4}");
    }

    // y = 3·reading + other; every `strayEvery`-th reading is a string instead of a number.
    private string WriteTable(string name, int rows, int strayEvery)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(7);
        var lines = new List<string> { "reading,other,y" };
        for (var i = 0; i < rows; i++)
        {
            var reading = rnd.NextDouble() * 10;
            var other = rnd.NextDouble() * 5;
            var y = 3 * reading + other + (rnd.NextDouble() - 0.5) * 0.3;
            var text = strayEvery > 0 && i % strayEvery == 0
                ? (i % (2 * strayEvery) == 0 ? "N/A" : "--")
                : reading.ToString("F3", CultureInfo.InvariantCulture);
            lines.Add($"{text},{other.ToString("F3", CultureInfo.InvariantCulture)},{y.ToString("F3", CultureInfo.InvariantCulture)}");
        }
        File.WriteAllLines(path, lines);
        return path;
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
                ModelName = "stray",
                DataFile = csv,
                LabelColumn = "y",
                Task = "regression",
                TimeLimitSeconds = 60,
                MaxTrials = 10,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("stray", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
