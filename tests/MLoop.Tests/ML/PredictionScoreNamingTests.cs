using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// "score" named two values: the structured paths (<c>--json</c>, <c>POST /predict</c>) answer the positive-class
/// probability under it, while a binary classifier's prediction CSV wrote the trainer's raw margin under
/// <c>Score</c> (−3.43 beside a probability of 0.335) and the probability under <c>Probability</c>. A caller
/// moving from one path to the other read the wrong number without an error. The CSV now calls the margin
/// <c>RawScore</c>; a regression's <c>Score</c> is its prediction, the same on every path, and is left alone.
/// </summary>
[Collection("FileSystem")]
public class PredictionScoreNamingTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-score-naming-tests", Guid.NewGuid().ToString("N"));

    public PredictionScoreNamingTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task Binary_csv_names_the_margin_RawScore_and_Probability_is_what_the_structured_path_calls_score()
    {
        var csv = WriteTable("binary.csv", 300, binary: true);
        var (modelPath, schema) = await TrainAsync(csv, "binary-classification", "label");

        var output = Path.Combine(_tempDir, "binary-out.csv");
        await new PredictionEngine().PredictAsync(modelPath, csv, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "label", preserveColumns: null, interval: null, taskType: "binary-classification");

        var lines = File.ReadAllLines(output);
        var header = lines[0].TrimStart('﻿').Split(',');
        Assert.Contains(PredictionEngine.RawScoreColumn, header);
        Assert.Contains("Probability", header);
        Assert.DoesNotContain("Score", header);

        var rawIndex = Array.IndexOf(header, PredictionEngine.RawScoreColumn);
        var probabilityIndex = Array.IndexOf(header, "Probability");
        var rows = lines.Skip(1).Select(l => l.Split(',')).ToList();
        var raw = rows.Select(r => double.Parse(r[rawIndex], CultureInfo.InvariantCulture)).ToList();
        var probability = rows.Select(r => double.Parse(r[probabilityIndex], CultureInfo.InvariantCulture)).ToList();

        var served = new PredictionEngine()
            .PredictWithService(PredictionEngine.LoadCsvAsRows(csv, schema, "label"), schema, modelPath, "binary-classification", "label");
        for (var i = 0; i < rows.Count; i++)
            Assert.Equal(served.Rows[i].Score!.Value, probability[i], 4);

        // A margin runs past [0, 1]; if the two columns were the same number the rename would prove nothing.
        Assert.Contains(raw.Zip(probability), p => Math.Abs(p.First - p.Second) > 0.05);
    }

    [Fact]
    public async Task Regression_csv_keeps_Score_because_it_is_the_prediction()
    {
        var csv = WriteTable("regression.csv", 200, binary: false);
        var (modelPath, schema) = await TrainAsync(csv, "regression", "y");

        var output = Path.Combine(_tempDir, "regression-out.csv");
        await new PredictionEngine().PredictAsync(modelPath, csv, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "y", preserveColumns: null, interval: null, taskType: "regression");

        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        Assert.Contains("Score", header);
        Assert.DoesNotContain(PredictionEngine.RawScoreColumn, header);
    }

    private string WriteTable(string name, int rows, bool binary)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(11);
        var lines = new List<string> { binary ? "x,z,label" : "x,z,y" };
        for (var i = 0; i < rows; i++)
        {
            var x = rnd.NextDouble();
            var z = rnd.NextDouble();
            var answer = binary ? (x + 0.2 * z > 0.6 ? "a" : "b") : (3 * x + z).ToString("F4", CultureInfo.InvariantCulture);
            lines.Add($"{x.ToString("F4", CultureInfo.InvariantCulture)},{z.ToString("F4", CultureInfo.InvariantCulture)},{answer}");
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv, string task, string label)
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
                ModelName = task,
                DataFile = csv,
                LabelColumn = label,
                Task = task,
                TimeLimitSeconds = 10,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath(task, result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
