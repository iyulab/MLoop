using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A binary label written as numbers other than 0/1 (<c>0.0</c>/<c>1.0</c> from a dataframe export, <c>1</c>/<c>2</c>) was inferred as a
/// number, and no binary trainer takes one: every trainer failed its first trials and the run stopped with "expected Boolean, got
/// Single". Read as its two spellings, as OK/NG are, it trains — and every reader after training has to read it the same way: the
/// saved schema records the two values, <c>predict</c> answers in them, and <c>evaluate</c> scores against the label as the data
/// wrote it (before the readers agreed, evaluate read every actual label as the second class: accuracy 0.55 on a model that
/// was right 94 % of the time, its confusion table beside it correct).
/// </summary>
[Collection("FileSystem")]
public class NumericBinaryLabelSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-numeric-binary-tests", Guid.NewGuid().ToString("N"));

    public NumericBinaryLabelSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Theory]
    [InlineData("0.0", "1.0")]
    [InlineData("1", "2")]
    public async Task EveryReader_ReadsANumericBinaryLabelAsItsTwoSpellings(string negative, string positive)
    {
        var csv = WriteTable(negative, positive);
        var (modelPath, schema) = await TrainAsync(csv);

        var label = schema.Columns.Single(c => c.Name == "label");
        Assert.Equal(SchemaDataTypes.Categorical, label.DataType);
        Assert.Equal(new[] { negative, positive }, label.CategoricalValues);

        // mloop predict answers in the values the data wrote
        var output = Path.Combine(_tempDir, "out.csv");
        await new PredictionEngine().PredictAsync(modelPath, csv, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "label", preserveColumns: null, interval: null, taskType: "binary-classification");
        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        var predictedIndex = Array.IndexOf(header, "PredictedLabel");
        var predicted = File.ReadLines(output).Skip(1).Select(l => l.Split(',')[predictedIndex]).ToList();
        var actual = File.ReadLines(csv).Skip(1).Select(l => l.Split(',')[^1]).ToList();
        var agreement = predicted.Zip(actual).Count(p => p.First == p.Second) / (double)actual.Count;
        Assert.True(agreement > 0.9, $"predict agrees with the label on {agreement:P0}");

        // mloop evaluate scores the model against the same label (its accuracy tells which class it read)
        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, csv, "label", "binary-classification", trainedSchema: schema);
        Assert.True(evaluation.Metrics["accuracy"] > 0.9, $"evaluate accuracy {evaluation.Metrics["accuracy"]:F3} (predict agreement {agreement:P0})");
    }

    private string WriteTable(string negative, string positive)
    {
        var path = Path.Combine(_tempDir, "train.csv");
        var random = new Random(3);
        var lines = new List<string> { "a,b,c,label" };
        for (var i = 0; i < 400; i++)
        {
            var a = random.NextDouble();
            var positiveRow = a + 0.2 * random.NextDouble() > 0.6;
            lines.Add(string.Join(",",
                a.ToString("F4", CultureInfo.InvariantCulture),
                random.NextDouble().ToString("F4", CultureInfo.InvariantCulture),
                random.NextDouble().ToString("F4", CultureInfo.InvariantCulture),
                positiveRow ? positive : negative));
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
                ModelName = "numeric",
                DataFile = csv,
                LabelColumn = "label",
                Task = "binary-classification",
                TimeLimitSeconds = 60,
                MaxTrials = 6,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("numeric", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
