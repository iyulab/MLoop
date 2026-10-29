using System.Globalization;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Preprocessing;
using MLoop.Core.Storage;
using MLoop.Core.Training;

namespace MLoop.Tests.ML;

/// <summary>
/// A column a prep step declares (<c>mloop prep plan --set fill-missing:mean --columns x</c>, or a normalize) is
/// handled by a featurizer that reads it by its own name, so training loads it as a column of its own rather than
/// as a slot of the merged feature vector. The saved schema did not say so, and every reader after training —
/// <c>mloop evaluate</c>, <c>mloop predict</c> on a file, <c>threshold</c>, <c>calibrate</c>, <c>export onnx</c>
/// — merged it into the vector again and failed with "Could not find input column 'x'". Training itself only
/// warned that it could not profile the model's answers. These tests train with such a declaration and require
/// every reader to answer by the column.
/// </summary>
[Collection("FileSystem")]
public class PrepFeaturizerColumnSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-prep-seam-tests", Guid.NewGuid().ToString("N"));

    public PrepFeaturizerColumnSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Theory]
    [InlineData("fill-missing", "mean")]
    [InlineData("normalize", "min-max")]
    public async Task Every_reader_answers_by_a_column_a_prep_step_declares(string type, string method)
    {
        var csv = WriteTable("train.csv", 300);
        var (modelPath, schema) = await TrainAsync(csv, type, method);

        Assert.True(schema.Columns.Single(c => c.Name == "x").ReadByName, "the schema records that the model reads x by its own name");
        Assert.All(schema.Columns.Where(c => c.Name != "x"), c => Assert.False(c.ReadByName));

        var expected = File.ReadLines(csv).Skip(1).Select(l => l.Split(',')[^1]).ToList();

        // mloop predict on a file
        var output = Path.Combine(_tempDir, "out.csv");
        await new PredictionEngine().PredictAsync(modelPath, csv, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "label", preserveColumns: null, interval: null, taskType: "binary-classification");
        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        var predictedIndex = Array.IndexOf(header, "PredictedLabel");
        var filed = File.ReadLines(output).Skip(1).Select(l => l.Split(',')[predictedIndex]).ToList();

        // The serving path, on rows
        var rows = PredictionEngine.LoadCsvAsRows(csv, schema, "label");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "binary-classification", "label");
        var servedLabels = served.Rows.Select(r => r.PredictedLabel ?? "").ToList();

        // mloop evaluate
        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, csv, "label", "binary-classification",
            trainedSchema: schema);

        Assert.True(Agreement(filed, expected) > 0.95 && Agreement(servedLabels, expected) > 0.95 && evaluation.Metrics["auc"] > 0.95,
            $"agreement with the label — file predict {Agreement(filed, expected):P0}, serve {Agreement(servedLabels, expected):P0}; evaluate auc {evaluation.Metrics["auc"]:F3}");
    }

    // A schema written before the flag existed has no ReadByName; reading it must still work as before.
    [Fact]
    public void A_schema_without_the_flag_reads_as_not_read_by_name()
    {
        var json = """{"columns":[{"name":"x","dataType":"Numeric","purpose":"Feature"}],"capturedAt":"2026-01-01T00:00:00Z"}""";
        var schema = System.Text.Json.JsonSerializer.Deserialize<InputSchemaInfo>(json,
            new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web));

        Assert.False(schema!.Columns.Single().ReadByName);
    }

    // The flag round-trips through the saved schema and is not written for the columns that lack it.
    [Fact]
    public void The_flag_is_written_only_where_it_is_set()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns =
            [
                new ColumnSchema { Name = "x", DataType = SchemaDataTypes.Numeric, Purpose = "Feature", ReadByName = true },
                new ColumnSchema { Name = "y", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            ],
        };

        var json = System.Text.Json.JsonSerializer.Serialize(schema, new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web));

        Assert.Single(System.Text.RegularExpressions.Regex.Matches(json, "readByName"));
        var back = System.Text.Json.JsonSerializer.Deserialize<InputSchemaInfo>(json, new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web))!;
        Assert.True(back.Columns.Single(c => c.Name == "x").ReadByName);
    }

    private static double Agreement(IReadOnlyList<string> a, IReadOnlyList<string> b)
        => a.Zip(b).Count(p => p.First == p.Second) / (double)b.Count;

    // label = x > 0.5; y and z are noise.
    private string WriteTable(string name, int rows)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(11);
        var lines = new List<string> { "x,y,z,label" };
        for (var i = 0; i < rows; i++)
        {
            var x = rnd.NextDouble();
            var y = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            var z = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            lines.Add($"{x.ToString("F3", CultureInfo.InvariantCulture)},{y},{z},{(x > 0.5 ? "a" : "b")}");
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv, string type, string method)
    {
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
        var steps = new List<PrepStep> { new() { Type = type, Method = method, Columns = ["x"] } };
        var originalOut = Console.Out;
        Console.SetOut(new StringWriter());
        TrainingResult result;
        try
        {
            result = await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = "prep",
                DataFile = csv,
                LabelColumn = "label",
                Task = "binary-classification",
                TimeLimitSeconds = 60,
                MaxTrials = 10,
                PreFeaturizer = new PrepFeaturizerBuilder().Build(new MLContext(seed: 1), steps),
                PreFeaturizerColumns = PrepFeaturizerBuilder.ResolvePreFeaturizerColumns(steps).ToList(),
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("prep", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
