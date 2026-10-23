using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A yes/no feature column (<c>yes</c>/<c>no</c>, <c>true</c>/<c>false</c>, …) is read by training's loader as a
/// boolean and turned into a number by the model. The saved schema recorded it as a category, so every reader
/// after training — <c>mloop predict</c> on a file, the serving path on rows, <c>mloop evaluate</c> — loaded
/// it as text, the model's conversion gave a missing value for every row, and the model answered as if the
/// column were absent: a classifier whose only signal was that column gave one answer for every row, with
/// accuracy 1.0 at training and AUC 0.5 at evaluation, and no error anywhere. These tests train on data where
/// the label is the yes/no column and require every reader to answer by it.
/// </summary>
[Collection("FileSystem")]
public class YesNoFeatureSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-yesno-seam-tests", Guid.NewGuid().ToString("N"));

    public YesNoFeatureSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    // One yes/no column keeps its own name in the loaded view; two fold into one boolean vector.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task Every_reader_answers_by_a_yes_no_feature(int yesNoColumns)
    {
        var csv = WriteTable("train.csv", 300, yesNoColumns);
        var (modelPath, schema) = await TrainAsync(csv);

        Assert.Equal(SchemaDataTypes.Boolean, schema.Columns.Single(c => c.Name == "flag").DataType);

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
            $"agreement with the yes/no column — file predict {Agreement(filed, expected):P0}, serve {Agreement(servedLabels, expected):P0}; evaluate auc {evaluation.Metrics["auc"]:F3}");
    }

    [Theory]
    [InlineData("yes", true)]
    [InlineData("No", true)]
    [InlineData("t", true)]
    [InlineData("FALSE", true)]
    [InlineData("maybe", false)]
    public void A_boolean_column_reads_the_values_the_loader_reads(string value, bool reads)
        => Assert.Equal(reads, SchemaDataTypes.ValueReadsAs(SchemaDataTypes.Boolean, value));

    // A schema written before yes/no columns were recorded as Boolean: the values identify the column.
    [Theory]
    [InlineData(new[] { "no", "yes" }, Microsoft.ML.Data.DataKind.Boolean)]
    [InlineData(new[] { "False", "True" }, Microsoft.ML.Data.DataKind.Boolean)]
    [InlineData(new[] { "No", "Yes", "No phone service" }, Microsoft.ML.Data.DataKind.String)]
    [InlineData(new[] { "a", "b" }, Microsoft.ML.Data.DataKind.String)]
    public void An_older_schema_s_yes_no_category_loads_as_a_boolean(string[] values, Microsoft.ML.Data.DataKind kind)
    {
        var column = new ColumnSchema
        {
            Name = "flag", DataType = SchemaDataTypes.Categorical, Purpose = "Feature",
            CategoricalValues = [.. values], UniqueValueCount = values.Length,
        };

        Assert.Equal(kind, SchemaDataTypes.FeatureLoadKind(column, Microsoft.ML.Data.DataKind.String));
    }

    private static double Agreement(IReadOnlyList<string> a, IReadOnlyList<string> b)
        => a.Zip(b).Count(p => p.First == p.Second) / (double)b.Count;

    // label = flag; x and y are noise. With two yes/no columns the second is noise too.
    private string WriteTable(string name, int rows, int yesNoColumns)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(7);
        var lines = new List<string> { yesNoColumns == 2 ? "x,flag,y,other,label" : "x,flag,y,label" };
        for (var i = 0; i < rows; i++)
        {
            var flag = i % 3 == 0;
            var x = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            var y = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            var other = yesNoColumns == 2 ? (rnd.Next(2) == 0 ? ",yes" : ",no") : "";
            lines.Add($"{x},{(flag ? "yes" : "no")},{y}{other},{(flag ? "a" : "b")}");
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
                ModelName = "yesno",
                DataFile = csv,
                LabelColumn = "label",
                Task = "binary-classification",
                TimeLimitSeconds = 10,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("yesno", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
