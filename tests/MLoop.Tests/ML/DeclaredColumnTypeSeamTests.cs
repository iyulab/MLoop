using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A column the project declares <c>type: categorical</c> (<c>columns:</c> in mloop.yaml) that holds integers — a department code, a role
/// id — is a category, not a quantity. Inference reads it as a number and folds it into the merged numeric vector, where a role is looked
/// up by column name and finds nothing: the declaration reached no model, while the saved schema recorded it, so every reader after
/// training loaded the column as text and met a vector of another width than the one the model was fitted on
/// ("Feature vector dimension mismatch"; "Could not find input column 'Features'"). These tests train with the declaration and require
/// every reader to answer, and the declaration to have changed what the model learned.
/// </summary>
[Collection("FileSystem")]
public class DeclaredColumnTypeSeamTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-declared-type-seam-tests", Guid.NewGuid().ToString("N"));

    public DeclaredColumnTypeSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task Every_reader_answers_by_a_declared_category_held_in_integers()
    {
        var csv = WriteTable("train.csv", 600);
        var (modelPath, schema) = await TrainAsync(csv, new Dictionary<string, string> { ["dept"] = "categorical" });

        // The declaration was read as one: the schema records the column as a category with its values.
        var dept = schema.Columns.Single(c => c.Name == "dept");
        Assert.Equal(SchemaDataTypes.Categorical, dept.DataType);

        var expected = File.ReadLines(csv).Skip(1).Select(l => l.Split(',')[^1]).ToList();

        // mloop evaluate
        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, csv, "label", "binary-classification", trainedSchema: schema);

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

        // The label is a rule over the codes that no ordering of them carries; a model that read the column as a quantity cannot reach it.
        Assert.True(Agreement(filed, expected) > 0.9 && Agreement(servedLabels, expected) > 0.9 && evaluation.Metrics["auc"] > 0.9,
            $"agreement with the rule over the codes — file predict {Agreement(filed, expected):P0}, serve {Agreement(servedLabels, expected):P0}; evaluate auc {evaluation.Metrics["auc"]:F3}");
    }

    // The declared column was one of two numbers the loader folded into one vector: what is left is a single feature, which every reader
    // names by its header, and the model must be saved to read it so (it was saved to read 'Features', which nothing builds).
    [Fact]
    public async Task A_declared_column_that_leaves_one_numeric_feature_leaves_it_under_its_own_name()
    {
        var csv = WriteTable("two-numbers.csv", 600, withNoise: false);
        var (modelPath, schema) = await TrainAsync(csv, new Dictionary<string, string> { ["dept"] = "categorical" });

        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, csv, "label", "binary-classification", trainedSchema: schema);
        Assert.True(evaluation.Metrics["auc"] > 0.9, $"evaluate auc {evaluation.Metrics["auc"]:F3}");

        var rows = PredictionEngine.LoadCsvAsRows(csv, schema, "label");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "binary-classification", "label");
        Assert.NotEmpty(served.Rows);
    }

    // The loader's half of it: a declared column inside the merged range comes out of it as a column of its own, read as text.
    [Fact]
    public void A_declared_column_leaves_the_merged_numeric_range_and_loads_as_text()
    {
        var csv = WriteTable("loader.csv", 100);
        var loader = new CsvDataLoader(new Microsoft.ML.MLContext(seed: 1));

        var plain = loader.LoadData(csv, "label", "binary-classification");
        Assert.DoesNotContain(plain.Schema, c => c.Name == "dept");

        var declared = loader.LoadDataWithColumnKinds(csv, "label", "binary-classification", null, null, null,
            new Dictionary<string, Microsoft.ML.Data.DataKind> { ["dept"] = Microsoft.ML.Data.DataKind.String });
        var column = declared.Schema.GetColumnOrNull("dept");
        Assert.NotNull(column);
        Assert.Equal(Microsoft.ML.Data.TextDataViewType.Instance, column!.Value.Type);
        Assert.Null(loader.GetMergedColumnGroups()?.Values.FirstOrDefault(g => g.Contains("dept")));
    }

    [Theory]
    [InlineData("categorical", true)]
    [InlineData("Text", true)]
    [InlineData(" categorical ", true)]
    [InlineData("numeric", false)]
    [InlineData("ignore", false)]
    public void Only_text_and_category_declarations_change_how_a_column_loads(string declared, bool changes)
    {
        var kinds = MLoop.Core.Training.TrainingRunner.DeclaredTextKinds(new Dictionary<string, string> { ["dept"] = declared });
        Assert.Equal(changes, kinds is not null && kinds.ContainsKey("dept"));
    }

    private static double Agreement(IReadOnlyList<string> a, IReadOnlyList<string> b)
        => a.Zip(b).Count(p => p.First == p.Second) / (double)b.Count;

    // label = the department code is one of every third code; x, y and amount are noise. dept, amount are integers, x and y are not.
    private string WriteTable(string name, int rows, bool withNoise = true)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(11);
        var lines = new List<string> { withNoise ? "x,dept,y,amount,label" : "x,dept,label" };
        for (var i = 0; i < rows; i++)
        {
            var dept = 100 + rnd.Next(40);
            var x = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            var y = rnd.NextDouble().ToString("F3", CultureInfo.InvariantCulture);
            var label = dept % 3 == 0 ? "a" : "b";
            lines.Add(withNoise ? $"{x},{dept},{y},{rnd.Next(1000)},{label}" : $"{x},{dept},{label}");
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv, Dictionary<string, string> declared)
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
                ModelName = "declared",
                DataFile = csv,
                LabelColumn = "label",
                Task = "binary-classification",
                TimeLimitSeconds = 60,
                MaxTrials = 10,
                ColumnOverrides = declared,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }
        var modelPath = Path.Combine(store.GetExperimentPath("declared", result.ExperimentId), ExperimentLayout.ModelFileName);
        return (modelPath, result.Schema!);
    }
}
