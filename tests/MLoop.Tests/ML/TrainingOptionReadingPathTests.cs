using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Export;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;
using MLoop.Core.Training;

namespace MLoop.Tests.ML;

/// <summary>
/// What a training option does to the feature set is read again by every path that loads a file for the model —
/// <c>mloop predict</c> on a file, serving, <c>mloop evaluate</c>, and the threshold and calibration rows. A
/// `type: ignore` on a numeric column used to be a training no-op (the loader had merged the column into the
/// <c>Features</c> vector, so no column of its name existed for the role to ignore) while the saved schema marked
/// it excluded and every reader dropped it: the model took a vector one slot wider than any file it was later
/// given, and <c>evaluate</c> / <c>predict</c> failed on a model that had trained and promoted without a word.
/// This matrix trains under each option that decides which columns the model reads, and requires every reader to
/// answer by the column that is left.
/// </summary>
[Collection("FileSystem")]
public class TrainingOptionReadingPathTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-option-reading-tests", Guid.NewGuid().ToString("N"));

    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TrainingOptionReadingPathTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    /// <summary>A training option that decides which columns the model reads, with the data it is exercised on.</summary>
    public sealed record Option(string Name, Dictionary<string, string>? Overrides = null, string? Weight = null,
        bool LeakIsTheLabel = false);

    public static TheoryData<string> OptionNames() => new(Options().Select(o => o.Name));

    // Every row is the same table: x decides the label, leak / noise / tag / flag1 / flag2 / when / w are the columns an
    // option can change the reading of. A row of this list is a way to train; none may leave a reader behind.
    private static IReadOnlyList<Option> Options() =>
    [
        new("no option"),
        new("ignore a numeric column", new() { ["leak"] = "ignore" }, LeakIsTheLabel: true),
        new("ignore a text column", new() { ["tag"] = "ignore" }),
        new("a row weight", Weight: "w"),
        new("a row weight and an ignored numeric column", new() { ["leak"] = "ignore" }, Weight: "w", LeakIsTheLabel: true),
        new("a declared category", new() { ["tag"] = "categorical" }),
    ];

    [Theory]
    [MemberData(nameof(OptionNames))]
    public async Task Every_reader_answers_by_the_columns_the_option_leaves(string optionName)
    {
        var option = Options().Single(o => o.Name == optionName);
        var train = WriteTable("train.csv", 300, seed: 11, option.LeakIsTheLabel, leakAsInTraining: true);
        // The ignored column carries the label in training and is noise in the file read afterwards: a model
        // that read it answers wrongly here, and one that did not never notices.
        var test = WriteTable("test.csv", 150, seed: 23, option.LeakIsTheLabel, leakAsInTraining: false);
        var (modelPath, schema) = await TrainAsync(train, option);

        var expected = File.ReadLines(test).Skip(1).Select(l => l.Split(',')[^1]).ToList();

        // mloop predict on a file
        var output = Path.Combine(_tempDir, "out.csv");
        await new PredictionEngine().PredictAsync(modelPath, test, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "label", preserveColumns: null, interval: null, taskType: "binary-classification");
        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        var predictedIndex = Array.IndexOf(header, "PredictedLabel");
        var filed = File.ReadLines(output).Skip(1).Select(l => l.Split(',')[predictedIndex]).ToList();

        // The serving path, on rows
        var rows = PredictionEngine.LoadCsvAsRows(test, schema, "label");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "binary-classification", "label");
        var servedLabels = served.Rows.Select(r => r.PredictedLabel ?? "").ToList();

        // mloop evaluate
        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, test, "label", "binary-classification",
            trainedSchema: schema);

        // mloop threshold / calibrate
        var thresholdRows = await new EvaluationEngine().ThresholdRowsAsync(modelPath, test, "label", schema);

        Assert.True(Agreement(filed, expected) > 0.9 && Agreement(servedLabels, expected) > 0.9 && evaluation.Metrics["auc"] > 0.9,
            $"{optionName}: agreement with the label — file predict {Agreement(filed, expected):P0}, serve {Agreement(servedLabels, expected):P0}; evaluate auc {evaluation.Metrics["auc"]:F3}");
        Assert.NotNull(thresholdRows);
        Assert.Equal(expected.Count, thresholdRows!.Count);
    }

    // The same options under a multiclass model: the class probabilities the calibration is fit on come through the same loader.
    [Theory]
    [MemberData(nameof(OptionNames))]
    public async Task Every_reader_answers_a_multiclass_model_by_the_columns_the_option_leaves(string optionName)
    {
        var option = Options().Single(o => o.Name == optionName);
        var train = WriteTable("mtrain.csv", 450, seed: 13, option.LeakIsTheLabel, leakAsInTraining: true, classes: 3);
        var test = WriteTable("mtest.csv", 180, seed: 29, option.LeakIsTheLabel, leakAsInTraining: false, classes: 3);
        var (modelPath, schema) = await TrainAsync(train, option, "multiclass-classification");

        var expected = File.ReadLines(test).Skip(1).Select(l => l.Split(',')[^1]).ToList();

        var output = Path.Combine(_tempDir, "mout.csv");
        await new PredictionEngine().PredictAsync(modelPath, test, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken: default,
            labelColumnOverride: "label", preserveColumns: null, interval: null, taskType: "multiclass-classification");
        var header = File.ReadLines(output).First().TrimStart('﻿').Split(',');
        var predictedIndex = Array.IndexOf(header, "PredictedLabel");
        var filed = File.ReadLines(output).Skip(1).Select(l => l.Split(',')[predictedIndex]).ToList();

        var rows = PredictionEngine.LoadCsvAsRows(test, schema, "label");
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, "multiclass-classification", "label");
        var servedLabels = served.Rows.Select(r => r.PredictedLabel ?? "").ToList();

        var evaluation = await new EvaluationEngine().EvaluateAsync(modelPath, test, "label", "multiclass-classification", trainedSchema: schema);
        var calibrationRows = await new EvaluationEngine().MulticlassCalibrationRowsAsync(modelPath, test, "label", schema);

        Assert.True(Agreement(filed, expected) > 0.75 && Agreement(servedLabels, expected) > 0.75,
            $"{optionName}: agreement with the label — file predict {Agreement(filed, expected):P0}, serve {Agreement(servedLabels, expected):P0}");
        Assert.True(evaluation.Metrics.Values.Any(), "evaluate answered with metrics");
        Assert.NotNull(calibrationRows);
        Assert.Equal(expected.Count, calibrationRows!.Count);
    }

    // The graph `mloop export onnx` writes is one more reader of the same columns: it takes the input view the model was
    // trained on and must answer what the ML.NET model answers, option by option. `--verify` runs it in ONNX Runtime and
    // compares; without that runtime the command refuses and says how to install it, and the row is skipped, not passed.
    [Theory]
    [MemberData(nameof(OptionNames))]
    public async Task Export_onnx_agrees_with_a_binary_model_for_each_option(string optionName)
        => await ExportAgreesAsync(optionName, "binary-classification", classes: 2);

    [Theory]
    [MemberData(nameof(OptionNames))]
    public async Task Export_onnx_agrees_with_a_multiclass_model_for_each_option(string optionName)
        => await ExportAgreesAsync(optionName, "multiclass-classification", classes: 3);

    private async Task ExportAgreesAsync(string optionName, string task, int classes)
    {
        var option = Options().Single(o => o.Name == optionName);
        var train = WriteTable("etrain.csv", classes == 2 ? 300 : 450, seed: 17, option.LeakIsTheLabel, leakAsInTraining: true, classes);
        var (modelPath, _) = await TrainAsync(train, option, task);
        var experimentId = Path.GetFileName(Path.GetDirectoryName(modelPath))!;

        Directory.SetCurrentDirectory(_tempDir);
        var (exit, output, error) = await CliRunner.RunAsync(
            "export", "onnx", experimentId, "--name", "option", "--data", train, "--verify", "--json");
        if (exit != 0 && (output + error).Contains("mloop runtime install onnxruntime", StringComparison.Ordinal))
            Assert.Skip("onnxruntime is not installed: the refusal names the install, the graph was not run.");

        Assert.True(exit == 0, $"{optionName} ({task}): export onnx exit {exit}: {output}{error}");
        using var doc = JsonDocument.Parse(output);
        var manifest = OnnxManifest.FromJson(await File.ReadAllTextAsync(doc.RootElement.GetProperty("manifestPath").GetString()!));
        Assert.NotNull(manifest.Verification);
        Assert.True(manifest.Verification!.Passed, $"{optionName} ({task}): the graph disagrees with the model: {manifest.Verification}");
        // ...and it reads the columns the option leaves: an ignored column is no input of the graph, a weight is none either.
        var read = manifest.Inputs.SelectMany(i => i.SourceColumns).ToHashSet();
        foreach (var ignored in option.Overrides?.Where(o => o.Value == "ignore").Select(o => o.Key) ?? [])
            Assert.DoesNotContain(ignored, read);
        if (option.Weight is { } weight)
            Assert.DoesNotContain(weight, read);
    }

    // The matrix is only as complete as its list: a way to train that decides which columns the model reads has to be
    // a row of it. The options the product declares are named here once, and a new one fails this until it has a row.
    [Fact]
    public void The_matrix_covers_every_option_that_changes_what_the_model_reads()
    {
        var covered = Options().SelectMany(o => new[]
        {
            o.Overrides is not null ? "column override" : null,
            o.Weight is not null ? "weight column" : null,
        }).Where(s => s is not null).Distinct().Order().ToList();

        Assert.Equal(["column override", "weight column"], covered);
        // ...and a column override is exercised on a numeric column and on a text one: the first lives inside the
        // merged feature vector and the second does not, which is the whole difference this matrix exists for.
        var ignored = Options().Where(o => o.Overrides is not null).SelectMany(o => o.Overrides!.Keys).Distinct().Order().ToList();
        Assert.Contains("leak", ignored);
        Assert.Contains("tag", ignored);
    }

    private static double Agreement(IReadOnlyList<string> a, IReadOnlyList<string> b)
        => a.Zip(b).Count(p => p.First == p.Second) / (double)b.Count;

    private string WriteTable(string name, int rows, int seed, bool leakIsTheLabel, bool leakAsInTraining, int classes = 2)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(seed);
        var inv = CultureInfo.InvariantCulture;
        var lines = new List<string> { "x,noise,leak,tag,flag1,flag2,when,w,label" };
        for (var i = 0; i < rows; i++)
        {
            var x = rnd.NextDouble();
            var positive = x > 0.5;
            var label = classes == 2 ? (positive ? "a" : "b") : x < 1.0 / 3 ? "a" : x < 2.0 / 3 ? "b" : "c";
            var leak = leakIsTheLabel && leakAsInTraining ? (label == "a" ? 0.0 : label == "b" ? 0.5 : 1.0) : rnd.NextDouble();
            lines.Add(string.Join(',',
                x.ToString("F3", inv),
                rnd.NextDouble().ToString("F3", inv),
                leak.ToString("F3", inv),
                "abc"[rnd.Next(3)],
                rnd.Next(2) == 0 ? "yes" : "no",
                rnd.Next(2) == 0 ? "yes" : "no",
                new DateTime(2026, 1, 1).AddDays(i).ToString("yyyy-MM-dd", inv),
                (0.5 + rnd.NextDouble()).ToString("F3", inv),
                label));
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv, Option option, string task = "binary-classification")
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
                ModelName = "option",
                DataFile = csv,
                LabelColumn = "label",
                Task = task,
                TimeLimitSeconds = 60,
                MaxTrials = 10,
                ColumnOverrides = option.Overrides,
                WeightColumn = option.Weight,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("option", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
