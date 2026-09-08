using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Drift guard: the two predict paths must name a predicted class the same way, and must name it
/// the way the user's data named it.
/// </summary>
/// <remarks>
/// <para>
/// Binary classification takes a Boolean label, so a label written as <c>OK</c>/<c>NG</c> is
/// converted before training. Nothing undid the conversion afterwards. Measured on a model trained
/// on those two values: the CSV path answered <c>1</c>/<c>0</c> and the structured path answered
/// <c>True</c>/<c>False</c> — the predictions agreed row for row, but the class names the user
/// wrote were absent from both answers, and the two answers did not agree with each other either.
/// The saved schema carried the vocabulary the whole time.
/// </para>
/// <para>
/// The two paths render through separate code — the CSV path through ML.NET's text writer, the
/// structured path through its own extractor — so nothing but a test spanning both can hold them
/// together. <see cref="BinaryLabelVocabulary"/> is the shared answer; this pins that both keep
/// asking it.
/// </para>
/// </remarks>
public class CrossPathLabelRenderingTests : IDisposable
{
    private readonly List<string> _tempDirs = [];

    public void Dispose()
    {
        foreach (var d in _tempDirs)
        {
            try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { }
        }
        GC.SuppressFinalize(this);
    }

    private sealed class Reading
    {
        public float Temp { get; set; }
        public float Press { get; set; }
        public bool Result { get; set; }
    }

    private static InputSchemaInfo SchemaWithVocabulary(List<string>? vocabulary) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "Temp", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "Press", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema
            {
                Name = "Result",
                DataType = SchemaDataTypes.Categorical,
                Purpose = "Label",
                CategoricalValues = vocabulary,
                UniqueValueCount = vocabulary?.Count,
            },
        ],
    };

    /// <param name="vocabulary">null stands for a model that recorded none.</param>
    private async Task<(List<string> Csv, List<string> Structured)> BothPaths(List<string>? vocabulary)
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-label-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);

        var ml = new MLContext(seed: 7);
        var training = Enumerable.Range(0, 60).Select(i =>
        {
            var positive = i % 2 == 0;
            return new Reading
            {
                Temp = positive ? 20f + i % 5 : 36f + i % 5,
                Press = positive ? 1.0f + (i % 5) * 0.05f : 2.2f + (i % 5) * 0.05f,
                Result = positive,
            };
        }).ToList();

        var data = ml.Data.LoadFromEnumerable(training);
        var featurized = ml.Transforms.Concatenate("Features", "Temp", "Press").Fit(data).Transform(data);
        var model = ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Result", featureColumnName: "Features")
            .Fit(featurized);

        var modelPath = Path.Combine(dir, "model.zip");
        ml.Model.Save(model, featurized.Schema, modelPath);

        var predictCsv = Path.Combine(dir, "predict.csv");
        await File.WriteAllTextAsync(predictCsv, "Temp,Press\n21.0,1.05\n37.0,2.30\n");

        var outputCsv = Path.Combine(dir, "out.csv");
        await new PredictionEngine().PredictAsync(
            modelPath, predictCsv, outputCsv, SchemaWithVocabulary(vocabulary),
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: null,
            preserveColumns: null,
            interval: null,
            taskType: "binary-classification");

        var lines = await File.ReadAllLinesAsync(outputCsv);
        var header = lines[0].TrimStart('﻿').Split(',');
        var labelIndex = Array.IndexOf(header, "PredictedLabel");
        var csv = lines.Skip(1).Where(l => l.Length > 0).Select(l => l.Split(',')[labelIndex]).ToList();

        var rows = new[]
        {
            new Dictionary<string, object> { ["Temp"] = 21.0, ["Press"] = 1.05 },
            new Dictionary<string, object> { ["Temp"] = 37.0, ["Press"] = 2.30 },
        };
        var structured = new PredictionService(ml)
            .Predict(rows, SchemaWithVocabulary(vocabulary), modelPath, "binary-classification", "Result")
            .Rows.Select(r => r.PredictedLabel ?? "<null>").ToList();

        return (csv, structured);
    }

    [Fact]
    public async Task BothPathsAnswerInTheVocabularyTheModelWasTrainedOn()
    {
        var (csv, structured) = await BothPaths(["OK", "NG"]);

        Assert.Equal(csv, structured);
        Assert.All(csv, label => Assert.Contains(label, new[] { "OK", "NG" }));
        Assert.Equal("OK", csv[0]);
        Assert.Equal("NG", csv[1]);
    }

    /// <summary>
    /// And with no recorded vocabulary they still agree with each other — which is the half that was
    /// broken independently of the class names: one path printed the Boolean as digits, the other as
    /// words, for the same prediction.
    /// </summary>
    [Fact]
    public async Task WithoutAVocabularyTheTwoPathsStillAgree()
    {
        var (csv, structured) = await BothPaths(vocabulary: null);

        Assert.Equal(csv, structured);
        Assert.All(csv, label => Assert.Contains(label, new[] { "True", "False" }));
    }
}
