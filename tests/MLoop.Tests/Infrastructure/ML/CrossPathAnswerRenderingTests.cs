using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Runtime;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Drift guard: a question-answering prediction written to CSV carries the same best answer and score
/// the structured path reports.
/// </summary>
/// <remarks>
/// The question-answering trainer writes its answers to an <c>Answer</c> column, best first. The CSV
/// path kept only the columns other tasks answer in (<c>PredictedLabel</c>, <c>Score</c>, …), so the
/// answers were dropped before the degeneracy guard read them, and every prediction of a working model
/// was refused as "every defining output field came back null" — while <c>--json</c> on the same file
/// answered. The stand-in model here emits the trainer's output shape from the passage alone, as the
/// trainer reads only the passage and the question, so the test does not need the pretrained encoder.
/// </remarks>
public class CrossPathAnswerRenderingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-qa-csv-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
        GC.SuppressFinalize(this);
    }

    private sealed class Row
    {
        public string context { get; set; } = "";
        public string question { get; set; } = "";
    }

    private static readonly InputSchemaInfo Schema = new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "context", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
            new ColumnSchema { Name = "question", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
        ],
    };

    private static bool TorchRuntimeInstalled()
    {
        var runtime = RuntimeRegistry.GetRequiredByTask("question-answering");
        return runtime != null && new RuntimeManager().IsInstalled(runtime);
    }

    [Fact]
    public async Task The_csv_carries_the_best_answer_and_its_score_as_the_structured_path_does()
    {
        var ml = new MLContext(seed: 0);
        var data = ml.Data.LoadFromEnumerable([new Row { context = "seoul is big", question = "q" }]);
        // The trainer's output shape: answers best first, and a score vector.
        var model = ml.Transforms.Text.TokenizeIntoWords("Answer", "context")
            .Append(ml.Transforms.Text.ProduceWordBags("Score", "context"))
            .Fit(data);
        var modelPath = Path.Combine(_dir, "model.zip");
        ml.Model.Save(model, data.Schema, modelPath);

        var input = Path.Combine(_dir, "predict.csv");
        await File.WriteAllLinesAsync(input, ["context,question", "busan is far,where", "seoul is big,what"]);
        var output = Path.Combine(_dir, "out.csv");

        var count = await new PredictionEngine().PredictAsync(
            modelPath, input, output, Schema, CategoricalMapper.UnknownValueStrategy.Auto,
            taskType: "question-answering");

        var lines = (await File.ReadAllLinesAsync(output)).Where(l => l.Length > 0).ToList();
        var header = lines[0].TrimStart('﻿').Split(',');
        var csv = lines.Skip(1).Select(l => l.Split(','))
            .Select(f => (
                Answer: f[Array.IndexOf(header, "PredictedLabel")],
                Score: double.Parse(f[Array.IndexOf(header, "Score")], System.Globalization.CultureInfo.InvariantCulture)))
            .ToList();

        // The stand-in's score vector is the passage's word counts over the fitted vocabulary (seoul, is,
        // big): its first slot is 0 for "busan is far" and 1 for "seoul is big".
        Assert.Equal(2, count);
        Assert.Equal(["busan", "seoul"], csv.Select(r => r.Answer));
        Assert.Equal([0.0, 1.0], csv.Select(r => r.Score));

        // The structured path loads the task's native runtime before it reads a model, so the
        // comparison runs where that runtime is installed (not on CI); the values above hold everywhere.
        if (TorchRuntimeInstalled())
        {
            var structured = new PredictionService(ml).Predict(
                [
                    new Dictionary<string, object> { ["context"] = "busan is far", ["question"] = "where" },
                    new Dictionary<string, object> { ["context"] = "seoul is big", ["question"] = "what" },
                ],
                Schema, modelPath, "question-answering", "text").Rows;

            Assert.Equal(structured.Select(r => r.PredictedLabel), csv.Select(r => (string?)r.Answer));
            Assert.Equal(structured.Select(r => r.Score), csv.Select(r => (double?)r.Score));
        }
        Assert.DoesNotContain("Answer", header);
    }
}
