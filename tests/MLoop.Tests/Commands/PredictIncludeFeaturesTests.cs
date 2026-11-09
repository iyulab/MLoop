using MLoop.CLI.Commands;
using MLoop.Core.Models;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>predict</c> writes the input's columns beside the answers. The label column is left out (it is what the model answers), except
/// for a time-series detector, whose "label" is the series it judges: with a one-column series every column was left out and the
/// file began with a column that had no name and no cells.
/// </summary>
public sealed class PredictIncludeFeaturesTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-includefeatures-" + Guid.NewGuid().ToString("N"));

    public PredictIncludeFeaturesTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static InputSchemaInfo Schema(string label, params string[] features) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = label, DataType = "Single", Purpose = "Label" },
            .. features.Select(f => new ColumnSchema { Name = f, DataType = "Single", Purpose = "Feature" })
        ]
    };

    private string[] Merge(string input, string predictions, InputSchemaInfo schema, string? task)
    {
        var inputPath = Path.Combine(_directory, "in.csv");
        var outputPath = Path.Combine(_directory, "out.csv");
        File.WriteAllText(inputPath, input);
        File.WriteAllText(outputPath, predictions);
        PredictCommand.MergeInputWithPredictions(inputPath, outputPath, schema, task);
        return File.ReadAllLines(outputPath).Select(l => l.TrimStart('﻿')).ToArray();
    }

    [Fact]
    public void A_one_column_series_keeps_its_value_beside_the_verdict()
    {
        var lines = Merge("value\n1.5\n2.5\n", "PredictedLabel,Score\n,\nTrue,0.9\n", Schema("value"), "time-series-anomaly");

        Assert.Equal(["value,PredictedLabel,Score", "1.5,,", "2.5,True,0.9"], lines);
    }

    [Fact]
    public void For_other_tasks_the_label_is_still_left_out()
    {
        var lines = Merge("a,y\n1,10\n2,20\n", "Score\n9.5\n19.5\n", Schema("y", "a"), "regression");

        Assert.Equal(["a,Score", "1,9.5", "2,19.5"], lines);
    }

    [Fact]
    public void A_file_with_no_column_left_to_include_does_not_start_with_a_nameless_column()
    {
        var lines = Merge("y\n10\n20\n", "Score\n9.5\n19.5\n", Schema("y"), "regression");

        Assert.Equal(["Score", "9.5", "19.5"], lines);
    }
}
