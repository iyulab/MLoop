using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Drift guard: a time-series anomaly prediction written to CSV answers in the columns anomaly
/// detection answers in — <c>PredictedLabel</c> (is it an anomaly) and <c>Score</c> — with the values
/// the structured path reports as <c>isAnomaly</c> and <c>anomalyScore</c>.
/// </summary>
/// <remarks>
/// The detectors write one <c>Prediction</c> vector (alert, raw score, …). None of the columns the CSV
/// path keeps was present, so it fell back to writing the whole scored view: every input column twice,
/// the loader's <c>Features</c> vector slot by slot, and the answer as three unnamed numbers at the end.
/// The stand-in model emits that vector from text, so the test reads no numeric input the loader would
/// merge.
/// </remarks>
public class CrossPathTimeSeriesAnomalyRenderingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-tsa-csv-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
        GC.SuppressFinalize(this);
    }

    private sealed class Row
    {
        public string s { get; set; } = "";
        public string t { get; set; } = "";
    }

    private static readonly InputSchemaInfo Schema = new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "s", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
            new ColumnSchema { Name = "t", DataType = SchemaDataTypes.Text, Purpose = "Feature" },
        ],
    };

    [Fact]
    public async Task The_csv_says_whether_each_point_is_an_anomaly_and_its_score()
    {
        var ml = new MLContext(seed: 0);
        var data = ml.Data.LoadFromEnumerable([new Row { s = "a b c", t = "x" }]);
        // The detectors' output shape: a double vector whose slots are alert, raw score, …
        // Here the slots are word counts over the fitted vocabulary (a, b, c).
        var model = ml.Transforms.Text.ProduceWordBags("bag", "s", ngramLength: 1)
            .Append(ml.Transforms.Conversion.ConvertType(TimeSeriesAnomalyOutput.PredictionColumnName, "bag", DataKind.Double))
            .Fit(data);
        var modelPath = Path.Combine(_dir, "model.zip");
        ml.Model.Save(model, data.Schema, modelPath);

        var input = Path.Combine(_dir, "series.csv");
        await File.WriteAllLinesAsync(input, ["s,t", "a a c,x", "b,y"]);
        var output = Path.Combine(_dir, "out.csv");

        await new PredictionEngine().PredictAsync(
            modelPath, input, output, Schema, CategoricalMapper.UnknownValueStrategy.Auto,
            taskType: "time-series-anomaly");

        var lines = (await File.ReadAllLinesAsync(output)).Where(l => l.Length > 0).ToList();
        var header = lines[0].TrimStart('﻿').Split(',');
        var csv = lines.Skip(1).Select(l => l.Split(','))
            .Select(f => (
                Alert: f[Array.IndexOf(header, "PredictedLabel")],
                Score: double.Parse(f[Array.IndexOf(header, "Score")], System.Globalization.CultureInfo.InvariantCulture)))
            .ToList();

        // "a a c" → slots (2, 0, 1): an alert, score 0. "b" → (0, 1, 0): no alert, score 1.
        Assert.Equal(["True", "False"], csv.Select(r => r.Alert));
        Assert.Equal([0.0, 1.0], csv.Select(r => r.Score));
        Assert.DoesNotContain(header, h => h.StartsWith("Features", StringComparison.Ordinal));
        Assert.DoesNotContain(header, h => h.StartsWith(TimeSeriesAnomalyOutput.PredictionColumnName, StringComparison.Ordinal));

        var structured = new PredictionService(ml).Predict(
            [new Dictionary<string, object> { ["s"] = "a a c", ["t"] = "x" }, new Dictionary<string, object> { ["s"] = "b", ["t"] = "y" }],
            Schema, modelPath, "time-series-anomaly", "s").Rows;
        Assert.Equal(structured.Select(r => r.IsAnomaly?.ToString()), csv.Select(r => (string?)r.Alert));
        Assert.Equal(structured.Select(r => r.AnomalyScore), csv.Select(r => (double?)r.Score));
    }
}
