using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// ① drift guard: the CLI CSV path (<see cref="PredictionEngine"/> <c>mloop predict --output</c>) must
/// carry the SAME normalized per-row <c>Confidence</c> the <c>--json</c>/serve path
/// (<see cref="PredictionService"/>) exposes — both derived from the single
/// <see cref="ConfidencePolicy"/> authority, never re-derived per path. Previously confidence lived only
/// on the structured path, forcing a consumer wanting CSV + confidence to run predict twice. These tests
/// pin that the CSV now carries an additive <c>Confidence</c> column equal to the structured value.
/// </summary>
public class PredictConfidenceColumnTests : IDisposable
{
    private readonly List<string> _tempDirs = new();

    public void Dispose()
    {
        foreach (var d in _tempDirs)
        {
            try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { }
        }
    }

    private string NewDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-conf-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    public class SimpleReg
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    public class SimpleBinary
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public bool Label { get; set; }
    }

    [Fact]
    public async Task Regression_WithInterval_CsvConfidenceMatchesServicePath()
    {
        var ml = new MLContext(seed: 42);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(1, 20)
            .Select(i => new SimpleReg { X = i, Y = 2f * i }));
        var model = ml.Transforms.Concatenate("Features", "X")
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Y")).Fit(data);

        var dir = NewDir();
        var modelPath = Path.Combine(dir, ExperimentLayout.ModelFileName);
        ml.Model.Save(model, data.Schema, modelPath);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };
        // Constant-width band with a residual scale => a normalized confidence of 1 - half/(3*std).
        var interval = new RegressionInterval(HalfWidth: 1.5, Confidence: 0.90, ResidualStd: 2.0);

        var xs = new[] { 6.0f, 11.0f };

        // --- structured path (single authority) ---
        var serviceRows = xs.Select(x => new Dictionary<string, object> { ["X"] = x }).ToArray();
        var serviceResult = new PredictionService(ml)
            .Predict(serviceRows, schema, modelPath, "regression", "Y", interval);
        Assert.All(serviceResult.Rows, r => Assert.NotNull(r.Confidence));

        // --- CSV path ---
        var inputCsv = Path.Combine(dir, "input.csv");
        await File.WriteAllTextAsync(inputCsv,
            "X\n" + string.Join("\n", xs.Select(x => x.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        var outputCsv = Path.Combine(dir, "output.csv");

        await new PredictionEngine().PredictAsync(
            modelPath, inputCsv, outputCsv, schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: "Y",
            preserveColumns: null,
            interval: interval,
            taskType: "regression");

        var csvConfidences = ReadConfidenceColumn(outputCsv);
        Assert.Equal(xs.Length, csvConfidences.Count);

        // The CSV Confidence equals the structured path's confidence for every row (float CSV round-trip).
        for (int i = 0; i < xs.Length; i++)
        {
            Assert.NotNull(csvConfidences[i]);
            Assert.Equal(serviceResult.Rows[i].Confidence!.Value, csvConfidences[i]!.Value, 3);
        }
    }

    [Fact]
    public async Task Regression_WithoutInterval_OmitsConfidenceColumn()
    {
        // A bare regression point estimate has no confidence signal (Score is a target value), so the
        // additive column is omitted entirely rather than written as an all-empty column.
        var ml = new MLContext(seed: 42);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(1, 20)
            .Select(i => new SimpleReg { X = i, Y = 2f * i }));
        var model = ml.Transforms.Concatenate("Features", "X")
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Y")).Fit(data);

        var dir = NewDir();
        var modelPath = Path.Combine(dir, ExperimentLayout.ModelFileName);
        ml.Model.Save(model, data.Schema, modelPath);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var inputCsv = Path.Combine(dir, "input.csv");
        await File.WriteAllTextAsync(inputCsv, "X\n6\n11");
        var outputCsv = Path.Combine(dir, "output.csv");

        await new PredictionEngine().PredictAsync(
            modelPath, inputCsv, outputCsv, schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: "Y",
            preserveColumns: null,
            interval: null,             // no band → no confidence
            taskType: "regression");

        var header = File.ReadAllLines(outputCsv)[0];
        Assert.DoesNotContain("Confidence", header);
    }

    [Fact]
    public async Task Binary_CsvHasConfidenceInUnitInterval_MatchingServicePath()
    {
        var ml = new MLContext(seed: 42);
        var trainRows = new List<SimpleBinary>();
        for (int i = 0; i < 20; i++)
        {
            trainRows.Add(new SimpleBinary { X1 = 1f + i * 0.01f, X2 = 1f, Label = true });
            trainRows.Add(new SimpleBinary { X1 = 5f + i * 0.01f, X2 = 5f, Label = false });
        }
        var data = ml.Data.LoadFromEnumerable(trainRows);
        // Trainer-only over a pre-built "Features" vector — matches how the CLI predict path's InferColumns
        // merges the numeric feature columns into "Features" at load time (an embedded Concatenate over the
        // individual X1/X2 would instead expect them un-merged and clash with InferColumns' featurization).
        var featurized = ml.Transforms.Concatenate("Features", "X1", "X2").Fit(data).Transform(data);
        var model = ml.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: "Label", featureColumnName: "Features").Fit(featurized);

        var dir = NewDir();
        var modelPath = Path.Combine(dir, ExperimentLayout.ModelFileName);
        ml.Model.Save(model, featurized.Schema, modelPath);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X1", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "X2", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Categorical", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["X1"] = 1.0f, ["X2"] = 1.0f },
            new Dictionary<string, object> { ["X1"] = 5.0f, ["X2"] = 5.0f },
        };
        var serviceResult = new PredictionService(ml)
            .Predict(rows.Select(r => new Dictionary<string, object>(r)).ToArray(),
                schema, modelPath, "binary-classification", "Label");

        var inputCsv = Path.Combine(dir, "input.csv");
        await File.WriteAllTextAsync(inputCsv, "X1,X2\n1,1\n5,5");
        var outputCsv = Path.Combine(dir, "output.csv");

        await new PredictionEngine().PredictAsync(
            modelPath, inputCsv, outputCsv, schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: "Label",
            preserveColumns: null,
            interval: null,
            taskType: "binary-classification");

        var csvConfidences = ReadConfidenceColumn(outputCsv);
        Assert.Equal(2, csvConfidences.Count);
        for (int i = 0; i < 2; i++)
        {
            Assert.NotNull(csvConfidences[i]);
            Assert.InRange(csvConfidences[i]!.Value, 0.5, 1.0);
            Assert.Equal(serviceResult.Rows[i].Confidence!.Value, csvConfidences[i]!.Value, 3);
        }
    }

    private static List<double?> ReadConfidenceColumn(string csvPath)
    {
        var lines = File.ReadAllLines(csvPath);
        var header = lines[0].Split(',');
        int idx = Array.IndexOf(header, "Confidence");
        Assert.True(idx >= 0, "CSV missing Confidence column");

        var values = new List<double?>();
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            var cols = lines[i].Split(',');
            var cell = idx < cols.Length ? cols[idx] : "";
            values.Add(string.IsNullOrEmpty(cell)
                ? (double?)null
                : double.Parse(cell, System.Globalization.CultureInfo.InvariantCulture));
        }
        return values;
    }
}
