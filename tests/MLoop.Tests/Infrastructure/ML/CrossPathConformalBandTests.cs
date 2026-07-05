using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// PRED-1 drift guard: the two predict paths must produce the SAME regression conformal band.
///
/// <para><see cref="PredictionEngine"/> (CLI <c>mloop predict</c>, CSV→CSV) and
/// <see cref="PredictionService"/> (serve <c>/predict</c> + CLI <c>--json</c>, rows→structured) build the
/// heteroscedastic band through separate code — the engine via a CustomMapping, the service via
/// <see cref="RegressionInterval.WidthFor"/> in ExtractRegressionRows. Both must resolve the per-row
/// half-width to <c>q·(max(σ,0)+β)</c>; if one re-inlines a divergent formula the CSV and the JSON would
/// silently disagree on the band while every other test still passes — exactly the F-33/F-27 cross-path
/// drift class. Both paths route through <see cref="RegressionInterval.WidthFor"/>; this test pins that
/// they keep producing an identical band for the same model + input.</para>
/// </summary>
public class CrossPathConformalBandTests : IDisposable
{
    private readonly List<string> _tempDirs = new();
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
        {
            try { if (File.Exists(f)) File.Delete(f); } catch { }
        }
        foreach (var d in _tempDirs)
        {
            try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { }
        }
    }

    public class SimpleReg
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    [Fact]
    public async Task Heteroscedastic_CsvPath_And_ServicePath_ProduceIdenticalBands()
    {
        // Build a hetero regression model whose residual magnitude grows with X, exactly as the serve-path
        // test (PredictionServiceTests.Predict_Regression_Heteroscedastic_...) does, so the σ-model widens
        // the band per row. Dedicated seeded context keeps the SDCA aux fit deterministic.
        var ml = new MLContext(seed: 42);
        var trainRows = Enumerable.Range(0, 60).Select(i =>
        {
            float x = (i % 20) + 1;
            return new SimpleReg { X = x, Y = 2f * x + 0.5f * x * ((i % 2 == 0) ? 1 : -1) };
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(trainRows);
        var mainModel = ml.Transforms.Concatenate("Features", "X")
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Y")).Fit(data);
        var scored = mainModel.Transform(data);

        var norm = AutoMLRunner.ComputeNormalizedConformal(ml, scored, "Y", new[] { "Features" });
        Assert.NotNull(norm);
        var metrics = new Dictionary<string, double>(AutoMLRunner.ComputeConformalIntervals(scored, "Y"));
        foreach (var kv in norm!.Metrics) metrics[kv.Key] = kv.Value;
        var interval = RegressionInterval.FromMetrics("regression", metrics);
        Assert.NotNull(interval);
        Assert.True(interval!.IsHeteroscedastic);

        var dir = Path.Combine(Path.GetTempPath(), "mloop-crosspath-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        var modelPath = Path.Combine(dir, ExperimentLayout.ModelFileName);
        ml.Model.Save(mainModel, data.Schema, modelPath);
        ml.Model.Save(norm.AuxModel, null, Path.Combine(dir, ExperimentLayout.ResidualModelFileName));

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        // Two inputs with very different X => the σ-model gives them very different band widths, so the
        // test would catch a formula that only agreed on constant-width.
        var xs = new[] { 1.0f, 20.0f };

        // --- Path A: PredictionService (rows -> structured) ---
        var serviceRows = xs.Select(x => new Dictionary<string, object> { ["X"] = x }).ToArray();
        var serviceResult = new PredictionService(ml)
            .Predict(serviceRows, schema, modelPath, "regression", "Y", interval);
        Assert.Equal(xs.Length, serviceResult.Rows.Count);

        // --- Path B: PredictionEngine (CSV -> CSV) ---
        var inputCsv = Path.Combine(dir, "input.csv");
        await File.WriteAllTextAsync(inputCsv,
            "X\n" + string.Join("\n", xs.Select(x => x.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        var outputCsv = Path.Combine(dir, "output.csv");

        var count = await new PredictionEngine().PredictAsync(
            modelPath, inputCsv, outputCsv, schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: "Y",
            preserveColumns: null,
            interval: interval);
        Assert.Equal(xs.Length, count);

        var csvBands = ReadCsvBands(outputCsv);
        Assert.Equal(xs.Length, csvBands.Count);

        // The two paths must agree on the band for every row — same lower, same upper, hence same width.
        // Both are float CSV round-trips of the same q·(max(σ,0)+β) formula, so tolerance is tight.
        for (int i = 0; i < xs.Length; i++)
        {
            var sRow = serviceResult.Rows[i];
            Assert.NotNull(sRow.ScoreLowerBound);
            Assert.NotNull(sRow.ScoreUpperBound);

            Assert.Equal(sRow.ScoreLowerBound!.Value, csvBands[i].Lower, 2);
            Assert.Equal(sRow.ScoreUpperBound!.Value, csvBands[i].Upper, 2);

            double serviceWidth = sRow.ScoreUpperBound!.Value - sRow.ScoreLowerBound!.Value;
            double csvWidth = csvBands[i].Upper - csvBands[i].Lower;
            Assert.Equal(serviceWidth, csvWidth, 2);
        }

        // Sanity: the fixture actually exercised the per-row (heteroscedastic) branch, not a shared
        // constant width — otherwise the guard would pass trivially.
        double w0 = csvBands[0].Upper - csvBands[0].Lower;
        double w1 = csvBands[1].Upper - csvBands[1].Lower;
        Assert.True(Math.Abs(w0 - w1) > 1e-3, $"expected per-row widths to differ (w0={w0:F3}, w1={w1:F3})");
    }

    private static List<(double Lower, double Upper)> ReadCsvBands(string csvPath)
    {
        var lines = File.ReadAllLines(csvPath);
        var header = lines[0].Split(',');
        int iLower = Array.IndexOf(header, "ScoreLowerBound");
        int iUpper = Array.IndexOf(header, "ScoreUpperBound");
        Assert.True(iLower >= 0, "CSV missing ScoreLowerBound column");
        Assert.True(iUpper >= 0, "CSV missing ScoreUpperBound column");

        var bands = new List<(double, double)>();
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            var cols = lines[i].Split(',');
            bands.Add((
                double.Parse(cols[iLower], System.Globalization.CultureInfo.InvariantCulture),
                double.Parse(cols[iUpper], System.Globalization.CultureInfo.InvariantCulture)));
        }
        return bands;
    }
}
