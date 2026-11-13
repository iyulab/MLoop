using Microsoft.ML;
using MLoop.Core.Evaluation;
using Xunit;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// The bootstrap interval of a test file's metrics: reproducible, centred on the reported value, wider for fewer
/// rows, responsive to weights, silent where it cannot say anything — and covering what it claims to cover.
/// </summary>
public class MetricBootstrapTests(ITestOutputHelper output)
{
    private readonly MLContext _ml = new(seed: 0);

    private sealed class RegressionRow
    {
        public float Label { get; set; }
        public float Score { get; set; }
        public float W { get; set; } = 1;
    }

    private sealed class BinaryRow
    {
        public bool Label { get; set; }
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
        public float W { get; set; } = 1;
    }

    // y = x + noise, the model answers x: a population R² of 0.5 when the noise is as wide as x.
    private static List<RegressionRow> RegressionSample(int n, int seed, Func<int, float>? weight = null)
    {
        var rng = new Random(seed);
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        return [.. Enumerable.Range(0, n).Select(i =>
        {
            var x = Normal();
            return new RegressionRow { Label = (float)(x + Normal()), Score = (float)x, W = weight?.Invoke(i) ?? 1 };
        })];
    }

    private IMetricRows ReadRegression(List<RegressionRow> rows, bool weighted = false)
        => WeightedMetrics.ReadRegression(_ml.Data.LoadFromEnumerable(rows), "Label", weighted ? "W" : null)!;

    [Fact]
    public void TheSameRows_GiveTheSameInterval()
    {
        var rows = ReadRegression(RegressionSample(110, 1));

        var first = MetricBootstrap.Measure(rows);
        var second = MetricBootstrap.Measure(rows);

        Assert.Equal(first.Intervals["r_squared"], second.Intervals["r_squared"]);
        Assert.Equal(110, first.Rows);
        Assert.Null(first.Note);
    }

    [Fact]
    public void TheIntervalHoldsTheReportedValue_AndEveryMetricGetsOne()
    {
        var rows = ReadRegression(RegressionSample(200, 2));
        var point = rows.Compute()!;

        var result = MetricBootstrap.Measure(rows);

        Assert.Equal(point.Keys.OrderBy(k => k), result.Intervals.Keys.OrderBy(k => k));
        foreach (var (key, interval) in result.Intervals)
        {
            Assert.True(interval.Low <= point[key] && point[key] <= interval.High, $"{key}: {point[key]} outside [{interval.Low}, {interval.High}]");
            Assert.True(interval.Low < interval.High);
            Assert.Equal(MetricBootstrap.Resamples, interval.Resamples);
        }
    }

    [Fact]
    public void FewerRowsGiveAWiderInterval()
    {
        var small = MetricBootstrap.Measure(ReadRegression(RegressionSample(40, 3))).Intervals["r_squared"];
        var large = MetricBootstrap.Measure(ReadRegression(RegressionSample(1_000, 3))).Intervals["r_squared"];

        Assert.True(small.High - small.Low > 2 * (large.High - large.Low),
            $"n=40 width {small.High - small.Low:F3}, n=1000 width {large.High - large.Low:F3}");
    }

    [Fact]
    public void AWeightChangesWhatTheRowsStandFor()
    {
        var sample = RegressionSample(150, 4, i => i < 30 ? 6f : 1f);
        var unweighted = MetricBootstrap.Measure(ReadRegression(sample)).Intervals["r_squared"];
        var weighted = MetricBootstrap.Measure(ReadRegression(sample, weighted: true)).Intervals["r_squared"];

        Assert.NotEqual(unweighted, weighted);
    }

    [Fact]
    public void ARowThatCountsForNothingIsNotInTheSample()
    {
        var sample = RegressionSample(60, 5, i => i < 20 ? 0f : 1f);

        var result = MetricBootstrap.Measure(ReadRegression(sample, weighted: true));

        Assert.Equal(40, result.Rows);
    }

    [Fact]
    public void AMetricARowSetCannotDefine_GetsNoInterval()
    {
        // Eight positives in sixty: many resamples hold none, so AUC is defined in fewer of them than accuracy.
        var rng = new Random(6);
        var rows = Enumerable.Range(0, 60).Select(i =>
        {
            var y = i < 1; // one positive: a resample without it has no AUC (about 36% of them)
            var score = (float)(y ? 0.9 : rng.NextDouble() * 0.8);
            return new BinaryRow { Label = y, Score = score, Probability = score, PredictedLabel = score > 0.5f };
        }).ToList();
        var read = WeightedMetrics.ReadBinary(_ml.Data.LoadFromEnumerable(rows), "Label", null)!;

        var result = MetricBootstrap.Measure(read);

        Assert.True(result.Intervals.ContainsKey("accuracy"));
        if (result.Intervals.TryGetValue("auc", out var auc))
            Assert.True(auc.Resamples < result.Intervals["accuracy"].Resamples, "AUC must have been undefined in some resamples");
    }

    [Fact]
    public void APerfectScore_IsNotOfferedAsAnIntervalOfZeroWidth()
    {
        var rng = new Random(8);
        var rows = Enumerable.Range(0, 70).Select(i =>
        {
            var y = i % 2 == 0;
            var score = (float)((y ? 1 : -1) * (0.5 + rng.NextDouble()));
            return new BinaryRow { Label = y, Score = score, Probability = 1 / (1 + MathF.Exp(-score)), PredictedLabel = y };
        }).ToList();
        var read = WeightedMetrics.ReadBinary(_ml.Data.LoadFromEnumerable(rows), "Label", null)!;

        var result = MetricBootstrap.Measure(read);

        Assert.False(result.Intervals.ContainsKey("accuracy"));
        Assert.Contains("perfect score", result.Note, StringComparison.Ordinal);
        Assert.Contains("accuracy", result.Note, StringComparison.Ordinal);
    }

    [Fact]
    public void ARowSetTooLargeToResample_SaysSoInsteadOfGuessing()
    {
        var rows = new Rows(2_000_000);

        var result = MetricBootstrap.Measure(rows);

        Assert.Empty(result.Intervals);
        Assert.Contains("not computed", result.Note, StringComparison.Ordinal);
    }

    [Fact]
    public void OneRow_IsNotASample()
    {
        var result = MetricBootstrap.Measure(ReadRegression(RegressionSample(1, 7)));

        Assert.Empty(result.Intervals);
        Assert.NotNull(result.Note);
    }

    /// <summary>
    /// Coverage: how often the interval holds the population value. Recorded, not hidden: a percentile bootstrap
    /// covers somewhat less than 95% on small samples. The asserted floor is loose on purpose — it catches a
    /// broken resampler, and the printed numbers are what the design note quotes.
    /// </summary>
    [Fact]
    public void Coverage_OfTheRSquaredInterval_IsRecordedAndNotFarBelowItsLevel()
    {
        // The population R² of y = x + e with e as wide as x: 1 − Var(e) / Var(y) = 0.5.
        const double population = 0.5;
        foreach (var (n, reps) in new[] { (40, 300), (110, 300), (500, 150) })
        {
            var covered = 0;
            for (var r = 0; r < reps; r++)
            {
                var interval = MetricBootstrap.Measure(ReadRegression(RegressionSample(n, 10_000 * n + r))).Intervals["r_squared"];
                if (interval.Low <= population && population <= interval.High)
                    covered++;
            }
            var coverage = covered / (double)reps;
            output.WriteLine($"n={n}: {covered}/{reps} intervals hold the population R² = {coverage:P1}");
            Assert.True(coverage >= 0.85, $"n={n}: coverage {coverage:P1}");
        }
    }

    // A row set with nothing to compute from: only its size matters to the work cap.
    private sealed class Rows(int count) : IMetricRows
    {
        public int Count => count;
        public double[] Weights { get; } = Enumerable.Repeat(1.0, count).ToArray();
        public Dictionary<string, double>? Compute(double[] weights) => new() { ["x"] = 1 };
    }
}
