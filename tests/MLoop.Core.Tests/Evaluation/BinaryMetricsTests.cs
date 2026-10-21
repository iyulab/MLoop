using Microsoft.ML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class BinaryMetricsTests
{
    private sealed class Scored
    {
        public bool Label { get; set; }
        public float Score { get; set; }
        public bool PredictedLabel { get; set; }
    }

    private static Microsoft.ML.Data.BinaryClassificationMetrics Evaluate(params (bool Label, float Score)[] rows)
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(rows.Select(r => new Scored { Label = r.Label, Score = r.Score, PredictedLabel = r.Score > 0 }));
        return ml.BinaryClassification.EvaluateNonCalibrated(data);
    }

    /// <summary>
    /// A name <c>train --metric</c> accepts is a name every binary result carries — the report used to
    /// leave out auprc and negative_precision, so a run optimizing either recorded no value for it.
    /// </summary>
    [Fact]
    public void Describe_ReportsEveryBinaryMetricTheOptimizerAccepts()
    {
        var metrics = BinaryMetrics.Describe(Evaluate((true, 2f), (true, 0.5f), (false, -1f), (false, 0.3f), (true, -0.2f)));

        Assert.Equal(MetricNames.BinaryFamily.OrderBy(n => n), metrics.Keys.OrderBy(n => n));
        Assert.Equal(0.5, metrics["negative_precision"], 9);   // answered no: -1 (a no) and -0.2 (a yes)
    }

    [Fact]
    public void Describe_ReportsOnlyFiniteValues()
    {
        // Every row answered yes: nothing answered no. ML.NET reports 0 there; whatever it reports, a
        // metric file never carries NaN or an infinity.
        var metrics = BinaryMetrics.Describe(Evaluate((true, 1f), (false, 0.5f)));

        Assert.All(metrics.Values, v => Assert.True(double.IsFinite(v)));
    }
}
