using Microsoft.ML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// A name <c>train --metric</c> accepts is a name every result of that task carries (a report may
/// carry more — log_loss_reduction is reported, not optimized). The reports used
/// to leave out auprc, negative_precision and log_loss_reduction, so a run optimizing one of them
/// recorded no value for the metric it optimized.
/// </summary>
public class MetricReportTests
{
    private readonly MLContext _ml = new(seed: 1);

    private sealed class BinaryScored
    {
        public bool Label { get; set; }
        public float Score { get; set; }
        public bool PredictedLabel { get; set; }
    }

    private sealed class RegressionScored
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }

    private sealed class MulticlassRow
    {
        public float X { get; set; }
        public string Y { get; set; } = "";
    }

    [Fact]
    public void Binary_ReportsEveryMetricItsOptimizerAccepts()
    {
        var rows = new[] { (true, 2f), (true, 0.5f), (false, -1f), (false, 0.3f), (true, -0.2f) }
            .Select(r => new BinaryScored { Label = r.Item1, Score = r.Item2, PredictedLabel = r.Item2 > 0 });
        var metrics = MetricReport.Describe(_ml.BinaryClassification.EvaluateNonCalibrated(_ml.Data.LoadFromEnumerable(rows)));

        Assert.Subset(metrics.Keys.ToHashSet(), MetricNames.OptimizableFor("binary-classification")!.ToHashSet());
        Assert.Equal(0.5, metrics["negative_precision"], 9);   // answered no: -1 (a no) and -0.2 (a yes)
    }

    [Fact]
    public void Multiclass_ReportsEveryMetricItsOptimizerAccepts()
    {
        var data = _ml.Data.LoadFromEnumerable(Enumerable.Range(0, 30)
            .Select(i => new MulticlassRow { X = i % 3 + i % 7 * 0.05f, Y = (i % 3).ToString() }));
        var model = _ml.Transforms.Conversion.MapValueToKey("Label", "Y")
            .Append(_ml.Transforms.Concatenate("Features", "X"))
            .Append(_ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
            .Fit(data);
        var metrics = MetricReport.Describe(_ml.MulticlassClassification.Evaluate(model.Transform(data)));

        Assert.Subset(metrics.Keys.ToHashSet(), MetricNames.OptimizableFor("multiclass-classification")!.ToHashSet());
        Assert.Contains("log_loss_reduction", metrics.Keys);
    }

    [Fact]
    public void Regression_ReportsEveryMetricItsOptimizerAccepts()
    {
        var rows = new[] { (1f, 1.1f), (2f, 1.9f), (3f, 3.2f) }.Select(r => new RegressionScored { Label = r.Item1, Score = r.Item2 });
        var metrics = MetricReport.Describe(_ml.Regression.Evaluate(_ml.Data.LoadFromEnumerable(rows)));

        Assert.Subset(metrics.Keys.ToHashSet(), MetricNames.OptimizableFor("regression")!.ToHashSet());
    }

    private sealed class Point
    {
        [Microsoft.ML.Data.VectorType(2)]
        public float[] Features { get; set; } = [];
    }

    private IDataView Points(params (float X, float Y)[] points)
        => _ml.Data.LoadFromEnumerable(points.Select(p => new Point { Features = [p.X, p.Y] }));

    [Fact]
    public void Clustering_ReportsOnlyWhatMlNetCouldCompute()
    {
        var train = Points((0, 0), (0.1f, 0), (0, 0.2f), (0.2f, 0.1f), (5, 5), (5.1f, 5), (5, 5.2f), (5.2f, 5.1f));
        var model = _ml.Clustering.Trainers.KMeans(numberOfClusters: 2).Fit(train);

        // Rows in both clusters: the index exists. No label column: no mutual information.
        var both = MetricReport.Describe(_ml.Clustering.Evaluate(model.Transform(train), scoreColumnName: "Score", featureColumnName: "Features"));
        Assert.True(both["davies_bouldin_index"] > 0);
        Assert.True(both.ContainsKey("average_distance"));
        Assert.False(both.ContainsKey("normalized_mutual_information"));

        // Rows in a single cluster: ML.NET answers 0, which would read as the best possible index.
        var one = MetricReport.Describe(_ml.Clustering.Evaluate(
            model.Transform(Points((0, 0), (0.1f, 0.1f))), scoreColumnName: "Score", featureColumnName: "Features"));
        Assert.False(one.ContainsKey("davies_bouldin_index"));
    }
}
