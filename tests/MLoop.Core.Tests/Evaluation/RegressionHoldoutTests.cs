using Microsoft.ML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// A regression holdout that carries no information is not scored: ML.NET's evaluator answers it with zeros (no row) or an
/// r² of millions (one row, or a label that never varies), and a quality gate reads those as a model.
/// </summary>
public class RegressionHoldoutTests
{
    private readonly MLContext _ml = new(seed: 0);

    private sealed class Row
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }

    private IDataView View(params (float Label, float Score)[] rows)
        => _ml.Data.LoadFromEnumerable(rows.Select(r => new Row { Label = r.Label, Score = r.Score }));

    [Fact]
    public void Count_SkipsRowsWithANonFiniteLabelOrScore_AndMeasuresTheLabelSpread()
    {
        var census = RegressionHoldout.Count(View((1, 1), (3, 3), (float.NaN, 2), (5, float.PositiveInfinity), (2, 2)), "Label");

        Assert.Equal(3, census.Rows);
        Assert.Equal(2.0 / 3, census.LabelVariance, 9);   // labels 1, 3, 2 → mean 2, population variance 2/3
    }

    [Fact]
    public void MarkUndefined_WithNoScoreableRow_MarksEveryMetric_AndSaysWhy()
    {
        var metrics = new Dictionary<string, double> { ["r_squared"] = 0, ["rmse"] = 0, ["mae"] = 0, ["mse"] = 0 };

        var reason = RegressionHoldout.MarkUndefined(metrics, RegressionHoldout.Count(View(), "Label"));

        Assert.NotNull(reason);
        Assert.All(metrics.Values, v => Assert.True(double.IsNaN(v)));
    }

    [Fact]
    public void MarkUndefined_WithASingleRow_MarksOnlyRSquared()
    {
        var metrics = new Dictionary<string, double> { ["r_squared"] = 17940886.7, ["rmse"] = 0.023, ["mae"] = 0.023, ["mse"] = 0.0005 };

        var reason = RegressionHoldout.MarkUndefined(metrics, RegressionHoldout.Count(View((4, 4.023f)), "Label"));

        Assert.NotNull(reason);
        Assert.True(double.IsNaN(metrics["r_squared"]));
        Assert.Equal(0.023, metrics["rmse"]);
        Assert.Equal(0.0005, metrics["mse"]);
    }

    [Fact]
    public void MarkUndefined_WithALabelThatNeverVaries_MarksOnlyRSquared()
    {
        var metrics = new Dictionary<string, double> { ["r_squared"] = -1e9, ["rmse"] = 1 };

        var reason = RegressionHoldout.MarkUndefined(metrics, RegressionHoldout.Count(View((2, 1), (2, 3), (2, 2)), "Label"));

        Assert.NotNull(reason);
        Assert.True(double.IsNaN(metrics["r_squared"]));
        Assert.Equal(1, metrics["rmse"]);
    }

    [Fact]
    public void MarkUndefined_WhenTheHoldoutCanBeScored_LeavesTheMetricsAlone()
    {
        var metrics = new Dictionary<string, double> { ["r_squared"] = 0.9, ["rmse"] = 0.4 };

        Assert.Null(RegressionHoldout.MarkUndefined(metrics, RegressionHoldout.Count(View((1, 1.1f), (3, 2.8f), (2, 2)), "Label")));
        Assert.Equal(0.9, metrics["r_squared"]);
    }
}
