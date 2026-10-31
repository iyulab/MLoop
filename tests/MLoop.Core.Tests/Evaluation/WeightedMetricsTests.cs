using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// Weighted test metrics agree with ML.NET's evaluators when every weight is 1, and a weight of k
/// counts a row as k copies of it.
/// </summary>
public class WeightedMetricsTests
{
    private readonly MLContext _ml = new(seed: 0);

    private sealed class BinaryRow
    {
        public bool Label { get; set; }
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
        public float W { get; set; }
    }

    private static List<BinaryRow> BinaryRows(int seed = 4)
    {
        var rng = new Random(seed);
        var rows = new List<BinaryRow>();
        for (var i = 0; i < 300; i++)
        {
            var y = rng.NextDouble() < 0.4;
            var score = (float)((y ? 0.6 : -0.6) + rng.NextDouble() * 2 - 1);
            score = MathF.Round(score, 1); // ties, so the tie rule is exercised
            rows.Add(new BinaryRow
            {
                Label = y,
                Score = score,
                PredictedLabel = score > 0,
                Probability = 1 / (1 + MathF.Exp(-score)),
                W = 1,
            });
        }
        return rows;
    }

    [Fact]
    public void Binary_WithoutAWeightColumn_CountsEachRowOnce()
    {
        var view = _ml.Data.LoadFromEnumerable(BinaryRows());

        Assert.Equal(WeightedMetrics.Binary(view, "Label", "W"), WeightedMetrics.Binary(view, "Label", weightColumn: null));
    }

    [Fact]
    public void Binary_AModelThatPredictsNoPositive_HasF1Zero_AndNoPrecision()
    {
        var rows = BinaryRows().Select(r => { r.PredictedLabel = false; return r; }).ToList();
        var weighted = WeightedMetrics.Binary(_ml.Data.LoadFromEnumerable(rows), "Label", "W")!;

        Assert.Equal(0, weighted["recall"]);
        Assert.Equal(0, weighted["f1_score"]);
        Assert.False(weighted.ContainsKey("precision"));
    }

    // ML.NET's evaluator throws on rows of one class ("AUC is not defined"); a fold of rare positives is such
    // rows, and a search scores it with what those rows define.
    [Fact]
    public void Binary_RowsOfOneClass_AreScored_WithoutWhatTheyLeaveUndefined()
    {
        var rows = BinaryRows().Where(r => !r.Label).ToList();
        var view = _ml.Data.LoadFromEnumerable(rows);
        Assert.ThrowsAny<Exception>(() => _ml.BinaryClassification.Evaluate(view, "Label"));

        var metrics = WeightedMetrics.Binary(view, "Label", weightColumn: null)!;

        Assert.Equal(rows.Count(r => !r.PredictedLabel) / (double)rows.Count, metrics["accuracy"], 9);
        Assert.Equal(metrics["accuracy"], metrics["negative_recall"], 9);
        Assert.False(metrics.ContainsKey("auc"));
        Assert.False(metrics.ContainsKey("recall"));
    }

    [Fact]
    public void Binary_UnitWeights_MatchTheEvaluator()
    {
        var view = _ml.Data.LoadFromEnumerable(BinaryRows());
        var expected = _ml.BinaryClassification.Evaluate(view, "Label");

        var weighted = WeightedMetrics.Binary(view, "Label", "W")!;

        Assert.Equal(expected.Accuracy, weighted["accuracy"], 9);
        Assert.Equal(expected.PositivePrecision, weighted["precision"], 9);
        Assert.Equal(expected.PositiveRecall, weighted["recall"], 9);
        Assert.Equal(expected.NegativeRecall, weighted["negative_recall"], 9);
        Assert.Equal(expected.F1Score, weighted["f1_score"], 9);
        Assert.Equal(expected.AreaUnderRocCurve, weighted["auc"], 6);
        Assert.Equal(CalibrationError.Binary(view, "Label")!.Value, weighted["ece"], 9);
    }

    [Fact]
    public void Binary_AWeightOfK_CountsTheRowKTimes()
    {
        var rows = BinaryRows();
        var duplicated = new List<BinaryRow>();
        for (var i = 0; i < rows.Count; i++)
        {
            var k = i % 4; // weights 0..3, zero included
            rows[i].W = k;
            for (var c = 0; c < k; c++)
                duplicated.Add(new BinaryRow { Label = rows[i].Label, PredictedLabel = rows[i].PredictedLabel, Score = rows[i].Score, Probability = rows[i].Probability, W = 1 });
        }

        var weighted = WeightedMetrics.Binary(_ml.Data.LoadFromEnumerable(rows), "Label", "W")!;
        var copies = WeightedMetrics.Binary(_ml.Data.LoadFromEnumerable(duplicated), "Label", "W")!;

        Assert.Equal(copies.Keys.Order(), weighted.Keys.Order());
        foreach (var key in copies.Keys)
            Assert.Equal(copies[key], weighted[key], 9);
    }

    [Fact]
    public void Auc_IsTheWeightedShareOfCorrectlyOrderedPairs()
    {
        // One positive (w 2) above one negative (w 1), one positive (w 1) tied with it: (2·1 + 1·0.5) / (3·1).
        var auc = WeightedMetrics.Auc([(true, 0.9, 2), (false, 0.4, 1), (true, 0.4, 1)]);
        Assert.Equal(2.5 / 3, auc!.Value, 12);
        Assert.Null(WeightedMetrics.Auc([(true, 0.9, 1), (false, 0.4, 0)]));
    }

    private sealed class MultiRow
    {
        public string Label { get; set; } = "";
        [VectorType(3)] public float[] Score { get; set; } = [];
        public float W { get; set; }
    }

    [Fact]
    public void Multiclass_UnitWeights_MatchTheEvaluator()
    {
        var rng = new Random(8);
        var names = new[] { "a", "b", "c" };
        var rows = Enumerable.Range(0, 240).Select(_ =>
        {
            var y = rng.Next(3);
            var raw = Enumerable.Range(0, 3).Select(c => (float)(rng.NextDouble() + (c == y ? 0.8 : 0))).ToArray();
            var sum = raw.Sum();
            return new MultiRow { Label = names[y], Score = raw.Select(v => v / sum).ToArray(), W = 1 };
        }).ToList();
        // Keys in a, b, c order, so slot i is the class with key i + 1 — as a trainer's output is.
        var data = _ml.Data.LoadFromEnumerable(rows);
        var scored = _ml.Transforms.Conversion.MapValueToKey("Label",
                keyData: _ml.Data.LoadFromEnumerable(names.Select(n => new KeyRow { Label = n })))
            .Fit(data).Transform(data);
        var expectedMicro = rows.Count(r => Array.IndexOf(r.Score, r.Score.Max()) == Array.IndexOf(names, r.Label)) / (double)rows.Count;

        var weighted = WeightedMetrics.Multiclass(scored, "Label", "W")!;

        Assert.Equal(expectedMicro, weighted["micro_accuracy"], 9);
        var perClass = names.Select(n => rows.Where(r => r.Label == n)
            .Average(r => Array.IndexOf(r.Score, r.Score.Max()) == Array.IndexOf(names, n) ? 1.0 : 0.0));
        Assert.Equal(perClass.Average(), weighted["macro_accuracy"], 9);
        var expectedLogLoss = rows.Average(r => -Math.Log(Math.Max(r.Score[Array.IndexOf(names, r.Label)], 1e-15)));
        Assert.Equal(expectedLogLoss, weighted["log_loss"], 9);
        Assert.Equal(CalibrationError.Multiclass(scored, "Label")!.Value, weighted["ece"], 9);
    }

    private sealed class KeyRow
    {
        public string Label { get; set; } = "";
    }

    private sealed class RegressionRow
    {
        public float Label { get; set; }
        public float Score { get; set; }
        public float W { get; set; }
    }

    [Fact]
    public void Regression_UnitWeights_MatchTheEvaluator_AndKCopies()
    {
        var rng = new Random(2);
        var rows = Enumerable.Range(0, 200).Select(i =>
        {
            var y = (float)(rng.NextDouble() * 10);
            return new RegressionRow { Label = y, Score = y + (float)(rng.NextDouble() - 0.5) * 2, W = 1 };
        }).ToList();
        var view = _ml.Data.LoadFromEnumerable(rows);
        var expected = _ml.Regression.Evaluate(view, "Label");

        var weighted = WeightedMetrics.Regression(view, "Label", "W")!;

        Assert.Equal(expected.RSquared, weighted["r_squared"], 6);
        Assert.Equal(expected.RootMeanSquaredError, weighted["rmse"], 6);
        Assert.Equal(expected.MeanAbsoluteError, weighted["mae"], 6);

        rows[0].W = 3;
        var copies = rows.Concat([new RegressionRow { Label = rows[0].Label, Score = rows[0].Score, W = 1 }, new RegressionRow { Label = rows[0].Label, Score = rows[0].Score, W = 1 }])
            .Select(r => new RegressionRow { Label = r.Label, Score = r.Score, W = 1 }).ToList();
        var w3 = WeightedMetrics.Regression(_ml.Data.LoadFromEnumerable(rows), "Label", "W")!;
        var c3 = WeightedMetrics.Regression(_ml.Data.LoadFromEnumerable(copies), "Label", "W")!;
        Assert.Equal(c3["r_squared"], w3["r_squared"], 9);
        Assert.Equal(c3["mae"], w3["mae"], 9);
    }
}
