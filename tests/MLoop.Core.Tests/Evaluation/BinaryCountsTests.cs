using Microsoft.ML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// The metrics counted from the answers a model gives, and how they replace ML.NET's (which re-derives each answer from
/// the raw score at 0 and so can disagree with a calibrated model's own answers).
/// </summary>
public class BinaryCountsTests
{
    private readonly MLContext _ml = new(seed: 0);

    private sealed class Row
    {
        public bool Label { get; set; }
        public bool PredictedLabel { get; set; }
        public float W { get; set; } = 1;
    }

    private IDataView View(params (bool Label, bool Predicted)[] rows)
        => _ml.Data.LoadFromEnumerable(rows.Select(r => new Row { Label = r.Label, PredictedLabel = r.Predicted }));

    // A model that answers "no" to every row has found none of the positives: recall 0 and F1 0. Its precision is the
    // one thing undefined (nothing was predicted positive).
    [Fact]
    public void AModelAnsweringNegativeToEveryRow_HasF1Zero_AndNoPrecision()
    {
        var metrics = BinaryCounts.Metrics(View((true, false), (true, false), (false, false), (false, false), (false, false)), "Label")!;

        Assert.Equal(0, metrics["recall"]);
        Assert.Equal(0, metrics["f1_score"]);
        Assert.False(metrics.ContainsKey("precision"));
        Assert.Equal(0.6, metrics["accuracy"], 9);
    }

    [Fact]
    public void F1_WhenBothPrecisionAndRecallAreDefined_IsTheirHarmonicMean()
    {
        var metrics = BinaryCounts.Metrics(View((true, true), (true, false), (false, true), (false, false)), "Label")!;

        Assert.Equal(0.5, metrics["precision"], 9);
        Assert.Equal(0.5, metrics["recall"], 9);
        Assert.Equal(0.5, metrics["f1_score"], 9);
    }

    [Fact]
    public void F1_WhenTheRowsHoldNoPositive_IsLeftOut()
    {
        var metrics = BinaryCounts.Metrics(View((false, true), (false, false)), "Label")!;

        Assert.False(metrics.ContainsKey("f1_score"));
        Assert.False(metrics.ContainsKey("recall"));
    }

    // The measured case: ML.NET's own numbers (from the raw score) said precision 1 and F1 1 for a model whose answers had
    // recall 0. Replacing only what the answers define left them standing.
    [Fact]
    public void ReplaceAnswerMetrics_RemovesAnAnswerMetricTheAnswersLeaveUndefined_AndKeepsTheRest()
    {
        var evaluator = new Dictionary<string, double>
        {
            ["accuracy"] = 0.67,
            ["precision"] = 1,
            ["recall"] = 0.5,
            ["negative_precision"] = 0.7,
            ["negative_recall"] = 1,
            ["f1_score"] = 1,
            ["auc"] = 1,
            ["auprc"] = 1,
            ["ece"] = 0.3,
        };
        var counted = BinaryCounts.Metrics(View((true, false), (true, false), (false, false), (false, false), (false, false)), "Label")!;

        BinaryCounts.ReplaceAnswerMetrics(evaluator, counted);

        Assert.False(evaluator.ContainsKey("precision"));
        Assert.Equal(0, evaluator["recall"]);
        Assert.Equal(0, evaluator["f1_score"]);
        Assert.Equal(0.6, evaluator["accuracy"], 9);
        Assert.Equal(1, evaluator["auc"]);          // scores, not answers: untouched
        Assert.Equal(0.3, evaluator["ece"]);
    }
}
