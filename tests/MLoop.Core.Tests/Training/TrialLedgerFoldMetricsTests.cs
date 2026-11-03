using MLoop.Core.Training;
using MLoop.Search;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A trial's record keeps the metrics of each fold beside their mean and spread: the first fold told a bad trial from a good one
/// or it did not, and only the folds themselves can say which. A holdout has no folds to keep.
/// </summary>
public sealed class TrialLedgerFoldMetricsTests
{
    private static IReadOnlyDictionary<string, double> Fold(double auc) => new Dictionary<string, double> { ["auc"] = auc };

    [Fact]
    public void ATrialValidatedOverFolds_KeepsEachFoldsMetrics()
    {
        var trial = new TrialResult
        {
            Trainer = "FastTree",
            Metrics = Fold(0.8),
            Folds = 3,
            StandardDeviation = Fold(0.05),
            FoldMetrics = [Fold(0.75), Fold(0.8), Fold(0.85)],
        };

        var record = TrialLedger.ToRecord(trial, 1);

        Assert.Equal([0.75, 0.8, 0.85], record.Validation!.FoldMetrics!.Select(f => f["auc"]));
    }

    [Fact]
    public void AHoldout_AndANonFiniteFold_KeepNothingOfTheKind()
    {
        var holdout = new TrialResult { Trainer = "FastTree", Metrics = Fold(0.8), Folds = 1, StandardDeviation = Fold(0) };
        Assert.Null(TrialLedger.ToRecord(holdout, 1).Validation!.FoldMetrics);

        var withNaN = holdout with { Folds = 2, FoldMetrics = [Fold(double.NaN), Fold(0.8)] };
        var kept = TrialLedger.ToRecord(withNaN, 2).Validation!.FoldMetrics!;
        Assert.Empty(kept[0]);
        Assert.Equal(0.8, kept[1]["auc"]);
    }

    [Fact]
    public void TheJson_OmitsFoldMetricsWhenThereAreNone()
    {
        var holdout = new TrialResult { Trainer = "FastTree", Metrics = Fold(0.8), Folds = 1, StandardDeviation = Fold(0) };
        var json = System.Text.Json.JsonSerializer.Serialize(TrialLedger.ToRecord(holdout, 1));

        Assert.DoesNotContain("oldMetrics", json, StringComparison.OrdinalIgnoreCase);
    }
}
