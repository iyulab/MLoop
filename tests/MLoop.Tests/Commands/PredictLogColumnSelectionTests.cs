using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// Drift guard for <c>mloop predict --log</c>: the value written to the prediction logs (which feed
/// feedback/trigger/drift) must be the canonical point prediction — <c>PredictedLabel</c> or a scalar
/// <c>Score</c> — never a trailing enrichment column. Adding the conformal band
/// (<c>ScoreLowerBound</c>/<c>ScoreUpperBound</c>) and the additive <c>Confidence</c> column made the old
/// positional "last column" heuristic log the enrichment instead of the prediction; these tests pin the
/// name-based selection that fixes it (and the pre-existing banded-regression case it also repaired).
/// </summary>
public class PredictLogColumnSelectionTests
{
    private static Dictionary<string, object> Row(params (string Key, object Value)[] cells)
    {
        var d = new Dictionary<string, object>();
        foreach (var (k, v) in cells) d[k] = v;
        return d;
    }

    [Fact]
    public void BandedRegression_SelectsScore_NotTrailingConfidenceOrBound()
    {
        // Column order as written by the CSV path: features, Score, band, Confidence.
        var row = Row(
            ("x1", "7.148"), ("x2", "3.339"),
            ("Score", "26.47"),
            ("ScoreLowerBound", "24.97"), ("ScoreUpperBound", "27.97"),
            ("Confidence", "0.130759"));

        Assert.Equal("26.47", PredictCommand.SelectPredictionValue(row));
    }

    [Fact]
    public void Classification_SelectsPredictedLabel_OverScoreVectorAndProbability()
    {
        var row = Row(
            ("x1", "1.0"),
            ("PredictedLabel", "true"),
            ("Score.0", "-2.1"), ("Score.1", "2.1"),
            ("Probability", "0.89"),
            ("Confidence", "0.89"));

        Assert.Equal("true", PredictCommand.SelectPredictionValue(row));
    }

    [Fact]
    public void PlainRegression_SelectsScore()
    {
        var row = Row(("x1", "3.0"), ("Score", "6.0"));
        Assert.Equal("6.0", PredictCommand.SelectPredictionValue(row));
    }

    [Fact]
    public void NoNamedPredictionColumn_FallsBackToLastColumn()
    {
        // Defensive legacy path: an output without PredictedLabel/Score keeps the old last-column behavior.
        var row = Row(("a", "1"), ("b", "2"), ("c", "3"));
        Assert.Equal("3", PredictCommand.SelectPredictionValue(row));
    }

    [Fact]
    public void TimeSeriesAnomaly_VectorOutput_SkipsAppendedConfidenceInFallback()
    {
        // time-series-anomaly emits a "Prediction" vector (Prediction.0=alert, .1=raw score, .2=detector
        // slot) — no PredictedLabel/scalar Score — so selection hits the fallback. With the additive
        // Confidence column trailing, the fallback must skip it and log the real last output slot
        // (Prediction.2), preserving pre-change behavior rather than logging the confidence.
        var row = Row(
            ("Prediction.0", "1"), ("Prediction.1", "3.2"), ("Prediction.2", "0.01"),
            ("Confidence", "0.44"));

        Assert.Equal("0.01", PredictCommand.SelectPredictionValue(row));
    }
}
