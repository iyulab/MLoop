using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Prediction;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// How many components a table's anomaly model keeps: the fewest that carry 90 % of the variance, always fewer than the
/// columns (a rebuild with every column is exact, so every score is rounding noise).
/// </summary>
public class AnomalyPcaRankTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(5, 4)]
    [InlineData(20, 19)]
    [InlineData(21, 20)]
    [InlineData(500, 20)]
    public void TheCeiling_LeavesARebuildErrorToScore_AndStopsAtTwenty(int columns, int expected)
        => Assert.Equal(expected, AnomalyPcaRank.Ceiling(columns));

    [Fact]
    public void TheRank_IsTheFewestComponentsThatReachNinetyPercent_InTheOrderOfTheirVariance()
    {
        // 6 + 3 + 0.6 + 0.4 = 10: the first carries 60 %, the first two 90 %.
        Assert.Equal(2, AnomalyPcaRank.Choose(new[] { 0.6, 6.0, 0.4, 3.0 }, 10, ceiling: 3));
        // 89 % after two, 98 % after three.
        Assert.Equal(3, AnomalyPcaRank.Choose(new[] { 5.0, 3.9, 1.0, 0.1 }, 10, ceiling: 3));
    }

    [Fact]
    public void TheRank_NeverPassesTheCeiling_AndFallsBackToItWhereNothingIsShared()
    {
        // Ten equal components: 90 % takes nine, the ceiling of this table is four.
        Assert.Equal(4, AnomalyPcaRank.Choose(Enumerable.Repeat(1.0, 10).ToArray(), 10, ceiling: 4));
        Assert.Equal(3, AnomalyPcaRank.Choose(new[] { 0.0, 0.0 }, 0, ceiling: 3));
        Assert.Equal(3, AnomalyPcaRank.Choose(new[] { 1.0 }, double.NaN, ceiling: 3));
    }

    [Fact]
    public void TheCarriedShare_IsWhatTheKeptComponentsCarry_AndNullWhereThereIsNothingToShare()
    {
        // 6 + 3 + 0.6 + 0.4 = 10: two components carry 90 %, one carries 60 %.
        var variance = new[] { 0.6, 6.0, 0.4, 3.0 };
        Assert.Equal(0.9, AnomalyPcaRank.CarriedShare(variance, 10, rank: 2)!.Value, 6);
        Assert.Equal(0.6, AnomalyPcaRank.CarriedShare(variance, 10, rank: 1)!.Value, 6);
        // Ten equal components held to a ceiling of four: the rank fell back, and the share says so (0.4 < 0.9).
        var flat = Enumerable.Repeat(1.0, 10).ToArray();
        var rank = AnomalyPcaRank.Choose(flat, 10, ceiling: 4);
        Assert.True(AnomalyPcaRank.CarriedShare(flat, 10, rank)!.Value < AnomalyPcaRank.ExplainedShare);
        // A rank that reached 90 % never reports below it.
        var reached = AnomalyPcaRank.Choose(variance, 10, ceiling: 3);
        Assert.True(AnomalyPcaRank.CarriedShare(variance, 10, reached)!.Value >= AnomalyPcaRank.ExplainedShare);
        Assert.Null(AnomalyPcaRank.CarriedShare(variance, 0, rank: 2));
        Assert.Null(AnomalyPcaRank.CarriedShare(variance, double.NaN, rank: 2));
    }

    private sealed class Row
    {
        [VectorType(4)]
        public float[] Features { get; set; } = [];
    }

    [Fact]
    public void ThreeColumnsOfOneSignalAndOneOfNoise_NeedTwoComponents()
    {
        // Three columns that are one signal (standardized: equal) and one that is noise: the variance is about
        // 3 + 1 + a little, so one component carries 74 %, two carry more than 99 %.
        var ml = new MLContext(seed: 1);
        var random = new Random(5);
        var rows = Enumerable.Range(0, 800).Select(_ =>
        {
            var z = random.NextDouble() * 2 - 1;
            float Jitter() => (float)(z + 0.02 * (random.NextDouble() - 0.5));
            return new Row { Features = [Jitter(), Jitter(), Jitter(), (float)(random.NextDouble() * 2 - 1)] };
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows, SchemaDefinition.Create(typeof(Row)));
        var standardized = ml.Transforms.NormalizeMeanVariance("Features", fixZero: false).Fit(data).Transform(data);

        var ceiling = AnomalyPcaRank.Ceiling(4);
        var fitted = ml.AnomalyDetection.Trainers.RandomizedPca("Features", rank: ceiling, oversampling: 20).Fit(standardized);
        var (component, total) = AnomalyPcaRank.Measure(standardized, fitted.Model);

        Assert.Equal(3, component.Length);
        Assert.Equal(4.0, total / 800, 1);                           // four standardized columns: a variance of 4 in all directions
        Assert.True(component.Max() / total is > 0.70 and < 0.78);   // one signal shared by three of four columns
        Assert.Equal(2, AnomalyPcaRank.Choose(component, total, ceiling));
    }

    private sealed class Scored
    {
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
    }

    [Fact]
    public void ARowWithoutAScore_IsNotCalledNotAnomalous()
    {
        // A score that is not a number — a row the model could not judge — used to arrive as isAnomaly:false with no
        // score, which reads as "judged, and fine". It answers null, as a time series' unjudged rows do.
        var ml = new MLContext();
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Scored { PredictedLabel = false, Score = 0.1f },
            new Scored { PredictedLabel = false, Score = float.NaN },
            new Scored { PredictedLabel = true, Score = 0.9f },
        });
        using var cursor = data.GetRowCursor(data.Schema);

        var rows = PredictionService.ExtractAnomalyRows(cursor, data.Schema["PredictedLabel"], data.Schema["Score"]);

        Assert.Equal(false, rows[0].IsAnomaly);
        Assert.Null(rows[1].IsAnomaly);
        Assert.Null(rows[1].AnomalyScore);
        Assert.Equal(true, rows[2].IsAnomaly);
    }
}
