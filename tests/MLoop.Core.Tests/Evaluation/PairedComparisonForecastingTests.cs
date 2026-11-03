using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// A forecaster is compared with another by the steps of its held-out stretch: the last <c>horizon</c> rows of the file, each
/// scored against the forecast made at the row before them. Every earlier row is NaN, so both models are judged on the same steps.
/// </summary>
public sealed class PairedComparisonForecastingTests
{
    private const int Horizon = 3;

    private sealed class Row
    {
        public float value { get; set; }

        [VectorType(Horizon)]
        public float[] ForecastedValues { get; set; } = new float[Horizon];
    }

    private static IDataView Scored(params (float Value, float[] Forecast)[] rows)
        => new MLContext(seed: 1).Data.LoadFromEnumerable(rows.Select(r => new Row { value = r.Value, ForecastedValues = r.Forecast }));

    [Fact]
    public void AForecaster_IsMeasuredByTheSquaredErrorOfItsHeldOutSteps()
        => Assert.Equal("squared_error", PairedComparison.MeasureOf("forecasting"));

    [Fact]
    public void TheHeldOutRows_AreScoredAgainstTheForecastMadeBeforeThem()
    {
        // rows 0..3 are history; the stretch is rows 3..5 (actual 10, 20, 30) and the forecast that looks at it is row 2's
        var scored = Scored(
            (1, [0, 0, 0]),
            (2, [0, 0, 0]),
            (3, [11, 22, 33]),     // made at the row before the stretch
            (10, [99, 99, 99]),    // the later rows' forecasts are of points beyond the file: not used
            (20, [99, 99, 99]),
            (30, [99, 99, 99]));

        var losses = PairedComparison.RowLosses(scored, "forecasting", "value")!;

        Assert.Equal(6, losses.Length);
        Assert.All(losses.Take(3), loss => Assert.True(double.IsNaN(loss)));
        Assert.Equal([1.0, 4.0, 9.0], losses.Skip(3));
    }

    [Fact]
    public void AFileNoLongerThanTheHorizon_HasNothingHeldOut()
        => Assert.Null(PairedComparison.RowLosses(Scored((1, [1, 1, 1]), (2, [1, 1, 1]), (3, [1, 1, 1])), "forecasting", "value"));

    [Fact]
    public void TwoForecasters_AreComparedOnTheSameSteps()
    {
        var better = Scored((1, [0, 0, 0]), (2, [10, 20, 30]), (10, [0, 0, 0]), (20, [0, 0, 0]), (30, [0, 0, 0]));
        var worse = Scored((1, [0, 0, 0]), (2, [40, 40, 40]), (10, [0, 0, 0]), (20, [0, 0, 0]), (30, [0, 0, 0]));

        var c = PairedComparison.RowLosses(better, "forecasting", "value")!;
        var b = PairedComparison.RowLosses(worse, "forecasting", "value")!;
        var result = PairedComparison.Compare("squared_error", c, b);

        Assert.Equal(3, result.Rows);
        Assert.Equal(PairedVerdict.CandidateBetter, result.Verdict);
    }
}
