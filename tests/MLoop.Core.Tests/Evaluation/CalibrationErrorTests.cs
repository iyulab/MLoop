using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class CalibrationErrorTests
{
    [Fact]
    public void CalibratedProbabilitiesHaveNoError()
    {
        // In the 0.7–0.8 bin, predictions of 0.75 are right three times out of four.
        var pairs = new[] { (0.75, true), (0.75, true), (0.75, true), (0.75, false) };

        Assert.Equal(0, CalibrationError.Compute(pairs)!.Value, precision: 10);
    }

    [Fact]
    public void OverconfidenceIsTheGapWeightedByBinShare()
    {
        // Half the rows say 0.95 and are right half the time (gap 0.45); half say 0.15 and the
        // positive rate there is 0.0 (gap 0.15). ECE = 0.5·0.45 + 0.5·0.15 = 0.30.
        var pairs = new[] { (0.95, true), (0.95, false), (0.15, false), (0.15, false) };

        Assert.Equal(0.30, CalibrationError.Compute(pairs)!.Value, precision: 10);
    }

    [Fact]
    public void NoPredictionsHaveNoError() => Assert.Null(CalibrationError.Compute([]));

    [Fact]
    public void BinaryReadsTheProbabilityColumn()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new BinaryRow { Label = true, Probability = 0.9f }, new BinaryRow { Label = false, Probability = 0.9f },
        });

        // Both rows in the 0.9 bin: said 0.9, positive half the time.
        Assert.Equal(0.4, CalibrationError.Binary(data, "Label")!.Value, precision: 5);
    }

    [Fact]
    public void BinaryWithoutProbabilitiesHasNoError()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[] { new ScoreOnlyRow { Label = true, Score = 3.2f } });

        Assert.Null(CalibrationError.Binary(data, "Label"));
    }

    [Fact]
    public void MulticlassUsesTheTopClassAgainstTheKeyedLabel()
    {
        var ml = new MLContext(seed: 1);
        var rows = ml.Data.LoadFromEnumerable(new[]
        {
            new MulticlassRow { Label = "a", Score = [0.8f, 0.1f, 0.1f] }, // top = a, right
            new MulticlassRow { Label = "b", Score = [0.8f, 0.1f, 0.1f] }, // top = a, wrong
            new MulticlassRow { Label = "c", Score = [0.1f, 0.1f, 0.8f] }, // top = c, right
        });
        var keyed = ml.Transforms.Conversion.MapValueToKey("Label").Fit(rows).Transform(rows);

        // All three in the 0.8 bin, right two times in three: |0.667 − 0.8| = 0.1333.
        Assert.Equal(0.8 - 2.0 / 3, CalibrationError.Multiclass(keyed, "Label")!.Value, precision: 5);
    }

    [Fact]
    public void ItIsReportedAsLowerIsBetterAndNeverOptimized()
    {
        Assert.True(MetricDirection.IsLowerBetter(CalibrationError.MetricName));
        Assert.True(MetricNames.IsKnown(CalibrationError.MetricName));
        Assert.NotNull(MetricNames.Rejection("binary-classification", CalibrationError.MetricName));
    }

    private sealed class BinaryRow { public bool Label { get; set; } public float Probability { get; set; } }
    private sealed class ScoreOnlyRow { public bool Label { get; set; } public float Score { get; set; } }
    private sealed class MulticlassRow
    {
        public string Label { get; set; } = "";
        [VectorType(3)] public float[] Score { get; set; } = [];
    }
}
