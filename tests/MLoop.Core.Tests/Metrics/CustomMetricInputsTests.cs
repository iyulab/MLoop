using Microsoft.ML;
using MLoop.Core.Metrics;
using MLoop.Core.Tests;

namespace MLoop.Core.Tests.Metrics;

/// <summary>
/// The arrays a metric script receives have the types <c>MetricContext</c> documents, one entry per
/// labeled row, predictions and labels aligned.
/// </summary>
public class CustomMetricInputsTests
{
    private sealed class Row
    {
        public float X { get; set; }
        public bool Flag { get; set; }
        public float Value { get; set; }
        public string Kind { get; set; } = "";
    }

    private static readonly MLContext Ml = new(seed: 1);

    private static IDataView Data()
    {
        var rows = Enumerable.Range(0, 60).Select(i => new Row
        {
            X = i / 60f,
            Flag = i % 2 == 0,
            Value = i * 0.5f,
            Kind = (i % 3) switch { 0 => "a", 1 => "b", _ => "c" },
        });
        return Ml.Data.LoadFromEnumerable(rows);
    }

    [Fact]
    public void Binary_IsBoolPredictionsAndLabels()
    {
        var data = Data();
        var model = Ml.Transforms.Concatenate("Features", "X")
            .Append(Ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("Flag")).Fit(data);

        var (predictions, labels) = CustomMetricInputs.Of(model.Transform(data), "binary-classification", "Flag")!.Value;

        var p = Assert.IsType<bool[]>(predictions);
        var y = Assert.IsType<bool[]>(labels);
        Assert.Equal(60, p.Length);
        Assert.Equal(30, y.Count(v => v));
    }

    [Fact]
    public void Multiclass_IsClassIndicesOfTheScoreSlots()
    {
        var data = Data();
        var model = Ml.Transforms.Conversion.MapValueToKey("Kind")
            .Append(Ml.Transforms.Concatenate("Features", "X", "Value"))
            .Append(Ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy("Kind")).Fit(data);

        var (predictions, labels) = CustomMetricInputs.Of(model.Transform(data), "multiclass-classification", "Kind")!.Value;

        var p = Assert.IsType<int[]>(predictions);
        var y = Assert.IsType<int[]>(labels);
        Assert.Equal(60, p.Length);
        Assert.Equal([0, 1, 2], y.Distinct().Order());
        Assert.All(p, c => Assert.InRange(c, 0, 2));
    }

    [Fact]
    public void Regression_IsFloatScoresAndLabels()
    {
        var data = Data();
        var model = Ml.Transforms.Concatenate("Features", "X")
            .Append(Ml.Regression.Trainers.SingleThreadSdca("Value")).Fit(data);

        var (predictions, labels) = CustomMetricInputs.Of(model.Transform(data), "regression", "Value")!.Value;

        Assert.Equal(60, Assert.IsType<float[]>(predictions).Length);
        Assert.Equal(29.5f, Assert.IsType<float[]>(labels).Max());
    }

    [Fact]
    public void ATaskTheContractDoesNotCover_IsNull()
        => Assert.Null(CustomMetricInputs.Of(Data(), "clustering", "Value"));
}
