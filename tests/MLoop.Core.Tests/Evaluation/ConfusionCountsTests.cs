using Microsoft.ML;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Evaluation;

public class ConfusionCountsTests
{
    private sealed class BinaryRow
    {
        public bool Label { get; set; }
        public bool PredictedLabel { get; set; }
    }

    private static IDataView Binary(params (bool Label, bool Predicted)[] rows)
        => new MLContext(seed: 1).Data.LoadFromEnumerable(rows.Select(r => new BinaryRow { Label = r.Label, PredictedLabel = r.Predicted }));

    private static InputSchemaInfo SchemaWithLabels(params string[] values) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns = [new ColumnSchema { Name = "Label", DataType = SchemaDataTypes.Categorical, Purpose = "Label", CategoricalValues = [.. values] }],
    };

    [Fact]
    public void ABinaryMatrix_NamesItsClassesInTheLabelsOwnWords()
    {
        var scored = Binary((true, true), (true, false), (false, false), (false, false), (false, true));

        var matrix = ConfusionCounts.Of(scored, "binary-classification", "Label", schema: SchemaWithLabels("OK", "NG"))!;

        Assert.Equal(["NG", "OK"], matrix.Classes);           // negative first, as the label conversion assigns them
        Assert.Equal([2.0, 1.0], matrix.Counts[0]);           // actual NG: 2 right, 1 called OK
        Assert.Equal([1.0, 1.0], matrix.Counts[1]);           // actual OK: 1 missed, 1 right
        Assert.Equal(5, matrix.Total);
    }

    [Fact]
    public void WithoutAVocabulary_ABinaryMatrixSaysFalseAndTrue()
        => Assert.Equal(["False", "True"], ConfusionCounts.Of(Binary((true, true)), "binary-classification", "Label")!.Classes);

    [Fact]
    public void Weights_AreSummed_AndRowsWithoutWeightAreLeftOut()
    {
        var scored = Binary((true, true), (true, false), (false, false));

        var matrix = ConfusionCounts.Of(scored, "binary-classification", "Label", weights: [3, 2, 0])!;

        Assert.Equal([0.0, 0.0], matrix.Counts[0]);
        Assert.Equal([2.0, 3.0], matrix.Counts[1]);
    }

    private sealed class MulticlassRow
    {
        public string Label { get; set; } = "";
        [Microsoft.ML.Data.VectorType(3)]
        public float[] Score { get; set; } = [];
    }

    [Fact]
    public void AMulticlassMatrix_ReadsTheClassNamesFromTheLabelKey()
    {
        var ml = new MLContext(seed: 1);
        var raw = ml.Data.LoadFromEnumerable(new[]
        {
            new MulticlassRow { Label = "high", Score = [0.7f, 0.2f, 0.1f] },
            new MulticlassRow { Label = "mid", Score = [0.6f, 0.3f, 0.1f] },    // mid answered high
            new MulticlassRow { Label = "low", Score = [0.1f, 0.1f, 0.8f] },
            new MulticlassRow { Label = "high", Score = [0.1f, 0.1f, 0.8f] },   // high answered low
        });
        var scored = ml.Transforms.Conversion.MapValueToKey("Label").Fit(raw).Transform(raw);

        var matrix = ConfusionCounts.Of(scored, "multiclass-classification", "Label")!;

        Assert.Equal(["high", "mid", "low"], matrix.Classes);
        Assert.Equal([1.0, 0.0, 1.0], matrix.Counts[0]);
        Assert.Equal([1.0, 0.0, 0.0], matrix.Counts[1]);
        Assert.Equal([0.0, 0.0, 1.0], matrix.Counts[2]);
    }

    [Fact]
    public void ShareOf_ReadsAnyClassSetMeasure()
    {
        // rows: actual; columns: predicted — over high, mid, low.
        var matrix = new ConfusionCounts(["high", "mid", "low"],
        [
            [6, 2, 2],
            [1, 8, 1],
            [0, 0, 10],
        ]);

        Assert.Equal(0.2, matrix.ShareOf(["high"], ["low"]));                 // the risky class let through as safe
        Assert.Equal(0.6, matrix.ShareOf(["HIGH"], ["high"]));                // its recall; names ignore case
        Assert.Equal(0.15, matrix.ShareOf(["high", "mid"], ["low"]));         // a set of classes on either side
        Assert.Null(matrix.ShareOf(["none"], ["low"]));                       // no such label: nothing to measure
    }

    [Theory]
    [InlineData("regression")]
    [InlineData("clustering")]
    public void ATaskThatIsNotClassification_HasNoMatrix(string task)
        => Assert.Null(ConfusionCounts.Of(Binary((true, true)), task, "Label"));
}
