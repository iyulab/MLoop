using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;

namespace MLoop.Search.Tests;

/// <summary>
/// Missing numbers: replaced by the column's mean on the rows the featurization is fitted on, and — for a
/// column that held missing values in training — marked by an indicator feature.
/// </summary>
public sealed class FeaturizerTests
{
    private sealed class Row
    {
        public float Label { get; set; }
        public float Gappy { get; set; }
        public float Full { get; set; }

        [VectorType(2)]
        public float[] Pair { get; set; } = [];
    }

    private static ColumnRoles Roles() => new("Label",
    [
        KeyValuePair.Create("Gappy", ColumnRole.Numeric),
        KeyValuePair.Create("Full", ColumnRole.Numeric),
        KeyValuePair.Create("Pair", ColumnRole.Numeric),
    ]);

    private static IDataView Data(MLContext ml) => ml.Data.LoadFromEnumerable(new[]
    {
        new Row { Label = 1, Gappy = 2, Full = 10, Pair = [1, 5] },
        new Row { Label = 0, Gappy = float.NaN, Full = 20, Pair = [3, 5] },
        new Row { Label = 1, Gappy = 4, Full = 30, Pair = [float.NaN, 5] },
    });

    [Fact]
    public void ColumnsWithMissingValues_NamesTheNumericColumnsThatHoldOne_ScalarAndVector()
    {
        var ml = new MLContext(1);

        Assert.Equal(new HashSet<string> { "Gappy", "Pair" }, Featurizer.ColumnsWithMissingValues(Data(ml), Roles()));
    }

    [Fact]
    public void AMissingNumber_IsTheColumnMean_AndItsIndicatorIsOne_OnlyForColumnsThatHeldOne()
    {
        var ml = new MLContext(1);
        var data = Data(ml);
        var roles = Roles();

        var featurizer = Featurizer.Build(ml, data.Schema, roles, CategoricalEncoding.OneHot, Featurizer.ColumnsWithMissingValues(data, roles))!;
        var features = featurizer.Fit(data).Transform(data).GetColumn<VBuffer<float>>("Features")
            .Select(v => v.DenseValues().ToArray()).ToArray();

        // Gappy, Full, Pair[0], Pair[1], Gappy.missing, Pair.missing[0], Pair.missing[1] — Full was never missing, so no indicator.
        Assert.All(features, f => Assert.Equal(7, f.Length));
        Assert.Equal([2, 10, 1, 5, 0, 0, 0], features[0]);
        Assert.Equal([3, 20, 3, 5, 1, 0, 0], features[1]);   // Gappy's mean of 2 and 4
        Assert.Equal([4, 30, 2, 5, 0, 1, 0], features[2]);   // Pair[0]'s mean of 1 and 3, per slot
    }

    [Fact]
    public void WithoutMissingValuesInTraining_NoIndicatorIsAdded_AndALaterGapIsTheTrainingMean()
    {
        var ml = new MLContext(1);
        var train = ml.Data.LoadFromEnumerable(new[]
        {
            new Row { Label = 1, Gappy = 2, Full = 10, Pair = [1, 5] },
            new Row { Label = 0, Gappy = 6, Full = 20, Pair = [3, 5] },
        });
        var roles = Roles();

        var withMissing = Featurizer.ColumnsWithMissingValues(train, roles);
        var model = Featurizer.Build(ml, train.Schema, roles, CategoricalEncoding.OneHot, withMissing)!.Fit(train);
        var scored = model.Transform(ml.Data.LoadFromEnumerable(new[] { new Row { Gappy = float.NaN, Full = 10, Pair = [1, 5] } }));

        Assert.Empty(withMissing);
        Assert.Equal([4, 10, 1, 5], scored.GetColumn<VBuffer<float>>("Features").Single().DenseValues().ToArray());
    }

    [Fact]
    public void AnIndicatorName_StepsAside_FromAColumnTheDataAlreadyHas()
    {
        var ml = new MLContext(1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Taken { Label = 1, A = float.NaN, AMissing = 7 },
            new Taken { Label = 0, A = 1, AMissing = 8 },
        });
        var roles = new ColumnRoles("Label", [KeyValuePair.Create("A", ColumnRole.Numeric), KeyValuePair.Create("A.missing", ColumnRole.Numeric)]);

        var featurizer = Featurizer.Build(ml, data.Schema, roles, CategoricalEncoding.OneHot, Featurizer.ColumnsWithMissingValues(data, roles))!;
        var transformed = featurizer.Fit(data).Transform(data);

        Assert.NotNull(transformed.Schema.GetColumnOrNull("A.missing2"));
        Assert.Equal([7f, 8f], transformed.GetColumn<float>("A.missing"));   // the data's own column, untouched
    }

    private sealed class Taken
    {
        public float Label { get; set; }
        public float A { get; set; }

        [ColumnName("A.missing")]
        public float AMissing { get; set; }
    }
}
