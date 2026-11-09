using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class MissingNumbersTests
{
    private sealed class Point
    {
        public float Value { get; set; }

        [VectorType(2)]
        public float[] Pair { get; set; } = [];
    }

    private static IDataView Data(MLContext ml, params (float Value, float[] Pair)[] rows) =>
        ml.Data.LoadFromEnumerable(rows.Select(r => new Point { Value = r.Value, Pair = r.Pair }));

    [Fact]
    public void CountIn_CountsEveryMissingValue_EachSlotOfAVector()
    {
        var ml = new MLContext(1);
        var data = Data(ml, (1, [float.NaN, float.NaN]), (float.NaN, [2, 3]), (float.NaN, [float.NaN, 4]));

        Assert.Equal(2, MissingNumbers.CountIn(data, "Value"));
        Assert.Equal(3, MissingNumbers.CountIn(data, "Pair"));
    }

    [Fact]
    public void ASeriesWithAGap_IsRefused_NamingTheColumnTheCountAndTheFix()
    {
        var ml = new MLContext(1);
        var data = Data(ml, (1, [0, 0]), (float.NaN, [0, 0]), (3, [0, 0]));

        var refusal = Assert.Throws<InvalidOperationException>(() => MissingNumbers.RefuseGapsInSeries(data, "Value"));

        Assert.Contains("'Value' has 1 missing value (data row 2).", refusal.Message);
        Assert.Contains("fill-missing:forward --columns Value", refusal.Message);
    }

    [Fact]
    public void ASeriesWithoutAGap_Passes()
    {
        var ml = new MLContext(1);

        MissingNumbers.RefuseGapsInSeries(Data(ml, (1, [0, 0]), (2, [0, 0])), "Value");
    }

    [Fact]
    public void ASeriesWithSeveralGaps_NamesTheFirstFewRows()
    {
        var ml = new MLContext(1);
        var rows = Enumerable.Range(1, 12).Select(i => (i % 2 == 0 ? float.NaN : (float)i, new float[] { 0, 0 })).ToArray();
        var refusal = Assert.Throws<InvalidOperationException>(() => MissingNumbers.RefuseGapsInSeries(Data(ml, rows), "Value"));

        Assert.Contains("has 6 missing values (the first at data rows 2, 4, 6, 8, 10).", refusal.Message);
        Assert.Equal<long>([2, 4, 6], MissingNumbers.FirstGapRows(Data(ml, rows), "Value", 3));
    }
}
