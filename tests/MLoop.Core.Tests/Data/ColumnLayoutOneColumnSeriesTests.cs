using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A series is a file of one column, which has no other column to split it from. Training read one (its own
/// branch in the loader) while <c>evaluate</c> asked the layout authority and was refused the file the model
/// had been trained on; the authority now reads a series, and the readers share it. A one-field header is
/// otherwise a file that is not comma-separated, so only a column of numbers all the way is a series.
/// </summary>
public sealed class ColumnLayoutOneColumnSeriesTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_series_{Guid.NewGuid():N}");

    public ColumnLayoutOneColumnSeriesTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(params string[] lines)
    {
        var path = Path.Combine(_dir, $"{Guid.NewGuid():N}.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    [Fact]
    public void AColumnOfNumbers_IsASeriesReadAsSingles_AndIsTheLabelWhenNamed()
    {
        var path = Write("value", "1.5", "2.5", "3", "4");

        var layout = ColumnLayoutInference.Infer(new MLContext(seed: 1), path, "value");

        Assert.Equal("value", layout.Label);
        var column = Assert.Single(layout.Options.Columns);
        Assert.Equal("value", column.Name);
        Assert.Equal(DataKind.Single, column.DataKind);
        Assert.Equal(ColumnLayoutPurpose.Label, Assert.Single(layout.Columns).Purpose);

        var values = new MLContext(seed: 1).Data.CreateTextLoader(layout.Options).Load(path)
            .GetColumn<float>("value").ToArray();
        Assert.Equal(new[] { 1.5f, 2.5f, 3f, 4f }, values);
    }

    [Fact]
    public void WithoutALabel_TheColumnIsANumericFeature()
    {
        var layout = ColumnLayoutInference.Infer(new MLContext(seed: 1), Write("reading", "1", "2", "3"), null);

        Assert.Null(layout.Label);
        Assert.Equal(ColumnLayoutPurpose.Numeric, Assert.Single(layout.Columns).Purpose);
    }

    [Fact]
    public void ALabelThatIsNotTheColumn_IsStillRefused()
    {
        Assert.Throws<ColumnLayoutException>(() => ColumnLayoutInference.Infer(new MLContext(seed: 1), Write("value", "1", "2"), "other"));
    }

    [Fact]
    public void AColumnThatIsNotNumbers_IsNotASeries()
    {
        // a semicolon-separated table reads as a header of one field: it must not be taken for a series
        Assert.Throws<ColumnLayoutException>(() => ColumnLayoutInference.Infer(new MLContext(seed: 1), Write("a;b;c", "1;2;3", "4;5;6"), null));
        Assert.Throws<ColumnLayoutException>(() => ColumnLayoutInference.Infer(new MLContext(seed: 1), Write("review", "great", "broken"), null));
    }

    [Fact]
    public void AHeaderOnly_IsNotASeries()
    {
        Assert.Null(ColumnLayoutInference.OneColumnSeries(Write("value"), "value"));
    }
}
