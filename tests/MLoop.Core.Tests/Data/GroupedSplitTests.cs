using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>A ranking's single-file split holds a query out whole - never a query's documents on both sides.</summary>
public sealed class GroupedSplitTests
{
    private sealed class TextQueryRow
    {
        public string Query { get; set; } = "";
        public float Feature { get; set; }
        public float Label { get; set; }
    }

    private sealed class NumberQueryRow
    {
        public float Query { get; set; }
        public float Feature { get; set; }
        public float Label { get; set; }
    }

    private static IDataView TextRows(MLContext ml, int queries = 60, int documents = 8) =>
        ml.Data.LoadFromEnumerable(Enumerable.Range(0, queries * documents).Select(i =>
            new TextQueryRow { Query = $"q{i / documents}", Feature = i % 7, Label = i % 5 }));

    private static IDataView NumberRows(MLContext ml, int queries = 60, int documents = 8) =>
        ml.Data.LoadFromEnumerable(Enumerable.Range(0, queries * documents).Select(i =>
            new NumberQueryRow { Query = i / documents, Feature = i % 7, Label = i % 5 }));

    private static HashSet<string> Queries(IDataView data, string column)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var value in data.GetColumn<string>(column)) seen.Add(value);
        return seen;
    }

    private static HashSet<string> NumberQueries(IDataView data, string column) =>
        data.GetColumn<float>(column).Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToHashSet(StringComparer.Ordinal);

    [Fact]
    public void ATextQueryIsOnOneSideOnly()
    {
        var ml = new MLContext(seed: 1);
        var (train, test) = GroupedSplit.Of(ml, TextRows(ml), "Query", 0.2);

        var trainQueries = Queries(train, "Query");
        var testQueries = Queries(test, "Query");
        Assert.NotEmpty(trainQueries);
        Assert.NotEmpty(testQueries);
        Assert.Empty(trainQueries.Intersect(testQueries));
        // a query is whole on its side: every one of its 8 documents
        Assert.Equal(trainQueries.Count * 8, train.GetColumn<string>("Query").Count());
        Assert.Equal(testQueries.Count * 8, test.GetColumn<string>("Query").Count());
    }

    [Fact]
    public void ANumericQueryIsOnOneSideOnly()
    {
        var ml = new MLContext(seed: 1);
        var (train, test) = GroupedSplit.Of(ml, NumberRows(ml), "Query", 0.2);

        var trainQueries = NumberQueries(train, "Query");
        var testQueries = NumberQueries(test, "Query");
        Assert.NotEmpty(trainQueries);
        Assert.NotEmpty(testQueries);
        Assert.Empty(trainQueries.Intersect(testQueries));
        Assert.Equal(60, trainQueries.Count + testQueries.Count);
    }

    [Fact]
    public void TheHeldOutShareFollowsTheFractionAndTheSplitRepeats()
    {
        var ml = new MLContext(seed: 1);
        var (_, test) = GroupedSplit.Of(ml, TextRows(ml, queries: 200), "Query", 0.25);
        var again = GroupedSplit.Of(new MLContext(seed: 9), TextRows(new MLContext(seed: 9), queries: 200), "Query", 0.25).Test;

        var share = Queries(test, "Query").Count / 200.0;
        Assert.InRange(share, 0.15, 0.35);
        Assert.Equal(Queries(test, "Query").OrderBy(q => q, StringComparer.Ordinal), Queries(again, "Query").OrderBy(q => q, StringComparer.Ordinal));
    }

    [Fact]
    public void ANoHeldOutFractionGivesTheWholeDataToBothSides()
    {
        var ml = new MLContext(seed: 1);
        var data = TextRows(ml);
        var (train, test) = GroupedSplit.Of(ml, data, "Query", 0);

        Assert.Same(data, train);
        Assert.Same(data, test);
        Assert.Throws<ArgumentException>(() => GroupedSplit.Of(ml, data, "Query", 1.0));
    }

    [Theory]
    [InlineData("ranking", "Query", true)]
    [InlineData("Ranking", "Query", true)]
    [InlineData("ranking", "Missing", false)]
    [InlineData("ranking", "", false)]
    [InlineData("ranking", null, false)]
    [InlineData("recommendation", "Query", false)]
    [InlineData("regression", "Query", false)]
    [InlineData("binary-classification", "Query", false)]
    public void OnlyARankingWithItsGroupColumnInTheDataSplitsByGroup(string task, string? groupColumn, bool expected)
    {
        var ml = new MLContext(seed: 1);

        Assert.Equal(expected, GroupedSplit.Applies(task, groupColumn, TextRows(ml)));
    }
}
