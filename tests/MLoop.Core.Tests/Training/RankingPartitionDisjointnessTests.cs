using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A ranking run has three kinds of rows and each must be a different one: the rows the candidate trainers are fitted on, the
/// queries they are compared on (a fifth of the training partition's queries) and the test queries the winner's score is reported
/// on. A score cannot show an overlap — a model that has seen a query ranks it well and the comparison or the report reads as
/// quiet — so this looks at the rows themselves: every row carries a key that is unique to it, and the three sets of keys must
/// not meet and must be the whole table (<see cref="PartitionDisjointnessTests"/> does the same for the tasks that split by row).
/// </summary>
/// <remarks>
/// A query is a unit: it is on one side whole, whether its identifier is text or a number. The guard asserts that it can see — a
/// split that hands the same rows to two sides is reported as overlapping by the same comparison, and a table with too few
/// queries to spare any is reported as having no selection partition.
/// </remarks>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class RankingPartitionDisjointnessTests
{
    private const int Queries = 60;
    private const int Documents = 10;
    private const int Seed = 11;

    private sealed class TextRow
    {
        public string Query { get; set; } = "";
        public float Key { get; set; }
        public float F0 { get; set; }
        public float Label { get; set; }
    }

    private sealed class NumberRow
    {
        public float Query { get; set; }
        public float Key { get; set; }
        public float F0 { get; set; }
        public float Label { get; set; }
    }

    /// <summary>Each row's <c>Key</c> is a distinct float, so a row can be told from every other wherever it is read.</summary>
    private static IEnumerable<(int Query, float Key, float F0, float Label)> Rows(int queries)
    {
        var random = new Random(Seed);
        var key = 0;
        for (var q = 0; q < queries; q++)
            for (var d = 0; d < Documents; d++)
            {
                var f0 = (float)(random.NextDouble() * 2 - 1);
                yield return (q, ++key, f0, Math.Clamp((float)Math.Round(2 + 1.5 * f0), 0, 4));
            }
    }

    private static IDataView Table(MLContext ctx, bool numberQueries, int queries) => numberQueries
        ? ctx.Data.LoadFromEnumerable(Rows(queries).Select(r => new NumberRow { Query = r.Query, Key = r.Key, F0 = r.F0, Label = r.Label }))
        : ctx.Data.LoadFromEnumerable(Rows(queries).Select(r => new TextRow { Query = $"q{r.Query}", Key = r.Key, F0 = r.F0, Label = r.Label }));

    private static HashSet<float> Keys(IDataView view) => [.. view.GetColumn<float>("Key")];

    private static HashSet<string> QueriesOf(IDataView view, bool numberQueries) => numberQueries
        ? [.. view.GetColumn<float>("Query").Select(q => q.ToString(System.Globalization.CultureInfo.InvariantCulture))]
        : [.. view.GetColumn<string>("Query")];

    private static TrainingConfig Config() => new() { ModelName = "rank", DataFile = "synthetic", LabelColumn = "Label", GroupColumn = "Query", Task = "ranking" };

    // The partitions a run makes, as the run makes them: the test queries are held out of the table, then a fifth of the
    // training partition's queries are set aside from the fitting.
    private static (IDataView Fit, IDataView? Selection, IDataView Test, IDataView Train) Partitions(MLContext ctx, IDataView table)
    {
        var (train, test) = GroupedSplit.Of(ctx, table, "Query", 0.2);
        var (fit, selection) = new TrainingRunner(ctx, new CsvDataLoader(ctx)).SelectionSplit(train, Config());
        return (fit, selection, test, train);
    }

    private static void AssertThreeWayDisjointAndWhole(IDataView table, (IDataView Fit, IDataView? Selection, IDataView Test, IDataView Train) p, bool numberQueries)
    {
        Assert.NotNull(p.Selection);
        var (fit, selection, test) = (Keys(p.Fit), Keys(p.Selection!), Keys(p.Test));

        // The guard sees: every partition has rows, and between them they are the whole table.
        var sizes = $"fit {fit.Count}, selection {selection.Count}, test {test.Count} of {Keys(table).Count} rows";
        Assert.True(fit.Count > 0 && selection.Count > 0 && test.Count > 0, sizes);
        Assert.Equal(Keys(table).Count, fit.Count + selection.Count + test.Count);

        Assert.Empty(fit.Intersect(selection));    // the comparison is on rows no candidate was fitted on
        Assert.Empty(fit.Intersect(test));
        Assert.Empty(selection.Intersect(test));   // and the reported score is on rows neither the fitting nor the comparison read

        // A query is whole on its side.
        var (fitQ, selectionQ, testQ) = (QueriesOf(p.Fit, numberQueries), QueriesOf(p.Selection!, numberQueries), QueriesOf(p.Test, numberQueries));
        Assert.Empty(fitQ.Intersect(selectionQ));
        Assert.Empty(fitQ.Intersect(testQ));
        Assert.Empty(selectionQ.Intersect(testQ));
        Assert.Equal((fitQ.Count + selectionQ.Count + testQ.Count) * Documents, fit.Count + selection.Count + test.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheQueriesTheCandidatesAreComparedOn_AreNeitherFittedNorTheReportedOnes(bool numberQueries)
    {
        var ctx = new MLContext(seed: Seed);
        var table = Table(ctx, numberQueries, Queries);

        AssertThreeWayDisjointAndWhole(table, Partitions(ctx, table), numberQueries);
    }

    [Fact]
    public void TheSelectionShare_IsAboutAFifthOfTheTrainingQueries()
    {
        var ctx = new MLContext(seed: Seed);
        var p = Partitions(ctx, Table(ctx, numberQueries: false, Queries));

        var trainQueries = QueriesOf(p.Train, numberQueries: false).Count;
        var selectionQueries = QueriesOf(p.Selection!, numberQueries: false).Count;

        Assert.InRange(selectionQueries / (double)trainQueries, 0.1, 0.3);
    }

    [Fact]
    public void TooFewQueriesToSpare_LeaveNothingToCompareOn_AndTheWholePartitionIsFitted()
    {
        var ctx = new MLContext(seed: Seed);
        var table = Table(ctx, numberQueries: false, queries: 9);

        var (fit, selection) = new TrainingRunner(ctx, new CsvDataLoader(ctx)).SelectionSplit(table, Config());

        Assert.Null(selection);
        Assert.Equal(Keys(table), Keys(fit));
    }

    [Fact]
    public void TheGuard_SeesAComparisonOnRowsTheCandidatesWereFitted_AndAReportOnRowsTheComparisonRead()
    {
        // The shapes the invariant forbids, handed to the same assertions: a comparison on the whole training partition (the fitted
        // rows included), and a report on the compared queries.
        var ctx = new MLContext(seed: Seed);
        var table = Table(ctx, numberQueries: false, Queries);
        var p = Partitions(ctx, table);

        AssertThreeWayDisjointAndWhole(table, p, numberQueries: false);   // the real partitions pass…
        Assert.ThrowsAny<Exception>(() => AssertThreeWayDisjointAndWhole(table, p with { Selection = p.Train }, numberQueries: false));
        Assert.ThrowsAny<Exception>(() => AssertThreeWayDisjointAndWhole(table, p with { Test = p.Selection! }, numberQueries: false));
    }
}
