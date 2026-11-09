using Microsoft.ML;

namespace MLoop.Core.Data;

/// <summary>
/// The train/test split of a task whose rows belong to a group that is judged whole: a ranking's query. Rows of one
/// query go to one side, as the ranking benchmarks' folds do — a split by row puts some documents of a query in training
/// and the rest in the test, and a model that has seen a query's other documents (its query-level features, how its
/// labels run) is scored better on the held-out ones than on a query it has never met. The ranking metric the run
/// reports is then a score for a case that does not occur: ranking new queries.
/// </summary>
public static class GroupedSplit
{
    /// <summary>The seed every split of this product uses, so a project trains on the same partitions each time.</summary>
    public const int Seed = 42;

    /// <summary>Whether a single-file run of <paramref name="task"/> is split by <paramref name="groupColumn"/> in <paramref name="data"/>.</summary>
    public static bool Applies(string? task, string? groupColumn, IDataView data) =>
        Models.TaskTypes.Canonical(task ?? "") == "ranking"
        && !string.IsNullOrWhiteSpace(groupColumn)
        && data.Schema.GetColumnOrNull(groupColumn) is not null;

    /// <summary>
    /// Splits <paramref name="data"/> so that no value of <paramref name="groupColumn"/> is on both sides.
    /// A fraction of 0 gives the whole data to both sides, as <c>IDataProvider.SplitData</c> does. The group is placed by a
    /// hash of its value and <paramref name="seed"/>: a second split of one of the sides must use another seed, or it hashes
    /// the same values to the same places and gives all of them to the side they were already on.
    /// </summary>
    public static (IDataView Train, IDataView Test) Of(MLContext ml, IDataView data, string groupColumn, double testFraction, int seed = Seed)
    {
        if (testFraction <= 0)
            return (data, data);
        if (testFraction >= 1)
            throw new ArgumentException("testFraction must be between 0 and 1 (exclusive).", nameof(testFraction));

        var split = ml.Data.TrainTestSplit(data, testFraction: testFraction, samplingKeyColumnName: groupColumn, seed: seed);
        return (split.TrainSet, split.TestSet);
    }
}
