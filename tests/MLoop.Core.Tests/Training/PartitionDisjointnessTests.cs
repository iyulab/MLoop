using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A run reports its metrics on rows it did not fit. A score cannot show that: the search settles on regularised models that
/// score noise at chance in-sample too (<see cref="StatisticalInvariantTests"/> tried), so a reported partition that overlapped the
/// fitted rows would read as quiet. This looks at the rows themselves — each partition is handed to the runner through a view that
/// records which rows a cursor reads from it, and the two sets must not meet.
/// </summary>
/// <remarks>
/// Every row carries a feature that is unique to it (a random float), which is how a read is attributed to a row. The guard asserts
/// that it can see: each partition's recorded set is the whole partition, and a split that hands the same rows to both sides is
/// reported as overlapping by the same recording.
/// </remarks>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class PartitionDisjointnessTests
{
    private const int Rows = 800;
    private const int Seed = 7;
    private const string Key = "F0";

    private sealed class Row
    {
        public float F0 { get; set; }
        public float F1 { get; set; }
        public float F2 { get; set; }
        public string Cls { get; set; } = "";
        public string Item { get; set; } = "";
        public bool Flag { get; set; }
        public float Value { get; set; }
    }

    private static List<Row> Data(int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, Rows).Select(_ =>
        {
            var f0 = (float)(random.NextDouble() * 2 - 1);
            return new Row
            {
                F0 = f0,
                F1 = (float)(random.NextDouble() * 2 - 1),
                F2 = (float)(random.NextDouble() * 2 - 1),
                Cls = f0 < -0.3f ? "low" : f0 < 0.3f ? "mid" : "high",
                Item = "i" + random.Next(12),
                Flag = f0 > 0,
                Value = 5 + 2 * f0 + (float)random.NextDouble(),
            };
        })];
    }

    /// <summary>The keys of the rows a cursor read through one partition's view.</summary>
    private sealed class Reads
    {
        private readonly object _lock = new();
        public HashSet<float> Keys { get; } = [];
        public void Add(float key) { lock (_lock) Keys.Add(key); }
    }

    private sealed class TrackedView(IDataView inner, Reads reads) : IDataView
    {
        public DataViewSchema Schema => inner.Schema;
        public bool CanShuffle => inner.CanShuffle;
        public long? GetRowCount() => inner.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            var key = inner.Schema[Key];
            return new TrackedCursor(inner.GetRowCursor(columnsNeeded.Append(key).Distinct(), rand), key, reads);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        {
            var key = inner.Schema[Key];
            return [.. inner.GetRowCursorSet(columnsNeeded.Append(key).Distinct(), n, rand).Select(c => (DataViewRowCursor)new TrackedCursor(c, key, reads))];
        }
    }

    private sealed class TrackedCursor : DataViewRowCursor
    {
        private readonly DataViewRowCursor _inner;
        private readonly ValueGetter<float> _key;
        private readonly Reads _reads;

        public TrackedCursor(DataViewRowCursor inner, DataViewSchema.Column key, Reads reads)
        {
            _inner = inner;
            _key = inner.GetGetter<float>(key);
            _reads = reads;
        }

        public override DataViewSchema Schema => _inner.Schema;
        public override long Position => _inner.Position;
        public override long Batch => _inner.Batch;

        public override bool MoveNext()
        {
            if (!_inner.MoveNext())
                return false;
            float key = 0;
            _key(ref key);
            _reads.Add(key);
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => _inner.GetGetter<TValue>(column);
        public override ValueGetter<DataViewRowId> GetIdGetter() => _inner.GetIdGetter();
        public override bool IsColumnActive(DataViewSchema.Column column) => _inner.IsColumnActive(column);
        protected override void Dispose(bool disposing)
        {
            if (disposing)
                _inner.Dispose();
            base.Dispose(disposing);
        }
    }

    /// <summary>Splits as the product's loader does, and hands each side to the runner through a recording view — or, for the self-check, the same side twice.</summary>
    private sealed class RecordingProvider(MLContext ctx, IDataView data, bool overlapOnPurpose = false) : IDataProvider
    {
        public Reads Train { get; } = new();
        public Reads Test { get; } = new();

        public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
            IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null, DataKind? labelKind = null) => data;
        public bool ValidateLabelColumn(IDataView d, string labelColumn) => d.Schema.GetColumnOrNull(labelColumn).HasValue;
        public DataSchema GetSchema(IDataView d) => new() { Columns = [], RowCount = 0 };

        public (IDataView trainSet, IDataView testSet) SplitData(IDataView d, double testFraction = 0.2)
        {
            var split = ctx.Data.TrainTestSplit(d, testFraction: testFraction, seed: Seed);
            var train = new TrackedView(split.TrainSet, Train);
            return (train, overlapOnPurpose ? new TrackedView(split.TrainSet, Test) : new TrackedView(split.TestSet, Test));
        }
    }

    private static async Task<RecordingProvider> RunAsync(string task, string label, bool overlapOnPurpose = false,
        string? userColumn = null, string? itemColumn = null)
    {
        var ctx = new MLContext(seed: Seed);
        var provider = new RecordingProvider(ctx, ctx.Data.LoadFromEnumerable(Data(Seed)), overlapOnPurpose);
        await new TrainingRunner(ctx, provider).RunAsync(new TrainingConfig
        {
            ModelName = "disjoint",
            DataFile = "synthetic",
            LabelColumn = label,
            Task = task,
            TimeLimitSeconds = 60,
            MaxTrials = 3,
            UserColumn = userColumn,
            ItemColumn = itemColumn,
        });
        return provider;
    }

    private static void AssertDisjointAndWhole(RecordingProvider provider)
    {
        // The guard sees: both partitions were read, and between them they are the whole table (a view that recorded nothing
        // would pass the disjointness below for the wrong reason).
        Assert.NotEmpty(provider.Train.Keys);
        Assert.NotEmpty(provider.Test.Keys);
        Assert.Equal(Rows, provider.Train.Keys.Count + provider.Test.Keys.Count);

        var shared = provider.Train.Keys.Intersect(provider.Test.Keys).Count();
        Assert.True(shared == 0, $"{shared} rows were read from both the training and the reported partition");
    }

    // Each partition carries its label under its own column name, so one table serves the three tasks.
    [Fact]
    public async Task Binary_TheRowsTheMetricsAreReportedOn_AreNotRowsTheModelWasFitted() =>
        AssertDisjointAndWhole(await RunAsync("binary-classification", nameof(Row.Flag)));

    [Fact]
    public async Task Multiclass_TheRowsTheMetricsAreReportedOn_AreNotRowsTheModelWasFitted() =>
        AssertDisjointAndWhole(await RunAsync("multiclass-classification", nameof(Row.Cls)));

    [Fact]
    public async Task Regression_TheRowsTheMetricsAreReportedOn_AreNotRowsTheModelWasFitted() =>
        AssertDisjointAndWhole(await RunAsync("regression", nameof(Row.Value)));

    // A recommendation is judged on rows too: its user and item columns are keys, not features, and the split is by row.
    [Fact]
    public async Task Recommendation_TheRowsTheMetricsAreReportedOn_AreNotRowsTheModelWasFitted() =>
        AssertDisjointAndWhole(await RunAsync("recommendation", nameof(Row.Value), userColumn: nameof(Row.Cls), itemColumn: nameof(Row.Item)));

    [Fact]
    public async Task TheRecording_SeesAPartitionThatReportsOnItsOwnFittedRows()
    {
        // The shape the invariant forbids: the reported side is the fitted side. The same recording must say so.
        var provider = await RunAsync("regression", nameof(Row.Value), overlapOnPurpose: true);

        Assert.NotEmpty(provider.Test.Keys);
        Assert.NotEmpty(provider.Train.Keys.Intersect(provider.Test.Keys));
    }
}
