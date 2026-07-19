using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// F-22 regression guard: clustering's K-search uses the Davies-Bouldin Index to pick the natural
/// number of clusters. ML.NET only computes DBI when <c>featureColumnName</c> is passed to
/// <c>Clustering.Evaluate</c>; without it DBI came back 0, so the "use DBI" branch was never taken
/// and the search silently fell back to average_distance (which monotonically decreases with K) and
/// always picked the largest K. This test feeds three well-separated clusters and asserts the search
/// (a) computes a real DBI (&gt; 0) and (b) selects K=3 rather than the search ceiling.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class ClusteringKSearchRegressionTests : IDisposable
{
    private readonly string _tmpDir;

    public ClusteringKSearchRegressionTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_clk_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private sealed class Point
    {
        public float F1 { get; set; }
        public float F2 { get; set; }
    }

    private sealed class InMemoryDataProvider : IDataProvider
    {
        private readonly MLContext _ctx;
        private readonly IDataView _fullData;

        public InMemoryDataProvider(MLContext ctx, IDataView fullData)
        {
            _ctx = ctx;
            _fullData = fullData;
        }

        public IDataView LoadData(string filePath, string? labelColumn = null,
            string? taskType = null, IEnumerable<string>? preserveColumns = null,
            IReadOnlyCollection<string>? featureExclusions = null)
            => _fullData;

        public bool ValidateLabelColumn(IDataView data, string labelColumn)
            => data.Schema.GetColumnOrNull(labelColumn).HasValue;

        public DataSchema GetSchema(IDataView data)
        {
            var cols = data.Schema.Where(c => !c.IsHidden)
                .Select(c => new ColumnInfo
                {
                    Name = c.Name,
                    Type = DataTypeHelper.GetFriendlyTypeName(c.Type),
                    IsLabel = false
                }).ToList();
            return new DataSchema { Columns = cols, RowCount = 0 };
        }

        public (IDataView trainSet, IDataView testSet) SplitData(IDataView data, double testFraction = 0.2)
        {
            var split = _ctx.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42);
            return (split.TrainSet, split.TestSet);
        }
    }

    [Fact]
    public async Task ClusteringAutoK_computesDaviesBouldin_andPicksNaturalK()
    {
        var ctx = new MLContext(seed: 42);

        // Three tight, far-apart clusters → the natural K is unambiguously 3.
        var rng = new Random(7);
        (float, float)[] centers = [(0f, 0f), (30f, 30f), (60f, 0f)];
        var rows = new List<Point>();
        for (int c = 0; c < centers.Length; c++)
            for (int i = 0; i < 120; i++)
                rows.Add(new Point
                {
                    F1 = centers[c].Item1 + (float)(rng.NextDouble() - 0.5),
                    F2 = centers[c].Item2 + (float)(rng.NextDouble() - 0.5)
                });

        var dataView = ctx.Data.LoadFromEnumerable(rows);

        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = "synthetic",   // ignored by InMemoryDataProvider
            LabelColumn = "",         // unsupervised — every numeric column is a feature
            Task = "clustering",
            TimeLimitSeconds = 30,
            NumClusters = 0           // auto-search k=2..maxK
        };

        var runner = new AutoMLRunner(ctx, new InMemoryDataProvider(ctx, dataView), _tmpDir);
        var result = await runner.RunAsync(config, cancellationToken: CancellationToken.None);

        Assert.NotNull(result);
        Assert.NotEmpty(result.Metrics);

        // F-22 core: DBI is actually computed (was a constant 0 before featureColumnName was passed).
        Assert.True(result.Metrics.TryGetValue("davies_bouldin_index", out var dbi),
            $"Expected davies_bouldin_index. Keys: [{string.Join(", ", result.Metrics.Keys)}]");
        Assert.True(dbi > 0, $"Davies-Bouldin Index should be a real positive value, got {dbi}");

        // F-22 consequence: the DBI-driven search finds the natural K=3, not the search ceiling.
        Assert.Equal(3, (int)result.Metrics["num_clusters"]);
    }
}
