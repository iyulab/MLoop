using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Covers the two actionable-throw paths added in upstream-007 stage 2 (commit 716bae2) for
/// when a deep-learning task is requested but <see cref="DeepLearningRegistry"/> has no module
/// registered. <c>MLoop.Core.Tests</c> deliberately never calls <c>DeepLearningRegistry.Register</c>
/// (see <see cref="DeepLearningRegistryTests"/>'s isolation note), so <c>DeepLearningRegistry.Current</c>
/// is null for the whole assembly and both throws below reflect real, unmodified process state.
/// </summary>
public class DeepLearningThrowPathTests
{
    // ---- DataLoaderFactory.Create -------------------------------------------------------

    [Fact]
    public void Create_DirectoryBasedTask_NoRegistry_ThrowsNotSupportedException()
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => DataLoaderFactory.Create("object-detection", new MLContext()));

        Assert.Contains("MLoop.Core.DeepLearning", ex.Message);
    }

    [Fact]
    public void Create_ImageClassificationTask_NoRegistry_ThrowsNotSupportedException()
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => DataLoaderFactory.Create("image-classification", new MLContext()));

        Assert.Contains("MLoop.Core.DeepLearning", ex.Message);
    }

    [Fact]
    public void Create_TabularTask_NoRegistry_DoesNotThrow_ReturnsCsvDataLoader()
    {
        // regression is not directory-based, so the guard at DataLoaderFactory.Create must
        // never be reached for it — this is the negative-control side of the same code path.
        var loader = DataLoaderFactory.Create("regression", new MLContext());

        Assert.IsType<CsvDataLoader>(loader);
    }

    /// <summary>
    /// Fake <see cref="IDeepLearningModule"/> that IS registered but whose
    /// <see cref="CreateDataLoader"/> returns null unconditionally — simulating a future
    /// directory-based task added to <see cref="DataLoaderFactory.IsDirectoryBased"/> without a
    /// corresponding case in the real module's loader switch.
    /// </summary>
    private sealed class NullLoaderDeepLearningModule : IDeepLearningModule
    {
        public bool CanHandleTask(string task) => true;

        public Task<AutoMLResult> TrainAsync(
            MLContext mlContext, Action<string> log, string task,
            IDataView trainSet, IDataView testSet, TrainingConfig config,
            IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
            => throw new NotImplementedException("Not exercised by this test.");

        public IDataProvider? CreateDataLoader(string task, MLContext mlContext, Action<string>? log) => null;
    }

    [Fact]
    public void Create_DirectoryBasedTask_RegisteredModuleReturnsNullLoader_ThrowsNotSupportedException()
    {
        DeepLearningRegistry.Register(new NullLoaderDeepLearningModule());
        try
        {
            var ex = Assert.Throws<NotSupportedException>(
                () => DataLoaderFactory.Create("object-detection", new MLContext()));

            Assert.Contains("object-detection", ex.Message);
            Assert.Contains("MLoop.Core.DeepLearning", ex.Message);
        }
        finally
        {
            // Restore null-registry assumption other Core.Tests (and this class's own
            // no-registry tests) rely on for the rest of the assembly's test run.
            DeepLearningRegistry.Register(null!);
        }
    }

    [Fact]
    public void Create_TabularTask_RegisteredModuleReturnsNullLoader_StillReturnsCsvDataLoader()
    {
        // Negative control: even with a DL module registered that returns null for every task,
        // a non-directory-based task must still fall back to CsvDataLoader unchanged.
        DeepLearningRegistry.Register(new NullLoaderDeepLearningModule());
        try
        {
            var loader = DataLoaderFactory.Create("regression", new MLContext());

            Assert.IsType<CsvDataLoader>(loader);
        }
        finally
        {
            DeepLearningRegistry.Register(null!);
        }
    }

    // ---- AutoMLRunner.RunAsync -> RunDeepLearningOrThrowAsync ----------------------------

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

    private sealed class Row
    {
        public float F1 { get; set; }
        public float Label { get; set; }
    }

    /// <summary>
    /// Exercises the real <see cref="AutoMLRunner.RunAsync"/> path (not just the private method
    /// directly) up to the switch's default arm, which calls <c>RunDeepLearningOrThrowAsync</c>.
    ///
    /// The task string deliberately is NOT one of the six DL task names the registry throw's
    /// message advertises (image-classification, text-classification, sentence-similarity, ner,
    /// object-detection, question-answering). All six require a native runtime via
    /// <c>RuntimeManager.EnsureRuntimeForTask</c> (called at line ~170 in RunAsync, before the
    /// switch) — when that runtime isn't installed under the user's `.mloop/runtimes` cache (the
    /// case in any clean dev/CI environment), it throws <c>InvalidOperationException</c> first,
    /// pre-empting the throw under test and making the test depend on unrelated machine state.
    /// An unrecognized task name skips the runtime gate (no runtime is registered for it) and
    /// still lands on the switch's default arm, reaching the exact throw this test targets:
    /// <c>DeepLearningRegistry.Current is null → NotSupportedException</c> in
    /// <c>RunDeepLearningOrThrowAsync</c>.
    /// </summary>
    [Fact]
    public async Task RunAsync_UnrecognizedTask_NoRegistry_ThrowsNotSupportedException()
    {
        var ctx = new MLContext(seed: 42);
        var rows = new List<Row>
        {
            new() { F1 = 1f, Label = 0f },
            new() { F1 = 2f, Label = 1f },
            new() { F1 = 3f, Label = 0f },
            new() { F1 = 4f, Label = 1f },
        };
        var dataView = ctx.Data.LoadFromEnumerable(rows);

        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = "synthetic",   // ignored by InMemoryDataProvider
            LabelColumn = "Label",
            Task = "text-generation", // not directory-based, not runtime-gated, not in the switch
            TimeLimitSeconds = 5
        };

        var tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_dlthrow_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tmpDir);
        try
        {
            var runner = new AutoMLRunner(ctx, new InMemoryDataProvider(ctx, dataView), tmpDir);

            var ex = await Assert.ThrowsAsync<NotSupportedException>(
                () => runner.RunAsync(config, cancellationToken: CancellationToken.None));

            Assert.Contains("MLoop.Core.DeepLearning", ex.Message);
        }
        finally
        {
            if (Directory.Exists(tmpDir))
                Directory.Delete(tmpDir, recursive: true);
        }
    }
}
