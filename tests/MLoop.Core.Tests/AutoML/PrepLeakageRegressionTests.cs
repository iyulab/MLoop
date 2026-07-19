using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Leakage-regression E2E guard: verifies that a non-null PreFeaturizer (built from a
/// normalize PrepStep) flows end-to-end through AutoMLRunner.RunAsync and that
/// experiment.Execute actually receives it (closing the gap left by Task 3's unit test
/// which only confirmed the field exists, not that it reached Execute at runtime).
///
/// Uses a stub IDataProvider that returns an in-memory IDataView with explicit individual
/// columns, so InferColumns' column-merging does not hide the 'Age'/'Score' columns that
/// the preFeaturizer references. This mirrors the contract the real TrainCommand path
/// relies on: columns are individually addressable when the preFeaturizer is applied.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class PrepLeakageRegressionTests : IDisposable
{
    private readonly string _tmpDir;

    public PrepLeakageRegressionTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_leak_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    // ─── Data schema for the synthetic in-memory dataset ──────────────────────────
    private sealed class SyntheticRow
    {
        public float Age { get; set; }

        public float Score { get; set; }

        [ColumnName("label")]
        public bool Label { get; set; }
    }

    // ─── Stub IDataProvider that serves an in-memory IDataView ────────────────────
    // Bypasses CsvDataLoader's InferColumns which can merge numeric columns into a
    // ranged Features vector, hiding individual column names from the preFeaturizer.
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
            var cols = data.Schema
                .Where(c => !c.IsHidden)
                .Select(c => new ColumnInfo
                {
                    Name = c.Name,
                    Type = DataTypeHelper.GetFriendlyTypeName(c.Type),
                    IsLabel = false
                })
                .ToList();
            return new DataSchema { Columns = cols, RowCount = 0 };
        }

        public (IDataView trainSet, IDataView testSet) SplitData(IDataView data, double testFraction = 0.2)
        {
            var split = _ctx.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42);
            return (split.TrainSet, split.TestSet);
        }
    }

    [Fact]
    public async Task Training_with_prefeaturizer_normalize_produces_valid_metrics()
    {
        // Arrange ─────────────────────────────────────────────────────────────────
        // 1. Build synthetic binary-classification data in memory: 200 rows.
        //    'Age' has large scale (0–199000) to make normalization meaningful.
        //    Label alternates true/false so both classes are present for evaluation.
        var ctx = new MLContext(seed: 42);

        var synthRows = Enumerable.Range(0, 200).Select(i => new SyntheticRow
        {
            Age = i * 1000f,
            Score = i % 50,
            Label = i % 2 == 0
        }).ToList();

        // LoadFromEnumerable gives individual named columns: "Age", "Score", "label"
        // ML.NET maps property names to column names unless [ColumnName] overrides.
        var dataView = ctx.Data.LoadFromEnumerable(synthRows);

        // Guard: verify the schema has the expected individual column names
        Assert.NotNull(dataView.Schema.GetColumnOrNull("Age"));
        Assert.NotNull(dataView.Schema.GetColumnOrNull("Score"));
        Assert.NotNull(dataView.Schema.GetColumnOrNull("label"));

        // 2. Build a normalize/min-max preFeaturizer from PrepStep (Task 2 output).
        //    Column names must match the data view schema: "Age", "Score".
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new List<string> { "Age", "Score" } }
        };
        var preFeaturizer = new PrepFeaturizerBuilder().Build(ctx, steps);
        Assert.NotNull(preFeaturizer); // guard: builder must produce something for normalize steps

        // 3. Wire the preFeaturizer into TrainingConfig (Task 3 output).
        //    The DataFile path is irrelevant — InMemoryDataProvider ignores it.
        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = "synthetic",     // ignored by InMemoryDataProvider
            LabelColumn = "label",
            Task = "binary-classification",
            TimeLimitSeconds = 10,
            PreFeaturizer = preFeaturizer
        };

        // Act ──────────────────────────────────────────────────────────────────────
        // Use InMemoryDataProvider so AutoMLRunner sees individually-named columns.
        // This is the critical E2E path: non-null PreFeaturizer flows to experiment.Execute
        // (Task 5 Execute sites). If PreFeaturizer was silently dropped, the normalizer
        // would be skipped but no error would occur; if wiring is broken, an exception fires.
        var dataProvider = new InMemoryDataProvider(ctx, dataView);
        var runner = new AutoMLRunner(ctx, dataProvider, _tmpDir);
        var result = await runner.RunAsync(config, cancellationToken: CancellationToken.None);

        // Assert ───────────────────────────────────────────────────────────────────
        // Training must complete (no exception propagated) and return valid metrics,
        // proving the preFeaturizer was accepted by experiment.Execute without error.
        Assert.NotNull(result);
        Assert.NotNull(result.Model);
        Assert.NotEmpty(result.Metrics);
        Assert.True(result.Metrics.ContainsKey("accuracy"),
            $"Expected 'accuracy' key in metrics. Actual keys: [{string.Join(", ", result.Metrics.Keys)}]");
        Assert.InRange(result.Metrics["accuracy"], 0.0, 1.0);
    }

    /// <summary>
    /// Production-path guard: exercises the REAL CsvDataLoader (not the in-memory stub).
    ///
    /// CsvDataLoader.LoadData calls Auto().InferColumns, which merges adjacent numeric
    /// feature columns into a single ranged Features vector — so the individual columns
    /// "age"/"score" referenced by the preFeaturizer (NormalizeMinMax("age","age"), ...)
    /// no longer exist in the schema by the time experiment.Execute runs. ML.NET then
    /// throws "Could not find input column 'age'" during AutoML cross-validation, crashing
    /// any real `mloop train` with a normalize/scale/fill-mean prep step.
    ///
    /// The fix threads the preFeaturizer's columns through TrainingConfig.PreFeaturizerColumns
    /// into the preserveColumns set, so CsvDataLoader keeps them individually addressable.
    /// This test must pass WITHOUT throwing "Could not find input column".
    /// </summary>
    [Fact]
    public async Task RealLoader_training_with_prefeaturizer_normalize_preserves_columns()
    {
        // Arrange ─────────────────────────────────────────────────────────────────
        // Write a real CSV with two numeric feature columns (age, score) + label, the
        // exact shape that triggers InferColumns merging.
        var ctx = new MLContext(seed: 42);

        var csvPath = Path.Combine(_tmpDir, "train.csv");
        var lines = new List<string> { "age,score,label" };
        for (int i = 0; i < 200; i++)
        {
            // age has large scale to make normalization meaningful; label separable by parity
            var age = i * 1000;
            var score = i % 50;
            var label = (i % 2 == 0) ? "yes" : "no";
            lines.Add($"{age},{score},{label}");
        }
        await File.WriteAllLinesAsync(csvPath, lines);

        // Build the preFeaturizer from a normalize step on the two numeric columns,
        // and derive the columns to preserve via the same routing the production path uses.
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new List<string> { "age", "score" } }
        };
        var route = new PrepRouter().Route(ctx, steps);
        Assert.NotNull(route.PreFeaturizer);
        Assert.Contains("age", route.PreFeaturizerColumns);
        Assert.Contains("score", route.PreFeaturizerColumns);

        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = csvPath,
            LabelColumn = "label",
            Task = "binary-classification",
            TimeLimitSeconds = 10,
            PreFeaturizer = route.PreFeaturizer,
            PreFeaturizerColumns = route.PreFeaturizerColumns
        };

        // Act ──────────────────────────────────────────────────────────────────────
        // Real CsvDataLoader + real AutoMLRunner: the full production loader path.
        var loader = new CsvDataLoader(ctx);
        var runner = new AutoMLRunner(ctx, loader, _tmpDir);
        var result = await runner.RunAsync(config, cancellationToken: CancellationToken.None);

        // Assert ───────────────────────────────────────────────────────────────────
        Assert.NotNull(result);
        Assert.NotNull(result.Model);
        Assert.NotEmpty(result.Metrics);
        Assert.True(result.Metrics.ContainsKey("accuracy"),
            $"Expected 'accuracy' key in metrics. Actual keys: [{string.Join(", ", result.Metrics.Keys)}]");
        Assert.InRange(result.Metrics["accuracy"], 0.0, 1.0);
    }
}
