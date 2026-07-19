using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// Tests for the compare command functionality.
/// </summary>
[Collection("FileSystem")]
public class CompareCommandTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly IModelRegistry _modelRegistry;

    public CompareCommandTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        // Create temporary test project with .mloop directory
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-compare-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);

        var mloopDir = Path.Combine(_testProjectRoot, ".mloop");
        Directory.CreateDirectory(mloopDir);

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);

        Directory.SetCurrentDirectory(_testProjectRoot);

        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
        _modelRegistry = new ModelRegistry(_fileSystem, _projectDiscovery, _experimentStore);
    }

    public void Dispose()
    {
        try
        {
            Directory.SetCurrentDirectory(_originalDirectory);
        }
        catch
        {
            try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { }
        }

        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public async Task Compare_CanLoadMultipleExperiments_ForSameModel()
    {
        // Arrange - Create multiple experiments for the same model
        var modelName = "churn";

        var exp1 = await CreateTestExperiment(modelName, "exp-001", 0.85, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85,
            ["F1Score"] = 0.82,
            ["AUC"] = 0.88
        });

        var exp2 = await CreateTestExperiment(modelName, "exp-002", 0.90, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90,
            ["F1Score"] = 0.87,
            ["AUC"] = 0.92
        });

        // Act - Load both experiments
        var loaded1 = await _experimentStore.LoadAsync(modelName, "exp-001", CancellationToken.None);
        var loaded2 = await _experimentStore.LoadAsync(modelName, "exp-002", CancellationToken.None);

        // Assert
        Assert.NotNull(loaded1);
        Assert.NotNull(loaded2);
        Assert.Equal("exp-001", loaded1.ExperimentId);
        Assert.Equal("exp-002", loaded2.ExperimentId);
        Assert.NotNull(loaded1.Metrics);
        Assert.NotNull(loaded2.Metrics);
        Assert.Equal(0.85, loaded1.Metrics["Accuracy"]);
        Assert.Equal(0.90, loaded2.Metrics["Accuracy"]);
    }

    [Fact]
    public async Task Compare_MetricsComparison_HigherAccuracyIsBetter()
    {
        // Arrange
        var modelName = "revenue";

        await CreateTestExperiment(modelName, "exp-001", 0.75, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.75,
            ["RSquared"] = 0.70
        });

        await CreateTestExperiment(modelName, "exp-002", 0.88, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.88,
            ["RSquared"] = 0.85
        });

        // Act
        var experiments = (await _experimentStore.ListAsync(modelName, CancellationToken.None))
            .OrderByDescending(e => e.BestMetric ?? 0)
            .ToList();

        // Assert
        Assert.Equal(2, experiments.Count);
        Assert.Equal("exp-002", experiments[0].ExperimentId); // Best should be first
        Assert.Equal(0.88, experiments[0].BestMetric);
    }

    [Fact]
    public async Task Compare_DifferentModels_AreIsolated()
    {
        // Arrange
        await CreateTestExperiment("model-a", "exp-001", 0.80, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });

        await CreateTestExperiment("model-b", "exp-001", 0.90, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });

        // Act
        var modelAExperiments = await _experimentStore.ListAsync("model-a", CancellationToken.None);
        var modelBExperiments = await _experimentStore.ListAsync("model-b", CancellationToken.None);

        // Assert
        Assert.Single(modelAExperiments);
        Assert.Single(modelBExperiments);
        Assert.Equal(0.80, modelAExperiments.First().BestMetric);
        Assert.Equal(0.90, modelBExperiments.First().BestMetric);
    }

    [Fact]
    public async Task Compare_ListAllCompleted_ExcludesFailedByDefault()
    {
        // Arrange
        var modelName = "test-model";

        await CreateTestExperiment(modelName, "exp-001", 0.85, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });

        await CreateTestExperiment(modelName, "exp-002", null, null, "Failed");

        await CreateTestExperiment(modelName, "exp-003", 0.90, new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });

        // Act
        var allExperiments = await _experimentStore.ListAsync(modelName, CancellationToken.None);
        var completedExperiments = allExperiments
            .Where(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            .ToList();

        // Assert
        Assert.Equal(3, allExperiments.Count());
        Assert.Equal(2, completedExperiments.Count);
    }

    // ---- Provided-state comparison (compare --metrics-file) ----

    private static MLoop.CLI.Commands.CompareCommand.ProvidedCandidate Cand(
        string id, params (string key, double val)[] metrics) =>
        new(id, metrics.ToDictionary(m => m.key, m => m.val));

    [Fact]
    public void CompareProvidedMetrics_HigherIsBetter_PicksMax()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("champion", ("r_squared", 0.85)), Cand("challenger", ("r_squared", 0.87)) },
            "r_squared");

        Assert.Equal("challenger", result.Best);
        Assert.Equal("maximize", result.Direction);
        Assert.Equal("challenger", result.Ranking[0].Id); // ranked best-first
    }

    [Fact]
    public void CompareProvidedMetrics_LowerIsBetter_PicksMin()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("champion", ("rmse", 1.30)), Cand("challenger", ("rmse", 1.20)) },
            "rmse");

        Assert.Equal("challenger", result.Best);
        Assert.Equal("minimize", result.Direction);
    }

    [Fact]
    public void CompareProvidedMetrics_NoMetric_SingleShared_Infers()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("a", ("r_squared", 0.80)), Cand("b", ("r_squared", 0.90)) },
            metric: null);

        Assert.Equal("r_squared", result.Metric);
        Assert.Equal("b", result.Best);
    }

    [Fact]
    public void CompareProvidedMetrics_NoMetric_MultipleShared_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
                new[]
                {
                    Cand("a", ("r_squared", 0.80), ("rmse", 1.0)),
                    Cand("b", ("r_squared", 0.90), ("rmse", 0.5))
                },
                metric: null));
    }

    [Fact]
    public void CompareProvidedMetrics_MetricNotFound_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
                new[] { Cand("a", ("r_squared", 0.80)) }, "auc"));
    }

    // ---- Direction provenance & honest exclusion/tie signals (compare provenance gaps) ----

    [Fact]
    public void CompareProvidedMetrics_KnownMetric_DirectionSourceKnown()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("a", ("r_squared", 0.80)), Cand("b", ("r_squared", 0.90)) },
            "r_squared");

        Assert.Equal("known", result.DirectionSource);
    }

    [Fact]
    public void CompareProvidedMetrics_UnknownMetric_DirectionSourceDefault_NotSilentlyKnown()
    {
        // A custom metric the direction authority does not recognize must not masquerade as an
        // authoritative "maximize" — the consumer needs to know it fell through to the default.
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("a", ("weirdmetric", 1.0)), Cand("b", ("weirdmetric", 2.0)) },
            "weirdmetric");

        Assert.Equal("maximize", result.Direction);
        Assert.Equal("default", result.DirectionSource);
    }

    [Fact]
    public void CompareProvidedMetrics_CandidateMissingMetric_ExcludedNotSilentlyDropped()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("champ", ("accuracy", 0.9)), Cand("chall", ("rmse", 1.0)) },
            "accuracy");

        Assert.Equal("champ", result.Best);
        Assert.Single(result.Ranking);
        var excluded = Assert.Single(result.Excluded);
        Assert.Equal("chall", excluded.Id);
        Assert.Equal("metric-missing", excluded.Reason);
    }

    [Fact]
    public void CompareProvidedMetrics_AllScored_ExcludedEmpty()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("a", ("r_squared", 0.80)), Cand("b", ("r_squared", 0.90)) },
            "r_squared");

        Assert.Empty(result.Excluded);
    }

    [Fact]
    public void CompareProvidedMetrics_EqualTopValues_TieSignalled()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("champ", ("r_squared", 0.90)), Cand("chall", ("r_squared", 0.90)) },
            "r_squared");

        Assert.True(result.Tie);
    }

    [Fact]
    public void CompareProvidedMetrics_DistinctTopValues_NoTie()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("champ", ("r_squared", 0.85)), Cand("chall", ("r_squared", 0.90)) },
            "r_squared");

        Assert.False(result.Tie);
    }

    [Fact]
    public void CompareProvidedMetrics_SingleCandidate_NoTie()
    {
        var result = MLoop.CLI.Commands.CompareCommand.CompareProvidedMetrics(
            new[] { Cand("only", ("r_squared", 0.90)) }, "r_squared");

        Assert.False(result.Tie);
    }

    [Theory]
    [InlineData("r_squared")]
    [InlineData("accuracy")]
    [InlineData("macro_accuracy")]
    [InlineData("auc")]
    [InlineData("ndcg")]
    [InlineData("f1_score")]
    [InlineData("rmse")]
    [InlineData("mae")]
    [InlineData("average_distance")]
    public void MetricDirection_IsKnown_RecognizedMetrics_True(string metric)
        => Assert.True(MLoop.Core.Evaluation.MetricDirection.IsKnown(metric));

    [Theory]
    [InlineData("weirdmetric")]
    [InlineData("custom_score")]
    [InlineData("")]
    public void MetricDirection_IsKnown_UnrecognizedMetrics_False(string metric)
        => Assert.False(MLoop.Core.Evaluation.MetricDirection.IsKnown(metric));

    [Fact]
    public void ParseCandidates_MissingId_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MLoop.CLI.Commands.CompareCommand.ParseCandidates(
                "[{\"metrics\":{\"r_squared\":0.8}}]"));
    }

    [Fact]
    public void ParseCandidates_Valid_RoundTrips()
    {
        var cands = MLoop.CLI.Commands.CompareCommand.ParseCandidates(
            "[{\"id\":\"exp-a\",\"metrics\":{\"r_squared\":0.87}}]");

        Assert.Single(cands);
        Assert.Equal("exp-a", cands[0].Id);
        Assert.Equal(0.87, cands[0].Metrics["r_squared"]);
    }

    private async Task<ExperimentData> CreateTestExperiment(
        string modelName,
        string experimentId,
        double? bestMetric,
        Dictionary<string, double>? metrics,
        string status = "Completed")
    {
        var experimentData = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = status,
            Task = "binary-classification",
            Config = new ExperimentConfig
            {
                DataFile = "test-data.csv",
                LabelColumn = "Target",
                TimeLimitSeconds = 60,
                Metric = "Accuracy",
                TestSplit = 0.2
            },
            Metrics = metrics
        };

        await _experimentStore.SaveAsync(modelName, experimentData, CancellationToken.None);
        return experimentData;
    }
}
