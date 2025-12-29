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
