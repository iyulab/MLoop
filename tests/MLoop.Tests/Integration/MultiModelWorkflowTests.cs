using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Integration;

/// <summary>
/// Integration tests for multi-model workflow scenarios.
/// Tests the complete lifecycle of managing multiple models within a single project.
/// </summary>
[Collection("FileSystem")]
public class MultiModelWorkflowTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly IModelRegistry _modelRegistry;

    public MultiModelWorkflowTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        // Create temporary test project with .mloop directory
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-multimodel-test-" + Guid.NewGuid());
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

    #region Multi-Model Experiment Management

    [Fact]
    public async Task MultiModel_IndependentExperimentIds_EachModelStartsFromExp001()
    {
        // Arrange & Act - Generate IDs for three different models
        var churnId1 = await _experimentStore.GenerateIdAsync("churn", CancellationToken.None);
        var revenueId1 = await _experimentStore.GenerateIdAsync("revenue", CancellationToken.None);
        var defaultId1 = await _experimentStore.GenerateIdAsync("default", CancellationToken.None);

        var churnId2 = await _experimentStore.GenerateIdAsync("churn", CancellationToken.None);
        var revenueId2 = await _experimentStore.GenerateIdAsync("revenue", CancellationToken.None);

        // Assert - Each model has independent ID sequence
        Assert.Equal("exp-001", churnId1);
        Assert.Equal("exp-001", revenueId1);
        Assert.Equal("exp-001", defaultId1);
        Assert.Equal("exp-002", churnId2);
        Assert.Equal("exp-002", revenueId2);
    }

    [Fact]
    public async Task MultiModel_ExperimentIsolation_ModelsDoNotShareExperiments()
    {
        // Arrange - Create experiments for different models
        var churnExp = await CreateExperimentAsync("churn", "exp-001", 0.85);
        var revenueExp = await CreateExperimentAsync("revenue", "exp-001", 0.90);

        // Act - List experiments for each model
        var churnExperiments = (await _experimentStore.ListAsync("churn", CancellationToken.None)).ToList();
        var revenueExperiments = (await _experimentStore.ListAsync("revenue", CancellationToken.None)).ToList();
        var defaultExperiments = (await _experimentStore.ListAsync("default", CancellationToken.None)).ToList();

        // Assert - Each model only sees its own experiments
        Assert.Single(churnExperiments);
        Assert.Single(revenueExperiments);
        Assert.Empty(defaultExperiments);

        Assert.Equal("churn", churnExperiments[0].ModelName);
        Assert.Equal("revenue", revenueExperiments[0].ModelName);
    }

    [Fact]
    public async Task MultiModel_ListAllExperiments_ReturnsAcrossAllModels()
    {
        // Arrange - Create experiments for multiple models
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("churn", "exp-002", 0.88);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);
        await CreateExperimentAsync("default", "exp-001", 0.75);

        // Act - List all experiments (null = all models)
        var allExperiments = (await _experimentStore.ListAsync(null, CancellationToken.None)).ToList();

        // Assert
        Assert.Equal(4, allExperiments.Count);
        Assert.Equal(2, allExperiments.Count(e => e.ModelName == "churn"));
        Assert.Single(allExperiments.Where(e => e.ModelName == "revenue"));
        Assert.Single(allExperiments.Where(e => e.ModelName == "default"));
    }

    #endregion

    #region Multi-Model Production Management

    [Fact]
    public async Task MultiModel_IndependentProduction_EachModelHasOwnProductionModel()
    {
        // Arrange - Create experiments and promote for each model
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);
        await CreateExperimentAsync("default", "exp-001", 0.75);

        // Act - Promote each model's experiment to production
        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("revenue", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("default", "exp-001", CancellationToken.None);

        // Assert - Each model has its own production model
        var churnProd = await _modelRegistry.GetProductionAsync("churn", CancellationToken.None);
        var revenueProd = await _modelRegistry.GetProductionAsync("revenue", CancellationToken.None);
        var defaultProd = await _modelRegistry.GetProductionAsync("default", CancellationToken.None);

        Assert.NotNull(churnProd);
        Assert.NotNull(revenueProd);
        Assert.NotNull(defaultProd);

        Assert.Equal("churn", churnProd.ModelName);
        Assert.Equal("revenue", revenueProd.ModelName);
        Assert.Equal("default", defaultProd.ModelName);
    }

    [Fact]
    public async Task MultiModel_PromotionIsolation_PromotingOneModelDoesNotAffectOthers()
    {
        // Arrange - Create and promote initial versions
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("churn", "exp-002", 0.90);
        await CreateExperimentAsync("revenue", "exp-001", 0.80);

        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("revenue", "exp-001", CancellationToken.None);

        // Act - Promote new version for churn only
        await _modelRegistry.PromoteAsync("churn", "exp-002", CancellationToken.None);

        // Assert - Churn updated, revenue unchanged
        var churnProd = await _modelRegistry.GetProductionAsync("churn", CancellationToken.None);
        var revenueProd = await _modelRegistry.GetProductionAsync("revenue", CancellationToken.None);

        Assert.Equal("exp-002", churnProd!.ExperimentId);
        Assert.Equal("exp-001", revenueProd!.ExperimentId);
    }

    [Fact]
    public async Task MultiModel_ListProductionModels_ReturnsAllPromotedModels()
    {
        // Arrange
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);
        await CreateExperimentAsync("ltv", "exp-001", 0.75);

        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("revenue", "exp-001", CancellationToken.None);
        // ltv not promoted

        // Act
        var allProduction = (await _modelRegistry.ListAsync(null, CancellationToken.None)).ToList();

        // Assert
        Assert.Equal(2, allProduction.Count);
        Assert.Contains(allProduction, m => m.ModelName == "churn");
        Assert.Contains(allProduction, m => m.ModelName == "revenue");
        Assert.DoesNotContain(allProduction, m => m.ModelName == "ltv");
    }

    [Fact]
    public async Task MultiModel_FilterProductionByName_ReturnsOnlySpecifiedModel()
    {
        // Arrange
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);
        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("revenue", "exp-001", CancellationToken.None);

        // Act
        var churnOnly = (await _modelRegistry.ListAsync("churn", CancellationToken.None)).ToList();

        // Assert
        Assert.Single(churnOnly);
        Assert.Equal("churn", churnOnly[0].ModelName);
    }

    #endregion

    #region Auto-Promotion with Multiple Models

    [Fact]
    public async Task MultiModel_AutoPromote_OnlyAffectsSpecifiedModel()
    {
        // Arrange - Create experiments for two models
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);

        // Promote initial churn model
        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);

        // Create better churn model
        await CreateExperimentAsync("churn", "exp-002", 0.95);

        // Act - Auto-promote for churn only
        var promoted = await _modelRegistry.AutoPromoteAsync("churn", "exp-002", "accuracy", CancellationToken.None);

        // Assert
        Assert.True(promoted);

        var churnProd = await _modelRegistry.GetProductionAsync("churn", CancellationToken.None);
        var revenueProd = await _modelRegistry.GetProductionAsync("revenue", CancellationToken.None);

        Assert.Equal("exp-002", churnProd!.ExperimentId);
        Assert.Null(revenueProd); // Revenue was never promoted
    }

    [Fact]
    public async Task MultiModel_ShouldPromote_ComparesWithinSameModel()
    {
        // Arrange - Set up production models with different metrics
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.60);

        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);
        await _modelRegistry.PromoteAsync("revenue", "exp-001", CancellationToken.None);

        // Create new experiments
        await CreateExperimentAsync("churn", "exp-002", 0.80);   // Worse than churn production
        await CreateExperimentAsync("revenue", "exp-002", 0.65); // Better than revenue production

        // Act
        var shouldPromoteChurn = await _modelRegistry.ShouldPromoteAsync("churn", "exp-002", "accuracy", CancellationToken.None);
        var shouldPromoteRevenue = await _modelRegistry.ShouldPromoteAsync("revenue", "exp-002", "accuracy", CancellationToken.None);

        // Assert - Each model compared against its own production, not others
        Assert.False(shouldPromoteChurn);   // 0.80 < 0.85
        Assert.True(shouldPromoteRevenue);  // 0.65 > 0.60
    }

    #endregion

    #region Directory Structure Verification

    [Fact]
    public async Task MultiModel_DirectoryStructure_CreatesModelNamespacedPaths()
    {
        // Arrange & Act
        await CreateExperimentAsync("churn", "exp-001", 0.85);
        await CreateExperimentAsync("revenue", "exp-001", 0.90);

        await _modelRegistry.PromoteAsync("churn", "exp-001", CancellationToken.None);

        // Assert - Verify directory structure
        var churnStagingPath = _experimentStore.GetExperimentPath("churn", "exp-001");
        var revenueStagingPath = _experimentStore.GetExperimentPath("revenue", "exp-001");
        var churnProductionPath = _modelRegistry.GetProductionPath("churn");
        var revenueProductionPath = _modelRegistry.GetProductionPath("revenue");

        Assert.Contains("churn", churnStagingPath);
        Assert.Contains("revenue", revenueStagingPath);
        Assert.Contains("staging", churnStagingPath);

        Assert.Contains("churn", churnProductionPath);
        Assert.Contains("production", churnProductionPath);

        Assert.True(Directory.Exists(churnStagingPath));
        Assert.True(Directory.Exists(revenueStagingPath));
        Assert.True(Directory.Exists(churnProductionPath));
        Assert.False(Directory.Exists(revenueProductionPath)); // Not promoted yet
    }

    #endregion

    #region Helper Methods

    private async Task<string> CreateExperimentAsync(string modelName, string experimentId, double metric)
    {
        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);
        Directory.CreateDirectory(experimentPath);

        // Create dummy model file
        var modelPath = Path.Combine(experimentPath, "model.zip");
        await File.WriteAllTextAsync(modelPath, "dummy model content");

        var experimentData = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = modelName == "revenue" ? "regression" : "classification",
            Config = new ExperimentConfig
            {
                DataFile = "data.csv",
                LabelColumn = modelName,
                TimeLimitSeconds = 60,
                Metric = "accuracy",
                TestSplit = 0.2
            },
            Metrics = new Dictionary<string, double>
            {
                ["accuracy"] = metric
            }
        };

        await _experimentStore.SaveAsync(modelName, experimentData, CancellationToken.None);

        return experimentId;
    }

    #endregion
}
