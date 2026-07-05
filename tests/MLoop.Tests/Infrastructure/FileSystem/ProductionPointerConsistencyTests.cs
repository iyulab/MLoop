using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// registry.json single-sourcing guard: the production pointer ("what experiment is in production") now
/// has ONE authority — <c>production/metadata.json</c> (via <see cref="ProductionMetadata"/>). Previously
/// it was duplicated in <c>registry.json</c>, and three readers (<see cref="ModelRegistry"/>,
/// <see cref="ModelNameResolver"/>, Ops <c>FilePromotionManager</c>) split across the two artifacts —
/// two of the same fact that could silently disagree (the BUG-48 cross-artifact class). This pins that
/// after a promote the CLI readers agree on the pointer and that the removed <c>registry.json</c> write
/// path stays removed. (FilePromotionManager's metadata.json read is pinned in MLoop.Ops.Tests.)
/// </summary>
[Collection("FileSystem")]
public class ProductionPointerConsistencyTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly ModelRegistry _modelRegistry;
    private readonly ModelNameResolver _modelNameResolver;
    private const string ModelName = "default";

    public ProductionPointerConsistencyTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-prodptr-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);
        Directory.SetCurrentDirectory(_testProjectRoot);

        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
        _modelRegistry = new ModelRegistry(_fileSystem, _projectDiscovery, _experimentStore);
        _modelNameResolver = new ModelNameResolver(
            _fileSystem, _projectDiscovery, new ConfigLoader(_fileSystem, _projectDiscovery));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }
        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public async Task AfterPromote_AllReadersAgreeOnPointer_AndNoRegistryJsonWritten()
    {
        var experimentId = await CreateDummyExperimentAsync("exp-001",
            new Dictionary<string, double> { ["r_squared"] = 0.85 });

        await _modelRegistry.PromoteAsync(ModelName, experimentId, CancellationToken.None);

        // Reader 1: ModelRegistry.GetProductionAsync (mloop list / API / cache / quality gate).
        var fromRegistry = await _modelRegistry.GetProductionAsync(ModelName, CancellationToken.None);
        Assert.NotNull(fromRegistry);
        Assert.Equal(experimentId, fromRegistry!.ExperimentId);

        // Reader 2: ModelNameResolver.ListAsync (mloop list model summaries).
        var summaries = await _modelNameResolver.ListAsync(CancellationToken.None);
        var summary = summaries.Single(s => s.Name == ModelName);
        Assert.Equal(experimentId, summary.ProductionExperiment);

        // Reader 3 (authority directly): production/metadata.json is the single source both read.
        var authority = await ProductionMetadata.ReadAsync(
            _modelRegistry.GetProductionPath(ModelName), CancellationToken.None);
        Assert.Equal(experimentId, authority!.ExperimentId);

        // The two CLI readers cannot disagree, because they read one artifact.
        Assert.Equal(fromRegistry.ExperimentId, summary.ProductionExperiment);

        // The removed registry.json write path stays removed — promote must not recreate it.
        var registryPath = Path.Combine(_testProjectRoot, "models", ModelName, "registry.json");
        Assert.False(File.Exists(registryPath),
            "registry.json should no longer be written — production/metadata.json is the sole authority.");
    }

    private async Task<string> CreateDummyExperimentAsync(string experimentId, Dictionary<string, double> metrics)
    {
        var experimentPath = _experimentStore.GetExperimentPath(ModelName, experimentId);
        Directory.CreateDirectory(experimentPath);
        await File.WriteAllTextAsync(
            Path.Combine(experimentPath, ExperimentLayout.ModelFileName), "dummy model content");

        var experimentData = new ExperimentData
        {
            ModelName = ModelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 60,
                Metric = metrics.Keys.First(),
                TestSplit = 0.2
            },
            Metrics = metrics
        };
        await _experimentStore.SaveAsync(ModelName, experimentData, CancellationToken.None);
        return experimentId;
    }
}
