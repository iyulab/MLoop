using System.Text.Json;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Tests.Commands;

public class PromoteCommandTests : IDisposable
{
    private readonly string _tempDir;

    public PromoteCommandTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-promote-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    private FilePromotionManager CreateManager()
    {
        var comparer = new FileModelComparer(_tempDir);
        return new FilePromotionManager(_tempDir, comparer);
    }

    #region RecordPromotionAsync

    [Fact]
    public async Task RecordPromotion_CreatesNewFile()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "default");
        Directory.CreateDirectory(modelsDir);
        var manager = CreateManager();

        await manager.RecordPromotionAsync("default", "exp-001", null, "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        Assert.True(File.Exists(historyPath));

        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        Assert.NotNull(records);
        Assert.Single(records);
        Assert.Equal("exp-001", records[0].ExperimentId);
        Assert.Equal("promote", records[0].Action);
    }

    [Fact]
    public async Task RecordPromotion_AppendsToExisting()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "default");
        Directory.CreateDirectory(modelsDir);
        var manager = CreateManager();

        await manager.RecordPromotionAsync("default", "exp-001", null, "promote");
        await manager.RecordPromotionAsync("default", "exp-002", "exp-001", "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        Assert.NotNull(records);
        Assert.Equal(2, records.Count);
        Assert.Equal("exp-002", records[1].ExperimentId);
        Assert.Equal("exp-001", records[1].PreviousExperimentId);
    }

    [Fact]
    public async Task RecordPromotion_RecordsModelName()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "mymodel");
        Directory.CreateDirectory(modelsDir);
        var manager = CreateManager();

        await manager.RecordPromotionAsync("mymodel", "exp-003", "exp-002", "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        Assert.NotNull(records);
        Assert.Equal("mymodel", records[0].ModelName);
    }

    [Fact]
    public async Task RecordPromotion_CreatesDirectoryIfNeeded()
    {
        var manager = CreateManager();

        await manager.RecordPromotionAsync("newmodel", "exp-001", null, "promote");

        var historyPath = Path.Combine(_tempDir, "models", "newmodel", "promotion-history.json");
        Assert.True(File.Exists(historyPath));
    }

    #endregion

    #region BackupProductionAsync

    [Fact]
    public async Task BackupProduction_NoProduction_ReturnsNull()
    {
        var manager = CreateManager();

        var result = await manager.BackupProductionAsync("default");

        Assert.Null(result);
    }

    [Fact]
    public async Task BackupProduction_WithProduction_CreatesBackup()
    {
        var manager = CreateManager();

        // Set up production with registry
        var modelDir = Path.Combine(_tempDir, "models", "default");
        var productionDir = Path.Combine(modelDir, "production");
        Directory.CreateDirectory(productionDir);
        File.WriteAllText(Path.Combine(productionDir, "model.zip"), "model-data");

        // Write registry with production experiment ID
        var registryPath = Path.Combine(modelDir, "model-registry.json");
        var registry = new { production = new { experimentId = "exp-001", modelPath = productionDir, promotedAt = DateTimeOffset.UtcNow } };
        await File.WriteAllTextAsync(registryPath, JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true }));

        var backupPath = await manager.BackupProductionAsync("default");

        Assert.NotNull(backupPath);
        Assert.True(Directory.Exists(backupPath));
        Assert.True(File.Exists(Path.Combine(backupPath, "model.zip")));
    }

    #endregion
}
