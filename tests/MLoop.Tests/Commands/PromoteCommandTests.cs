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
        return new FilePromotionManager(_tempDir);
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

        // Set up production the way ModelRegistry.PromoteAsync does: a production/ directory
        // carrying model.zip and its own metadata.json (the authoritative experiment-id source).
        // The backup must trigger off the directory's presence — driving it from a separate
        // registry file used to silently skip the backup, risking loss of the previous
        // production model on the next promote (BUG-48).
        var modelDir = Path.Combine(_tempDir, "models", "default");
        var productionDir = Path.Combine(modelDir, "production");
        Directory.CreateDirectory(productionDir);
        File.WriteAllText(Path.Combine(productionDir, "model.zip"), "model-data");

        var metadata = new { experimentId = "exp-001", promotedAt = DateTimeOffset.UtcNow };
        await File.WriteAllTextAsync(
            Path.Combine(productionDir, "metadata.json"),
            JsonSerializer.Serialize(metadata, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));

        var backupPath = await manager.BackupProductionAsync("default");

        Assert.NotNull(backupPath);
        Assert.True(Directory.Exists(backupPath));
        Assert.True(File.Exists(Path.Combine(backupPath, "model.zip")));
        // Labelled with the production experiment id read from metadata.json.
        Assert.StartsWith("exp-001-", Path.GetFileName(backupPath));
    }

    #endregion
}
