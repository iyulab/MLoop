using System.Text.Json;
using FluentAssertions;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class FilePromotionManagerTests : IDisposable
{
    private readonly string _testDir;
    private readonly FilePromotionManager _manager;

    public FilePromotionManagerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _manager = new FilePromotionManager(_testDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDir))
            {
                Directory.Delete(_testDir, recursive: true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    #region BackupProductionAsync Tests

    [Fact]
    public async Task BackupProductionAsync_ReturnsNull_WhenNoProductionDirectory()
    {
        var backupPath = await _manager.BackupProductionAsync("test-model");

        backupPath.Should().BeNull();
    }

    [Fact]
    public async Task BackupProductionAsync_CreatesBackup_WhenProductionExists()
    {
        // Production is a real directory carrying its own metadata.json (the artifact
        // ModelRegistry.PromoteAsync writes), NOT the obsolete model-registry.json. The
        // backup must trigger off the directory's presence — reading a separate registry
        // file used to make this silently skip the backup, risking loss of the previous
        // production model on the next promote (BUG-48).
        CreateProduction("test-model", experimentId: "exp-001");

        var backupPath = await _manager.BackupProductionAsync("test-model");

        backupPath.Should().NotBeNull();
        Directory.Exists(backupPath!).Should().BeTrue();
        File.Exists(Path.Combine(backupPath!, "model.zip")).Should().BeTrue();
        // Labelled with the production experiment id read from metadata.json.
        Path.GetFileName(backupPath!).Should().StartWith("exp-001-");
    }

    [Fact]
    public async Task BackupProductionAsync_StillBacksUp_WhenMetadataMissing()
    {
        // Even without metadata to name the backup, an existing production directory must
        // be preserved (fallback label) rather than silently skipped.
        var productionPath = Path.Combine(_testDir, "models", "test-model", "production");
        Directory.CreateDirectory(productionPath);
        await File.WriteAllTextAsync(Path.Combine(productionPath, "model.zip"), "dummy-model");

        var backupPath = await _manager.BackupProductionAsync("test-model");

        backupPath.Should().NotBeNull();
        Directory.Exists(backupPath!).Should().BeTrue();
        File.Exists(Path.Combine(backupPath!, "model.zip")).Should().BeTrue();
        Path.GetFileName(backupPath!).Should().StartWith("production-");
    }

    #endregion

    #region RecordPromotionAsync Tests

    [Fact]
    public async Task RecordPromotionAsync_WritesHistoryRecord()
    {
        await _manager.RecordPromotionAsync("test-model", "exp-002", "exp-001", "promote");

        var historyPath = Path.Combine(_testDir, "models", "test-model", "promotion-history.json");
        File.Exists(historyPath).Should().BeTrue();

        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(
            await File.ReadAllTextAsync(historyPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        records.Should().ContainSingle();
        records![0].ExperimentId.Should().Be("exp-002");
        records[0].PreviousExperimentId.Should().Be("exp-001");
        records[0].Action.Should().Be("promote");
    }

    [Fact]
    public async Task RecordPromotionAsync_AppendsToExistingHistory()
    {
        await _manager.RecordPromotionAsync("test-model", "exp-001", null, "promote");
        await _manager.RecordPromotionAsync("test-model", "exp-002", "exp-001", "promote");

        var historyPath = Path.Combine(_testDir, "models", "test-model", "promotion-history.json");
        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(
            await File.ReadAllTextAsync(historyPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        records.Should().HaveCount(2);
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Creates a production model directory the way ModelRegistry.PromoteAsync does:
    /// models/{name}/production/ with model.zip and a camelCase metadata.json.
    /// </summary>
    private void CreateProduction(string modelName, string experimentId)
    {
        var productionPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant(), "production");
        Directory.CreateDirectory(productionPath);
        File.WriteAllText(Path.Combine(productionPath, "model.zip"), "dummy-model");

        var metadata = JsonSerializer.Serialize(
            new { experimentId, promotedAt = DateTimeOffset.UtcNow },
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        File.WriteAllText(Path.Combine(productionPath, "metadata.json"), metadata);
    }

    #endregion
}
