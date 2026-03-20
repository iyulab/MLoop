using System.Text.Json;
using MLoop.CLI.Commands;

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

    #region CopyDirectory

    [Fact]
    public void CopyDirectory_CopiesFiles()
    {
        var src = Path.Combine(_tempDir, "src");
        var dst = Path.Combine(_tempDir, "dst");
        Directory.CreateDirectory(src);
        File.WriteAllText(Path.Combine(src, "model.zip"), "data");
        File.WriteAllText(Path.Combine(src, "config.json"), "{}");

        PromoteCommand.CopyDirectory(src, dst);

        Assert.True(File.Exists(Path.Combine(dst, "model.zip")));
        Assert.True(File.Exists(Path.Combine(dst, "config.json")));
    }

    [Fact]
    public void CopyDirectory_CopiesNestedDirectories()
    {
        var src = Path.Combine(_tempDir, "src");
        var nested = Path.Combine(src, "sub", "deep");
        Directory.CreateDirectory(nested);
        File.WriteAllText(Path.Combine(nested, "file.txt"), "content");

        var dst = Path.Combine(_tempDir, "dst");
        PromoteCommand.CopyDirectory(src, dst);

        Assert.True(File.Exists(Path.Combine(dst, "sub", "deep", "file.txt")));
        Assert.Equal("content", File.ReadAllText(Path.Combine(dst, "sub", "deep", "file.txt")));
    }

    [Fact]
    public void CopyDirectory_OverwritesExisting()
    {
        var src = Path.Combine(_tempDir, "src");
        var dst = Path.Combine(_tempDir, "dst");
        Directory.CreateDirectory(src);
        Directory.CreateDirectory(dst);

        File.WriteAllText(Path.Combine(src, "file.txt"), "new");
        File.WriteAllText(Path.Combine(dst, "file.txt"), "old");

        PromoteCommand.CopyDirectory(src, dst);

        Assert.Equal("new", File.ReadAllText(Path.Combine(dst, "file.txt")));
    }

    [Fact]
    public void CopyDirectory_EmptySource_CreatesEmptyDestination()
    {
        var src = Path.Combine(_tempDir, "src");
        var dst = Path.Combine(_tempDir, "dst");
        Directory.CreateDirectory(src);

        PromoteCommand.CopyDirectory(src, dst);

        Assert.True(Directory.Exists(dst));
        Assert.Empty(Directory.GetFiles(dst));
    }

    #endregion

    #region RecordPromotionHistoryAsync

    [Fact]
    public async Task RecordPromotionHistory_CreatesNewFile()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "default");
        Directory.CreateDirectory(modelsDir);

        await PromoteCommand.RecordPromotionHistoryAsync(
            _tempDir, "default", "exp-001", null, "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        Assert.True(File.Exists(historyPath));

        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json);
        Assert.NotNull(records);
        Assert.Single(records);
        Assert.Equal("exp-001", records[0]["experimentId"].GetString());
        Assert.Equal("promote", records[0]["action"].GetString());
    }

    [Fact]
    public async Task RecordPromotionHistory_AppendsToExisting()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "default");
        Directory.CreateDirectory(modelsDir);

        await PromoteCommand.RecordPromotionHistoryAsync(
            _tempDir, "default", "exp-001", null, "promote");
        await PromoteCommand.RecordPromotionHistoryAsync(
            _tempDir, "default", "exp-002", "exp-001", "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json);

        Assert.NotNull(records);
        Assert.Equal(2, records.Count);
        Assert.Equal("exp-002", records[1]["experimentId"].GetString());
        Assert.Equal("exp-001", records[1]["previousExperimentId"].GetString());
    }

    [Fact]
    public async Task RecordPromotionHistory_RecordsPreviousExperiment()
    {
        var modelsDir = Path.Combine(_tempDir, "models", "mymodel");
        Directory.CreateDirectory(modelsDir);

        await PromoteCommand.RecordPromotionHistoryAsync(
            _tempDir, "mymodel", "exp-003", "exp-002", "promote");

        var historyPath = Path.Combine(modelsDir, "promotion-history.json");
        var json = await File.ReadAllTextAsync(historyPath);
        var records = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json);

        Assert.NotNull(records);
        Assert.Equal("mymodel", records[0]["modelName"].GetString());
    }

    [Fact]
    public async Task RecordPromotionHistory_CreatesDirectoryIfNeeded()
    {
        // Do not pre-create the models directory
        await PromoteCommand.RecordPromotionHistoryAsync(
            _tempDir, "newmodel", "exp-001", null, "promote");

        var historyPath = Path.Combine(_tempDir, "models", "newmodel", "promotion-history.json");
        Assert.True(File.Exists(historyPath));
    }

    #endregion
}
