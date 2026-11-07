using System.Text.Json;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>
/// Direct coverage of the production-pointer authority. The consistency guard in MLoop.Tests exercises
/// this via a real promote; these pin the two edges in isolation: camelCase on-disk keys map onto the
/// typed record, and a missing file reads as "no production model" (null).
/// </summary>
public class ProductionMetadataTests : IDisposable
{
    private readonly List<string> _tempDirs = new();

    public void Dispose()
    {
        foreach (var d in _tempDirs)
        {
            try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { }
        }
    }

    private string NewProductionDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-prodmeta-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    [Fact]
    public async Task ReadAsync_CamelCaseMetadata_MapsAllFields()
    {
        // The exact camelCase shape the writer (FileSystemManager) produces — the read must not drift
        // from that casing.
        var dir = NewProductionDir();
        var json = """
        {
          "modelName": "default",
          "experimentId": "exp-007",
          "promotedAt": "2026-06-21T11:44:53.8830647Z",
          "metrics": { "accuracy": 1, "log_loss": 0.0024 },
          "task": "image-classification",
          "bestTrainer": "LightGbmMulti",
          "labelColumn": "Label"
        }
        """;
        await File.WriteAllTextAsync(Path.Combine(dir, ExperimentLayout.MetadataFileName), json);

        var meta = await ProductionMetadata.ReadAsync(dir);

        Assert.NotNull(meta);
        Assert.Equal("default", meta!.ModelName);
        Assert.Equal("exp-007", meta.ExperimentId);
        Assert.Equal(new DateTime(2026, 6, 21, 11, 44, 53, DateTimeKind.Utc), meta.PromotedAt!.Value, TimeSpan.FromSeconds(1));
        Assert.Equal("image-classification", meta.Task);
        Assert.Equal("LightGbmMulti", meta.BestTrainer);
        Assert.Equal("Label", meta.LabelColumn);
        Assert.NotNull(meta.Metrics);
        Assert.Equal(1.0, meta.Metrics!["accuracy"]);
        Assert.Equal(0.0024, meta.Metrics["log_loss"], 6);
    }

    [Fact]
    public async Task ReadAsync_MissingFile_ReturnsNull()
    {
        // No production model promoted yet — the "no production" signal every reader already handles.
        var dir = NewProductionDir();
        Assert.Null(await ProductionMetadata.ReadAsync(dir));
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{bad")]
    [InlineData("")]
    [InlineData("null")]
    public async Task ReadAsync_AFileThatCannotBeRead_IsNotAbsence(string content)
    {
        var dir = NewProductionDir();
        var path = Path.Combine(dir, ExperimentLayout.MetadataFileName);
        await File.WriteAllTextAsync(path, content);

        var ex = await Assert.ThrowsAsync<UnreadableStateFileException>(() => ProductionMetadata.ReadAsync(dir));

        Assert.Equal(path, ex.Path);
        Assert.Contains(path, ex.Message, StringComparison.Ordinal);
        Assert.Contains("mloop promote", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("Train and promote a model first", ex.Message, StringComparison.Ordinal);

        // The caller that can go on without the pointer gets the damage as a value.
        var (value, damage) = await ProductionMetadata.TryReadAsync(dir);
        Assert.Null(value);
        Assert.NotNull(damage);
    }

    [Fact]
    public async Task TryReadAsync_MissingFile_IsAbsence_NotDamage()
    {
        var (value, damage) = await ProductionMetadata.TryReadAsync(NewProductionDir());

        Assert.Null(value);
        Assert.Null(damage);
    }
}
