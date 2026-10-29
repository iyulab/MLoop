using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

public sealed class AtomicFileTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-atomicfile-" + Guid.NewGuid().ToString("N"));

    public AtomicFileTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Fact]
    public void WriteAllText_ReplacesTheContent_AndLeavesOnlyTheFile()
    {
        var path = Path.Combine(_directory, "decision-threshold.json");
        AtomicFile.WriteAllText(path, "{\"a\":1}");
        AtomicFile.WriteAllText(path, "{\"a\":2}");

        Assert.Equal("{\"a\":2}", File.ReadAllText(path));
        Assert.Equal(["decision-threshold.json"], Directory.GetFileSystemEntries(_directory).Select(Path.GetFileName));
    }

    [Fact]
    public void AFailedWrite_KeepsTheOldContent_AndLeavesNoTemporaryFile()
    {
        var path = Path.Combine(_directory, "calibration.json");
        File.WriteAllText(path, "old");

        Assert.Throws<InvalidOperationException>(() => AtomicFile.Replace(path, temporary =>
        {
            File.WriteAllText(temporary, "half");
            throw new InvalidOperationException("the writer failed part-way");
        }));

        Assert.Equal("old", File.ReadAllText(path));
        Assert.Equal(["calibration.json"], Directory.GetFileSystemEntries(_directory).Select(Path.GetFileName));
    }

    [Fact]
    public async Task ReplaceAsync_CreatesTheFile_WhenThereWasNone()
    {
        var path = Path.Combine(_directory, "new.txt");

        await AtomicFile.ReplaceAsync(path, temporary => File.WriteAllTextAsync(temporary, "fresh"));

        Assert.Equal("fresh", File.ReadAllText(path));
    }
}
