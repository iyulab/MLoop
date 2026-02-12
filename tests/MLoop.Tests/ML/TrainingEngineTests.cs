using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class TrainingEngineTests : IDisposable
{
    private readonly string _tempDir;

    public TrainingEngineTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-te-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public void Constructor_NullFileSystem_Throws()
    {
        var fs = new FileSystemManager();
        var discovery = new ProjectDiscovery(fs);
        var experimentStore = new ExperimentStore(fs, discovery, _tempDir);

        Assert.Throws<ArgumentNullException>(() =>
            new TrainingEngine(null!, experimentStore));
    }

    [Fact]
    public void Constructor_NullExperimentStore_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TrainingEngine(new FileSystemManager(), null!));
    }

    [Fact]
    public void Constructor_ValidArgs_DoesNotThrow()
    {
        var fs = new FileSystemManager();
        var discovery = new ProjectDiscovery(fs);
        var store = new ExperimentStore(fs, discovery, _tempDir);

        var engine = new TrainingEngine(fs, store);

        Assert.NotNull(engine);
    }
}
