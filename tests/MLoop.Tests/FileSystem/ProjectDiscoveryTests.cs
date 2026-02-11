using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.FileSystem;

public class ProjectDiscoveryTests : IDisposable
{
    private readonly FileSystemManager _fs = new();
    private readonly string _tempDir;

    public ProjectDiscoveryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-pd-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public void Constructor_NullFileSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ProjectDiscovery(null!));
    }

    [Fact]
    public void FindRoot_WithMLoopDir_ReturnsProjectRoot()
    {
        var mloopDir = Path.Combine(_tempDir, ".mloop");
        Directory.CreateDirectory(mloopDir);

        var discovery = new ProjectDiscovery(_fs);
        var root = discovery.FindRoot(_tempDir);

        Assert.Equal(_tempDir, root);
    }

    [Fact]
    public void FindRoot_FromSubdirectory_FindsRoot()
    {
        var mloopDir = Path.Combine(_tempDir, ".mloop");
        Directory.CreateDirectory(mloopDir);
        var subDir = Path.Combine(_tempDir, "src", "components");
        Directory.CreateDirectory(subDir);

        var discovery = new ProjectDiscovery(_fs);
        var root = discovery.FindRoot(subDir);

        Assert.Equal(_tempDir, root);
    }

    [Fact]
    public void FindRoot_NoMLoopDir_Throws()
    {
        // Use a path directly under the drive root to avoid ancestor .mloop directories
        var driveRoot = Path.GetPathRoot(Path.GetTempPath())!;
        var isolatedDir = Path.Combine(driveRoot, $"mloop-test-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(isolatedDir);
            var discovery = new ProjectDiscovery(_fs);

            var ex = Assert.Throws<InvalidOperationException>(() =>
                discovery.FindRoot(isolatedDir));

            Assert.Contains("Not inside a MLoop project", ex.Message);
        }
        finally
        {
            if (Directory.Exists(isolatedDir))
                Directory.Delete(isolatedDir, true);
        }
    }

    [Fact]
    public void IsProjectRoot_WithMLoopDir_ReturnsTrue()
    {
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
        var discovery = new ProjectDiscovery(_fs);

        Assert.True(discovery.IsProjectRoot(_tempDir));
    }

    [Fact]
    public void IsProjectRoot_WithoutMLoopDir_ReturnsFalse()
    {
        var discovery = new ProjectDiscovery(_fs);

        Assert.False(discovery.IsProjectRoot(_tempDir));
    }

    [Fact]
    public void GetMLoopDirectory_ReturnsCombinedPath()
    {
        var discovery = new ProjectDiscovery(_fs);

        var result = discovery.GetMLoopDirectory(_tempDir);

        Assert.Equal(Path.Combine(_tempDir, ".mloop"), result);
    }
}
