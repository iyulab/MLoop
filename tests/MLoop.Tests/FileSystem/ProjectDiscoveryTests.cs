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
        // Use an isolated temp directory without any .mloop ancestor.
        // On CI (Ubuntu), temp dirs won't have .mloop ancestors.
        // On dev machines, an ancestor may contain .mloop; skip gracefully.
        var isolatedDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(isolatedDir);

            if (HasMLoopAncestor(isolatedDir))
                return; // Cannot test: ancestor has .mloop

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

    private static bool HasMLoopAncestor(string path)
    {
        var dir = new DirectoryInfo(path).Parent;
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, ".mloop")))
                return true;
            dir = dir.Parent;
        }
        return false;
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
