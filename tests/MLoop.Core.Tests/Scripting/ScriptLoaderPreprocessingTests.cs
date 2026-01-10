using MLoop.Core.Scripting;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Scripting;

/// <summary>
/// Tests for ScriptLoader with IPreprocessingScript (Phase 0).
/// Tests for IMLoopHook and IMLoopMetric are in ScriptLoaderTests.cs (Phase 1).
/// </summary>
public class ScriptLoaderPreprocessingTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _cacheDirectory;
    private readonly ScriptLoader _scriptLoader;

    public ScriptLoaderPreprocessingTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), "mloop_test_" + Guid.NewGuid().ToString("N"));
        _cacheDirectory = Path.Combine(_testDirectory, ".cache");
        Directory.CreateDirectory(_testDirectory);

        _scriptLoader = new ScriptLoader(_cacheDirectory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDirectory))
        {
            Directory.Delete(_testDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task LoadScriptAsync_WithValidPreprocessingScript_ReturnsInstance()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "TestScript.cs");
        var scriptContent = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class TestScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var outputPath = Path.Combine(context.OutputDirectory, ""output.csv"");
        return await Task.FromResult(outputPath);
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Single(scripts);
        Assert.NotNull(scripts[0]);
    }

    [Fact]
    public async Task LoadScriptAsync_WithMultipleScripts_ReturnsAllInstances()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "MultipleScripts.cs");
        var scriptContent = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class Script1 : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""script1.csv""));
    }
}

public class Script2 : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""script2.csv""));
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Equal(2, scripts.Count);
    }

    [Fact]
    public async Task LoadScriptAsync_WithNonExistentFile_ReturnsEmptyList()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "NonExistent.cs");

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Empty(scripts);
    }

    [Fact]
    public async Task LoadScriptAsync_WithCompilationError_ReturnsEmptyList()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "InvalidScript.cs");
        var scriptContent = @"
using MLoop.Extensibility.Preprocessing;

public class InvalidScript : IPreprocessingScript
{
    // Missing required method implementation
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Empty(scripts);  // Graceful degradation
    }

    [Fact]
    public async Task LoadScriptAsync_UsesCaching_LoadsFromDllSecondTime()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "CachedScript.cs");
        var scriptContent = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class CachedScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""cached.csv""));
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act - First load (compilation)
        var scripts1 = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Act - Second load (from cache)
        var scripts2 = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Assert
        Assert.Single(scripts1);
        Assert.Single(scripts2);
        Assert.Single(cachedFiles1);  // DLL was created
        Assert.Single(cachedFiles2);  // Same DLL used
        Assert.Equal(cachedFiles1[0], cachedFiles2[0]);
    }

    [Fact]
    public async Task LoadScriptAsync_WithModifiedScript_RecompilesWithNewHash()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "ModifiableScript.cs");
        var scriptContent1 = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class ModifiableScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        // Version 1
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""v1.csv""));
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent1);

        // Act - First load
        var scripts1 = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Modify script
        var scriptContent2 = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class ModifiableScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        // Version 2 - modified
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""v2.csv""));
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent2);

        // Act - Second load with modified script
        var scripts2 = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Assert
        Assert.Single(scripts1);
        Assert.Single(scripts2);
        Assert.Single(cachedFiles1);
        Assert.Equal(2, cachedFiles2.Length);  // Both DLL versions cached (different hashes)
    }

    [Fact]
    public void ClearCache_RemovesAllCachedDlls()
    {
        // Arrange
        Directory.CreateDirectory(_cacheDirectory);
        var dummyFile = Path.Combine(_cacheDirectory, "dummy.dll");
        File.WriteAllText(dummyFile, "dummy content");

        // Act
        _scriptLoader.ClearCache();

        // Assert
        Assert.True(Directory.Exists(_cacheDirectory));  // Directory recreated
        Assert.Empty(Directory.GetFiles(_cacheDirectory));  // No files
    }

    [Fact]
    public async Task LoadScriptAsync_WithAbstractClass_IgnoresAbstractClass()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "AbstractScript.cs");
        var scriptContent = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public abstract class AbstractScript : IPreprocessingScript
{
    public abstract Task<string> ExecuteAsync(PreprocessContext context);
}

public class ConcreteScript : AbstractScript
{
    public override async Task<string> ExecuteAsync(PreprocessContext context)
    {
        return await Task.FromResult(Path.Combine(context.OutputDirectory, ""concrete.csv""));
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Single(scripts);  // Only ConcreteScript, not AbstractScript
    }

    [Fact]
    public async Task LoadScriptAsync_WithClassNotImplementingInterface_ReturnsEmptyList()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "UnrelatedClass.cs");
        var scriptContent = @"
using System.Threading.Tasks;

public class UnrelatedClass
{
    public string SomeMethod() => ""test"";
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Assert
        Assert.Empty(scripts);  // No IPreprocessingScript implementation found
    }
}
