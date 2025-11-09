using MLoop.Core.Scripting;
using MLoop.Extensibility;

namespace MLoop.Core.Tests.Scripting;

public class ScriptLoaderTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly string _cacheDirectory;
    private readonly ScriptLoader _scriptLoader;

    public ScriptLoaderTests()
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
    public async Task LoadScriptAsync_WithValidHook_ReturnsInstance()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "TestHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class TestHook : IMLoopHook
{
    public string Name => ""Test Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        return Task.FromResult(HookResult.Continue());
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        // Assert
        Assert.Single(hooks);
        Assert.Equal("Test Hook", hooks[0].Name);
    }

    [Fact]
    public async Task LoadScriptAsync_WithValidMetric_ReturnsInstance()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "TestMetric.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class TestMetric : IMLoopMetric
{
    public string Name => ""Test Metric"";
    public bool HigherIsBetter => true;

    public Task<double> CalculateAsync(MetricContext context)
    {
        return Task.FromResult(0.95);
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var metrics = await _scriptLoader.LoadScriptAsync<IMLoopMetric>(scriptPath);

        // Assert
        Assert.Single(metrics);
        Assert.Equal("Test Metric", metrics[0].Name);
        Assert.True(metrics[0].HigherIsBetter);
    }

    [Fact]
    public async Task LoadScriptAsync_WithMultipleClasses_ReturnsAllInstances()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "MultipleHooks.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class Hook1 : IMLoopHook
{
    public string Name => ""Hook 1"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}

public class Hook2 : IMLoopHook
{
    public string Name => ""Hook 2"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        // Assert
        Assert.Equal(2, hooks.Count);
        Assert.Contains(hooks, h => h.Name == "Hook 1");
        Assert.Contains(hooks, h => h.Name == "Hook 2");
    }

    [Fact]
    public async Task LoadScriptAsync_WithNonExistentFile_ReturnsEmptyList()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "NonExistent.cs");

        // Act
        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        // Assert
        Assert.Empty(hooks);
    }

    [Fact]
    public async Task LoadScriptAsync_WithInvalidSyntax_ReturnsEmptyList()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "InvalidSyntax.cs");
        var scriptContent = @"
public class InvalidHook : IMLoopHook
{
    // Missing required method implementation
    public string Name => ""Invalid"";
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        // Assert
        Assert.Empty(hooks);  // Graceful degradation
    }

    [Fact]
    public async Task LoadScriptAsync_UsesCaching_LoadsFromDllSecondTime()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "CachedHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class CachedHook : IMLoopHook
{
    public string Name => ""Cached Hook"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act - First load (compilation)
        var hooks1 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Act - Second load (from cache)
        var hooks2 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Assert
        Assert.Single(hooks1);
        Assert.Single(hooks2);
        Assert.Single(cachedFiles1);  // DLL was created
        Assert.Single(cachedFiles2);  // Same DLL used
        Assert.Equal(cachedFiles1[0], cachedFiles2[0]);
    }

    [Fact]
    public async Task LoadScriptAsync_WithModifiedScript_RecompilesWithNewHash()
    {
        // Arrange
        var scriptPath = Path.Combine(_testDirectory, "ModifiableHook.cs");
        var scriptContent1 = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class ModifiableHook : IMLoopHook
{
    public string Name => ""Version 1"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent1);

        // Act - First load
        var hooks1 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Modify script
        var scriptContent2 = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public class ModifiableHook : IMLoopHook
{
    public string Name => ""Version 2"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent2);

        // Act - Second load with modified script
        var hooks2 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        // Assert
        Assert.Single(hooks1);
        Assert.Equal("Version 1", hooks1[0].Name);
        Assert.Single(hooks2);
        Assert.Equal("Version 2", hooks2[0].Name);
        Assert.Single(cachedFiles1);
        Assert.Equal(2, cachedFiles2.Length);  // Both DLL versions cached
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
        var scriptPath = Path.Combine(_testDirectory, "AbstractHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility;

public abstract class AbstractHook : IMLoopHook
{
    public abstract string Name { get; }
    public abstract Task<HookResult> ExecuteAsync(HookContext context);
}

public class ConcreteHook : AbstractHook
{
    public override string Name => ""Concrete Hook"";
    public override Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        // Act
        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        // Assert
        Assert.Single(hooks);  // Only ConcreteHook, not AbstractHook
        Assert.Equal("Concrete Hook", hooks[0].Name);
    }
}
