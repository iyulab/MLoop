using MLoop.Core.Scripting;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Metrics;

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
        var scriptPath = Path.Combine(_testDirectory, "TestHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestHook : IMLoopHook
{
    public string Name => ""Test Hook"";

    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        return Task.FromResult(HookResult.Continue());
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        Assert.Single(hooks);
        Assert.Equal("Test Hook", hooks[0].Name);
    }

    [Fact]
    public async Task LoadScriptAsync_WithValidMetric_ReturnsInstance()
    {
        var scriptPath = Path.Combine(_testDirectory, "TestMetric.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class TestMetric : IMLoopMetric
{
    public string Name => ""Test Metric"";
    public string Description => ""A test metric"";

    public Task<MetricResult> CalculateAsync(MetricContext context)
    {
        return Task.FromResult(new MetricResult { Name = Name, Value = 0.95 });
    }
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        var metrics = await _scriptLoader.LoadScriptAsync<IMLoopMetric>(scriptPath);

        Assert.Single(metrics);
        Assert.Equal("Test Metric", metrics[0].Name);
        Assert.Equal("A test metric", metrics[0].Description);
    }

    [Fact]
    public async Task LoadScriptAsync_WithMultipleClasses_ReturnsAllInstances()
    {
        var scriptPath = Path.Combine(_testDirectory, "MultipleHooks.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

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

        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        Assert.Equal(2, hooks.Count);
        Assert.Contains(hooks, h => h.Name == "Hook 1");
        Assert.Contains(hooks, h => h.Name == "Hook 2");
    }

    [Fact]
    public async Task LoadScriptAsync_WithNonExistentFile_ReturnsEmptyList()
    {
        var scriptPath = Path.Combine(_testDirectory, "NonExistent.cs");

        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        Assert.Empty(hooks);
    }

    [Fact]
    public async Task LoadScriptAsync_WithInvalidSyntax_ReturnsEmptyList()
    {
        var scriptPath = Path.Combine(_testDirectory, "InvalidSyntax.cs");
        var scriptContent = @"
public class InvalidHook : IMLoopHook
{
    // Missing required method implementation
    public string Name => ""Invalid"";
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        Assert.Empty(hooks);
    }

    [Fact]
    public async Task LoadScriptAsync_UsesCaching_LoadsFromDllSecondTime()
    {
        var scriptPath = Path.Combine(_testDirectory, "CachedHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class CachedHook : IMLoopHook
{
    public string Name => ""Cached Hook"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent);

        var hooks1 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        var hooks2 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        Assert.Single(hooks1);
        Assert.Single(hooks2);
        Assert.Single(cachedFiles1);
        Assert.Single(cachedFiles2);
        Assert.Equal(cachedFiles1[0], cachedFiles2[0]);
    }

    [Fact]
    public async Task LoadScriptAsync_WithModifiedScript_RecompilesWithNewHash()
    {
        var scriptPath = Path.Combine(_testDirectory, "ModifiableHook.cs");
        var scriptContent1 = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class ModifiableHook : IMLoopHook
{
    public string Name => ""Version 1"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent1);

        var hooks1 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles1 = Directory.GetFiles(_cacheDirectory, "*.dll");

        var scriptContent2 = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class ModifiableHook : IMLoopHook
{
    public string Name => ""Version 2"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}";
        await File.WriteAllTextAsync(scriptPath, scriptContent2);

        var hooks2 = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);
        var cachedFiles2 = Directory.GetFiles(_cacheDirectory, "*.dll");

        Assert.Single(hooks1);
        Assert.Equal("Version 1", hooks1[0].Name);
        Assert.Single(hooks2);
        Assert.Equal("Version 2", hooks2[0].Name);
        Assert.Single(cachedFiles1);
        Assert.Equal(2, cachedFiles2.Length);
    }

    [Fact]
    public void ClearCache_RemovesAllCachedDlls()
    {
        Directory.CreateDirectory(_cacheDirectory);
        var dummyFile = Path.Combine(_cacheDirectory, "dummy.dll");
        File.WriteAllText(dummyFile, "dummy content");

        _scriptLoader.ClearCache();

        Assert.True(Directory.Exists(_cacheDirectory));
        Assert.Empty(Directory.GetFiles(_cacheDirectory));
    }

    [Fact]
    public async Task LoadScriptAsync_WithAbstractClass_IgnoresAbstractClass()
    {
        var scriptPath = Path.Combine(_testDirectory, "AbstractHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

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

        var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptPath);

        Assert.Single(hooks);
        Assert.Equal("Concrete Hook", hooks[0].Name);
    }
}
