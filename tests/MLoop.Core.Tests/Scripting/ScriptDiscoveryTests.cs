using MLoop.Core.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Metrics;

namespace MLoop.Core.Tests.Scripting;

public class ScriptDiscoveryTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly ScriptDiscovery _discovery;

    public ScriptDiscoveryTests()
    {
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop_discovery_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testProjectRoot);

        _discovery = new ScriptDiscovery(_testProjectRoot);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testProjectRoot))
        {
            Directory.Delete(_testProjectRoot, recursive: true);
        }
    }

    [Fact]
    public void IsExtensibilityAvailable_WithNoScriptsDirectory_ReturnsFalse()
    {
        // Act
        var result = _discovery.IsExtensibilityAvailable();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsExtensibilityAvailable_WithScriptsDirectory_ReturnsTrue()
    {
        // Arrange
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop", "scripts"));

        // Act
        var result = _discovery.IsExtensibilityAvailable();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void InitializeDirectories_CreatesStandardStructure()
    {
        // Act
        _discovery.InitializeDirectories();

        // Assert
        Assert.True(Directory.Exists(_discovery.GetHooksDirectory()));
        Assert.True(Directory.Exists(_discovery.GetMetricsDirectory()));
    }

    [Fact]
    public async Task DiscoverHooksAsync_WithNoHooksDirectory_ReturnsEmptyList()
    {
        var hooks = await _discovery.DiscoverHooksAsync();
        Assert.Empty(hooks);
    }

    [Fact]
    public async Task DiscoverMetricsAsync_WithNoMetricsDirectory_ReturnsEmptyList()
    {
        var metrics = await _discovery.DiscoverMetricsAsync();
        Assert.Empty(metrics);
    }

    [Fact]
    public async Task DiscoverHooksAsync_WithValidHookScript_ReturnsHook()
    {
        _discovery.InitializeDirectories();
        var hookScript = Path.Combine(_discovery.GetHooksDirectory(), "TestHook.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class TestDiscoveryHook : IMLoopHook
{
    public string Name => ""Test Discovery Hook"";
    public Task<HookResult> ExecuteAsync(HookContext context)
    {
        return Task.FromResult(HookResult.Continue());
    }
}";
        await File.WriteAllTextAsync(hookScript, scriptContent);

        var hooks = await _discovery.DiscoverHooksAsync();

        Assert.Single(hooks);
        Assert.Equal("Test Discovery Hook", hooks[0].Name);
    }

    [Fact]
    public async Task DiscoverMetricsAsync_WithValidMetricScript_ReturnsMetric()
    {
        _discovery.InitializeDirectories();
        var metricScript = Path.Combine(_discovery.GetMetricsDirectory(), "TestMetric.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class TestDiscoveryMetric : IMLoopMetric
{
    public string Name => ""Test Discovery Metric"";
    public string Description => ""Test metric for discovery"";

    public Task<MetricResult> CalculateAsync(MetricContext context)
    {
        return Task.FromResult(new MetricResult { Name = Name, Value = 0.85 });
    }
}";
        await File.WriteAllTextAsync(metricScript, scriptContent);

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Single(metrics);
        Assert.Equal("Test Discovery Metric", metrics[0].Name);
        Assert.Equal("Test metric for discovery", metrics[0].Description);
    }

    [Fact]
    public async Task DiscoverHooksAsync_WithMultipleScripts_ReturnsAllHooks()
    {
        _discovery.InitializeDirectories();

        var hook1 = Path.Combine(_discovery.GetHooksDirectory(), "Hook1.cs");
        await File.WriteAllTextAsync(hook1, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class Hook1 : IMLoopHook
{
    public string Name => ""Hook 1"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}");

        var hook2 = Path.Combine(_discovery.GetHooksDirectory(), "Hook2.cs");
        await File.WriteAllTextAsync(hook2, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class Hook2 : IMLoopHook
{
    public string Name => ""Hook 2"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}");

        var hooks = await _discovery.DiscoverHooksAsync();

        Assert.Equal(2, hooks.Count);
        Assert.Contains(hooks, h => h.Name == "Hook 1");
        Assert.Contains(hooks, h => h.Name == "Hook 2");
    }

    [Fact]
    public async Task DiscoverHooksAsync_WithInvalidScript_ContinuesWithOtherScripts()
    {
        _discovery.InitializeDirectories();

        var validHook = Path.Combine(_discovery.GetHooksDirectory(), "ValidHook.cs");
        await File.WriteAllTextAsync(validHook, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class ValidHook : IMLoopHook
{
    public string Name => ""Valid Hook"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}");

        var invalidHook = Path.Combine(_discovery.GetHooksDirectory(), "InvalidHook.cs");
        await File.WriteAllTextAsync(invalidHook, "public class InvalidHook { }");

        var hooks = await _discovery.DiscoverHooksAsync();

        Assert.Single(hooks);
        Assert.Equal("Valid Hook", hooks[0].Name);
    }

    [Fact]
    public async Task DiscoverMetricsAsync_WithMultipleMetricsInOneFile_ReturnsAll()
    {
        _discovery.InitializeDirectories();
        var metricScript = Path.Combine(_discovery.GetMetricsDirectory(), "MultipleMetrics.cs");
        var scriptContent = @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class Metric1 : IMLoopMetric
{
    public string Name => ""Metric 1"";
    public string Description => ""First metric"";
    public Task<MetricResult> CalculateAsync(MetricContext context)
        => Task.FromResult(new MetricResult { Name = Name, Value = 0.9 });
}

public class Metric2 : IMLoopMetric
{
    public string Name => ""Metric 2"";
    public string Description => ""Second metric"";
    public Task<MetricResult> CalculateAsync(MetricContext context)
        => Task.FromResult(new MetricResult { Name = Name, Value = 0.1 });
}";
        await File.WriteAllTextAsync(metricScript, scriptContent);

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Equal(2, metrics.Count);
        Assert.Contains(metrics, m => m.Name == "Metric 1");
        Assert.Contains(metrics, m => m.Name == "Metric 2");
    }

    [Fact]
    public async Task DiscoverHooksAsync_IgnoresNonCsFiles()
    {
        _discovery.InitializeDirectories();

        var hookScript = Path.Combine(_discovery.GetHooksDirectory(), "ValidHook.cs");
        await File.WriteAllTextAsync(hookScript, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class ValidHook : IMLoopHook
{
    public string Name => ""Valid Hook"";
    public Task<HookResult> ExecuteAsync(HookContext context) => Task.FromResult(HookResult.Continue());
}");

        var textFile = Path.Combine(_discovery.GetHooksDirectory(), "README.txt");
        await File.WriteAllTextAsync(textFile, "Some documentation");

        var hooks = await _discovery.DiscoverHooksAsync();

        Assert.Single(hooks);
    }

    [Fact]
    public void GetScriptsDirectory_ReturnsCorrectPath()
    {
        // Act
        var path = _discovery.GetScriptsDirectory();

        // Assert
        Assert.Equal(Path.Combine(_testProjectRoot, ".mloop", "scripts"), path);
    }

    [Fact]
    public void GetHooksDirectory_ReturnsCorrectPath()
    {
        // Act
        var path = _discovery.GetHooksDirectory();

        // Assert
        Assert.Equal(Path.Combine(_testProjectRoot, ".mloop", "scripts", "hooks"), path);
    }

    [Fact]
    public void GetMetricsDirectory_ReturnsCorrectPath()
    {
        // Act
        var path = _discovery.GetMetricsDirectory();

        // Assert
        Assert.Equal(Path.Combine(_testProjectRoot, ".mloop", "scripts", "metrics"), path);
    }

    [Fact]
    public async Task DiscoverHooksAsync_Performance_FastWhenNoDirectory()
    {
        // Warmup: JIT compile the code path
        await _discovery.DiscoverHooksAsync();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var hooks = await _discovery.DiscoverHooksAsync();
        stopwatch.Stop();

        Assert.Empty(hooks);
        Assert.True(stopwatch.ElapsedMilliseconds < 50,
            $"Discovery took {stopwatch.ElapsedMilliseconds}ms, expected < 50ms");
    }
}
