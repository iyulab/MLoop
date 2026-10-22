using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
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
        Assert.True(Directory.Exists(_discovery.GetDetectorsDirectory()));
        Assert.True(Directory.Exists(_discovery.GetMetricsDirectory()));
    }

    [Fact]
    public async Task DiscoverMetricsAsync_WithNoMetricsDirectory_ReturnsEmptyList()
    {
        var metrics = await _discovery.DiscoverMetricsAsync();
        Assert.Empty(metrics);
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
    public async Task DiscoverMetricsAsync_WithMultipleScripts_ReturnsAll()
    {
        _discovery.InitializeDirectories();

        var metric1x = Path.Combine(_discovery.GetMetricsDirectory(), "Metric1.cs");
        await File.WriteAllTextAsync(metric1x, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class Metric1 : IMLoopMetric
{
    public string Name => ""Metric 1"";
    public string Description => ""Metric 1"";
    public Task<MetricResult> CalculateAsync(MetricContext context) => Task.FromResult(new MetricResult { Name = Name, Value = 0.5 });
}");

        var metric2x = Path.Combine(_discovery.GetMetricsDirectory(), "Metric2.cs");
        await File.WriteAllTextAsync(metric2x, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class Metric2 : IMLoopMetric
{
    public string Name => ""Metric 2"";
    public string Description => ""Metric 2"";
    public Task<MetricResult> CalculateAsync(MetricContext context) => Task.FromResult(new MetricResult { Name = Name, Value = 0.5 });
}");

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Equal(2, metrics.Count);
        Assert.Contains(metrics, h => h.Name == "Metric 1");
        Assert.Contains(metrics, h => h.Name == "Metric 2");
    }

    [Fact]
    public async Task DiscoverMetricsAsync_WithInvalidScript_ContinuesWithOtherScripts()
    {
        _discovery.InitializeDirectories();

        var validMetric = Path.Combine(_discovery.GetMetricsDirectory(), "ValidMetric.cs");
        await File.WriteAllTextAsync(validMetric, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class ValidMetric : IMLoopMetric
{
    public string Name => ""Valid Metric"";
    public string Description => ""Valid Metric"";
    public Task<MetricResult> CalculateAsync(MetricContext context) => Task.FromResult(new MetricResult { Name = Name, Value = 0.5 });
}");

        var invalidMetric = Path.Combine(_discovery.GetMetricsDirectory(), "InvalidMetric.cs");
        await File.WriteAllTextAsync(invalidMetric, "public class InvalidMetric { }");

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Single(metrics);
        Assert.Equal("Valid Metric", metrics[0].Name);
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
    public async Task DiscoverMetricsAsync_IgnoresNonCsFiles()
    {
        _discovery.InitializeDirectories();

        var metricScript = Path.Combine(_discovery.GetMetricsDirectory(), "ValidMetric.cs");
        await File.WriteAllTextAsync(metricScript, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class ValidMetric : IMLoopMetric
{
    public string Name => ""Valid Metric"";
    public string Description => ""Valid Metric"";
    public Task<MetricResult> CalculateAsync(MetricContext context) => Task.FromResult(new MetricResult { Name = Name, Value = 0.5 });
}");

        var textFile = Path.Combine(_discovery.GetMetricsDirectory(), "README.txt");
        await File.WriteAllTextAsync(textFile, "Some documentation");

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Single(metrics);
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
        var path = _discovery.GetMetricsDirectory();

        // Assert
        Assert.Equal(Path.Combine(_testProjectRoot, ".mloop", "scripts", "metrics"), path);
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
    public void GetDetectorsDirectory_ReturnsCorrectPath()
    {
        var path = _discovery.GetDetectorsDirectory();

        Assert.Equal(Path.Combine(_testProjectRoot, ".mloop", "scripts", "detectors"), path);
    }

    [Fact]
    public void InitializeDirectories_CreatesDetectorsDirectory()
    {
        _discovery.InitializeDirectories();

        Assert.True(Directory.Exists(_discovery.GetDetectorsDirectory()));
    }

    [Fact]
    public async Task DiscoverDetectorsAsync_WithNoDetectorsDirectory_ReturnsEmptyList()
    {
        var detectors = await _discovery.DiscoverDetectorsAsync();
        Assert.Empty(detectors);
    }

    [Fact]
    public async Task DiscoverDetectorsAsync_WithValidDetectorScript_ReturnsDetector()
    {
        _discovery.InitializeDirectories();
        var detectorScript = Path.Combine(_discovery.GetDetectorsDirectory(), "TestDetector.cs");
        var scriptContent = @"
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

public class TestDiscoveryDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.BusinessRule;
    public bool IsApplicable(DataFrameColumn column) => true;
    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column, string columnName, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<DetectedPattern> patterns = new List<DetectedPattern>();
        return Task.FromResult(patterns);
    }
}";
        await File.WriteAllTextAsync(detectorScript, scriptContent);

        var detectors = await _discovery.DiscoverDetectorsAsync();

        Assert.Single(detectors);
        Assert.Equal(PatternType.BusinessRule, detectors[0].PatternType);
    }

    /// <summary>
    /// A project with no scripts directory discovers nothing — the half of "zero overhead when
    /// not used" that holds on any machine, at any load, and so is the half that runs everywhere.
    /// </summary>
    [Fact]
    public async Task DiscoverMetricsAsync_NoDirectory_DiscoversNothing()
    {
        Assert.Empty(await _discovery.DiscoverMetricsAsync());
        // Repeated: a probe that found nothing must not have left something behind for the next one.
        Assert.Empty(await _discovery.DiscoverMetricsAsync());
    }

    /// <summary>
    /// The timing half of the same guarantee. A wall-clock upper bound measures the machine as much
    /// as the code, so it carries the category that keeps it out of CI — every other timing bound in
    /// this suite already does, and this one's omission is what put a shared runner's scheduling
    /// jitter on the path of a release.
    /// </summary>
    [Fact]
    [Trait("Category", "Slow")]
    public async Task DiscoverMetricsAsync_Performance_FastWhenNoDirectory()
    {
        // Warmup: JIT compile the code path
        await _discovery.DiscoverMetricsAsync();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var metrics = await _discovery.DiscoverMetricsAsync();
        stopwatch.Stop();

        Assert.Empty(metrics);
        Assert.True(stopwatch.ElapsedMilliseconds < 50,
            $"Discovery took {stopwatch.ElapsedMilliseconds}ms, expected < 50ms");
    }
}
