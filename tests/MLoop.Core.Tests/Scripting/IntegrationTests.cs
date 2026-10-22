using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Metrics;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Scripting;

/// <summary>
/// Integration tests demonstrating end-to-end workflow with hooks and metrics.
/// </summary>
public class IntegrationTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly ScriptDiscovery _discovery;

    public IntegrationTests()
    {
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop_integration_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testProjectRoot);

        _discovery = new ScriptDiscovery(_testProjectRoot);
        _discovery.InitializeDirectories();
    }

    public void Dispose()
    {
        if (Directory.Exists(_testProjectRoot))
        {
            Directory.Delete(_testProjectRoot, recursive: true);
        }
    }

    [Fact]
    public async Task EndToEnd_MultipleMetrics_DiscoveredAndExecutable()
    {
        var metric1 = Path.Combine(_discovery.GetMetricsDirectory(), "Metric1.cs");
        await File.WriteAllTextAsync(metric1, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class Metric1 : IMLoopMetric
{
    public string Name => ""Metric 1"";
    public string Description => ""First metric"";
    public Task<MetricResult> CalculateAsync(MetricContext ctx)
        => Task.FromResult(new MetricResult { Name = Name, Value = 0.9 });
}");

        var metric2 = Path.Combine(_discovery.GetMetricsDirectory(), "Metric2.cs");
        await File.WriteAllTextAsync(metric2, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;
public class Metric2 : IMLoopMetric
{
    public string Name => ""Metric 2"";
    public string Description => ""Second metric"";
    public Task<MetricResult> CalculateAsync(MetricContext ctx)
        => Task.FromResult(new MetricResult { Name = Name, Value = 0.1 });
}");

        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Equal(2, metrics.Count);
        Assert.Contains(metrics, m => m.Name == "Metric 1");
        Assert.Contains(metrics, m => m.Name == "Metric 2");
    }

    /// <summary>
    /// Collects what a script logs, in memory.
    /// </summary>
    /// <remarks>
    /// It used to write to <see cref="Console"/>. Nothing asserted on that, but another test class
    /// in this assembly captures <see cref="Console.Out"/> to check the exclusion notices, and the
    /// runner executes distinct test classes in parallel — so these lines could land in that buffer
    /// while the lines that belonged there went missing from wherever they were meant to be read.
    /// Writing to a process-global stream to say something no one is listening for is a cost with no
    /// benefit; the messages stay reachable here for a test that ever does want them.
    /// </remarks>
    private class TestLogger : ILogger
    {
        public List<string> Messages { get; } = [];

        public void Info(string message) => Messages.Add($"[INFO] {message}");
        public void Warning(string message) => Messages.Add($"[WARN] {message}");
        public void Error(string message) => Messages.Add($"[ERROR] {message}");
        public void Error(string message, Exception exception) => Messages.Add($"[ERROR] {message}{Environment.NewLine}{exception}");
        public void Debug(string message) => Messages.Add($"[DEBUG] {message}");
    }

    /// <summary>
    /// Simple data class for binary classification tests.
    /// </summary>
    private class BinaryData
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public bool Label { get; set; }
    }
}
