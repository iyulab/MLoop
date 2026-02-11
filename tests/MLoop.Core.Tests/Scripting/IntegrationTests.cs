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
    public async Task EndToEnd_Hook_ExecutesSuccessfully()
    {
        var hookScript = Path.Combine(_discovery.GetHooksDirectory(), "DataValidationHook.cs");
        var scriptContent = @"
using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;

public class DataValidationHook : IMLoopHook
{
    public string Name => ""Data Validation"";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var preview = ctx.DataView.Preview(maxRows: 100);
        var rowCount = preview.RowView.Length;

        if (rowCount < 10)
        {
            return HookResult.Abort($""Insufficient data: {rowCount} rows"");
        }

        ctx.Logger.Info($""Validation passed: {rowCount} rows"");
        return HookResult.Continue();
    }
}";
        await File.WriteAllTextAsync(hookScript, scriptContent);

        Assert.True(File.Exists(hookScript));

        var hooks = await _discovery.DiscoverHooksAsync();
        Assert.Single(hooks);

        var mlContext = new MLContext(seed: 1);
        var data = mlContext.Data.LoadFromEnumerable(new[]
        {
            new { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
            new { Feature1 = 3.0f, Feature2 = 4.0f, Label = false },
            new { Feature1 = 5.0f, Feature2 = 6.0f, Label = true },
            new { Feature1 = 7.0f, Feature2 = 8.0f, Label = false },
            new { Feature1 = 9.0f, Feature2 = 10.0f, Label = true },
            new { Feature1 = 11.0f, Feature2 = 12.0f, Label = false },
            new { Feature1 = 13.0f, Feature2 = 14.0f, Label = true },
            new { Feature1 = 15.0f, Feature2 = 16.0f, Label = false },
            new { Feature1 = 17.0f, Feature2 = 18.0f, Label = true },
            new { Feature1 = 19.0f, Feature2 = 20.0f, Label = false },
            new { Feature1 = 21.0f, Feature2 = 22.0f, Label = true },
        });

        var hookContext = new HookContext
        {
            HookType = HookType.PreTrain,
            HookName = "DataValidationHook",
            MLContext = mlContext,
            DataView = data,
            ProjectRoot = _testProjectRoot,
            Logger = new TestLogger(),
            Metadata = new Dictionary<string, object>
            {
                ["ExperimentId"] = "test-experiment-001"
            }
        };

        var result = await hooks[0].ExecuteAsync(hookContext);

        Assert.Equal(HookAction.Continue, result.Action);
        Assert.Equal("Data Validation", hooks[0].Name);
    }

    [Fact]
    public async Task EndToEnd_MultipleHooksAndMetrics_DiscoveredAndExecutable()
    {
        var hook1 = Path.Combine(_discovery.GetHooksDirectory(), "Hook1.cs");
        await File.WriteAllTextAsync(hook1, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class Hook1 : IMLoopHook
{
    public string Name => ""Hook 1"";
    public Task<HookResult> ExecuteAsync(HookContext ctx) => Task.FromResult(HookResult.Continue());
}");

        var hook2 = Path.Combine(_discovery.GetHooksDirectory(), "Hook2.cs");
        await File.WriteAllTextAsync(hook2, @"
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;
public class Hook2 : IMLoopHook
{
    public string Name => ""Hook 2"";
    public Task<HookResult> ExecuteAsync(HookContext ctx) => Task.FromResult(HookResult.Continue());
}");

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

        var hooks = await _discovery.DiscoverHooksAsync();
        var metrics = await _discovery.DiscoverMetricsAsync();

        Assert.Equal(2, hooks.Count);
        Assert.Equal(2, metrics.Count);
        Assert.Contains(hooks, h => h.Name == "Hook 1");
        Assert.Contains(hooks, h => h.Name == "Hook 2");
        Assert.Contains(metrics, m => m.Name == "Metric 1");
        Assert.Contains(metrics, m => m.Name == "Metric 2");
    }

    /// <summary>
    /// Simple test logger implementation.
    /// </summary>
    private class TestLogger : ILogger
    {
        public void Info(string message) => Console.WriteLine($"[INFO] {message}");
        public void Warning(string message) => Console.WriteLine($"[WARN] {message}");
        public void Error(string message) => Console.WriteLine($"[ERROR] {message}");
        public void Error(string message, Exception exception) => Console.WriteLine($"[ERROR] {message}{Environment.NewLine}{exception}");
        public void Debug(string message) => Console.WriteLine($"[DEBUG] {message}");
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
