using Microsoft.ML;
using MLoop.Core.Metrics;
using MLoop.Core.Tests;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Metrics;

public class MetricEngineTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-metrics-{Guid.NewGuid():N}");
    private readonly Log _log = new();

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private sealed class Log : ILogger
    {
        public List<string> Warnings { get; } = [];
        public void Info(string message) { }
        public void Warning(string message) => Warnings.Add(message);
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
        public void Debug(string message) { }
    }

    private sealed class Row
    {
        public float X { get; set; }
        public bool Label { get; set; }
    }

    private void Script(string file, string name, string value) => File.WriteAllText(
        Path.Combine(Directory.CreateDirectory(Path.Combine(_root, ".mloop", "scripts", "metrics")).FullName, file), $$"""
        using System.Threading.Tasks;
        using MLoop.Extensibility.Metrics;
        public class {{Path.GetFileNameWithoutExtension(file)}}Metric : IMLoopMetric
        {
            public string Name => "{{name}}";
            public string Description => "";
            public Task<MetricResult> CalculateAsync(MetricContext ctx)
                => Task.FromResult(new MetricResult { Name = Name, Value = {{value}} });
        }
        """);

    private static (MLContext, IDataView) Scored()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 40).Select(i => new Row { X = i / 40f, Label = i >= 20 }));
        var model = ml.Transforms.Concatenate("Features", "X")
            .Append(ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression()).Fit(data);
        return (ml, model.Transform(data));
    }

    [Fact]
    public async Task AScriptsResult_IsAdded_ButNeverOverAMeasuredMetric_NorWhenNotFinite()
    {
        Script("Count.cs", "labeled_rows", "((bool[])ctx.Labels).Length");
        Script("Clash.cs", "accuracy", "0.0");
        Script("Nan.cs", "broken", "double.NaN");
        var metrics = new Dictionary<string, double> { ["accuracy"] = 0.9 };
        var (ml, scored) = Scored();

        var added = await new MetricEngine(_root, _log).AddToAsync(metrics, ml, scored, "binary-classification", "Label");

        Assert.Equal(["labeled_rows"], added);
        Assert.Equal(40, metrics["labeled_rows"]);
        Assert.Equal(0.9, metrics["accuracy"]);
        Assert.False(metrics.ContainsKey("broken"));
        Assert.Contains(_log.Warnings, w => w.Contains("'accuracy'") && w.Contains("already reports"));
        Assert.Contains(_log.Warnings, w => w.Contains("'broken'") && w.Contains("NaN"));
    }

    [Fact]
    public async Task NoScripts_AddsNothing_AndSaysNothing()
    {
        var (ml, scored) = Scored();
        Assert.Empty(await new MetricEngine(_root, _log).AddToAsync(new(), ml, scored, "binary-classification", "Label"));
        Assert.Empty(_log.Warnings);
    }

    [Fact]
    public async Task ATaskTheContractDoesNotCover_IsSaidOnce()
    {
        Script("Count.cs", "labeled_rows", "1");
        var (ml, scored) = Scored();

        Assert.Empty(await new MetricEngine(_root, _log).AddToAsync(new(), ml, scored, "clustering", "Label"));
        Assert.Contains("clustering", Assert.Single(_log.Warnings));
    }
}
