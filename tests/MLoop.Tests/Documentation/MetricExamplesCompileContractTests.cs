using MLoop.Core.Scripting;
using MLoop.Extensibility.Metrics;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The metric scripts the documentation ships compile against <see cref="IMLoopMetric"/> and yield one
/// metric each. They had not compiled for as long as anything ran metric scripts (nothing did, until
/// training computed them): missing <c>using</c> directives, and a task name the product never passes.
/// </summary>
public class MetricExamplesCompileContractTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), $"mloop-metric-examples-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try { Directory.Delete(_cache, recursive: true); } catch (IOException) { }
    }

    public static TheoryData<string> Examples()
    {
        var data = new TheoryData<string>();
        foreach (var file in Directory.GetFiles(Path.Combine(RepoSourceTree.RepoRoot, "docs", "examples", "metrics"), "*.cs").Order())
            data.Add(Path.GetFileName(file));
        return data;
    }

    [Fact]
    public void TheExamplesAreFound()
        => Assert.True(Examples().Count >= 3, "the scan found no metric examples — the guard below would pass on nothing");

    [Theory]
    [MemberData(nameof(Examples))]
    public async Task AMetricExample_CompilesToOneMetric_ForTheTaskNamesTheProductPasses(string file)
    {
        var errors = new List<string>();
        var path = Path.Combine(RepoSourceTree.RepoRoot, "docs", "examples", "metrics", file);

        var metrics = await new ScriptLoader(_cache, errors.Add).LoadScriptAsync<IMLoopMetric>(path);

        Assert.True(metrics.Count == 1, string.Join(Environment.NewLine, errors));
        Assert.DoesNotContain("\"BinaryClassification\"", File.ReadAllText(path));
    }
}
