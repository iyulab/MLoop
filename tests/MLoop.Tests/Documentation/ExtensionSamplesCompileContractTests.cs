using System.Text.RegularExpressions;
using MLoop.Core.Scripting;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Metrics;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every hook and metric a document shows must compile as MLoop compiles a script, and every shipped example must too.
/// </summary>
/// <remarks>
/// The examples and the samples in three documents were written against an interface MLoop no longer has (<c>HigherIsBetter</c>,
/// <c>Task&lt;double&gt;</c>, the namespace <c>MLoop.Extensibility</c>), and none of them compiled: the README told the reader to copy a
/// hook to a path where it never runs, to pass a metric script to <c>--metric</c> (refused), and named two example files that did
/// not exist. Nothing checked, because a code block in a document is not built. This builds them — through
/// <see cref="ScriptLoader"/>, the compiler a project's scripts go through, so a sample that passes here is one that loads.
/// </remarks>
public class ExtensionSamplesCompileContractTests : IDisposable
{
    private readonly string _work = Path.Combine(Path.GetTempPath(), "mloop-ext-" + Guid.NewGuid().ToString("N"));

    public ExtensionSamplesCompileContractTests() => Directory.CreateDirectory(_work);

    public void Dispose()
    {
        try { Directory.Delete(_work, recursive: true); } catch { }
    }

    private static readonly Regex CSharpBlock = new(@"```csharp\r?\n(?<code>.*?)\r?\n```", RegexOptions.Singleline | RegexOptions.Compiled);

    /// <summary>The documents whose code blocks are scripts a reader would copy.</summary>
    private static readonly string[] Documents =
    [
        "src/MLoop.Extensibility/README.md",
        "docs/ARCHITECTURE.md",
    ];

    /// <summary>Scripts to build, named for the failure message: the example files, and the hook/metric blocks of the documents.</summary>
    internal static IEnumerable<(string Name, string Code)> Samples()
    {
        foreach (var directory in new[] { "extensions", "preprocessing-scripts" })
            foreach (var file in Directory.EnumerateFiles(Path.Combine(RepoSourceTree.RepoRoot, "examples", directory), "*.cs", SearchOption.AllDirectories))
                yield return (Path.GetRelativePath(RepoSourceTree.RepoRoot, file).Replace('\\', '/'), File.ReadAllText(file));

        foreach (var document in Documents)
        {
            var text = File.ReadAllText(Path.Combine(RepoSourceTree.RepoRoot, document));
            int n = 0;
            foreach (Match block in CSharpBlock.Matches(text))
            {
                var code = block.Groups["code"].Value;
                if (IsScript(code))
                    yield return ($"{document} (script block {++n})", code);
            }
        }
    }

    /// <summary>A block is a script when it declares a hook or a metric — the other blocks of a document (a fragment, a config) are not built.</summary>
    internal static bool IsScript(string code) => code.Contains(": IMLoopHook", StringComparison.Ordinal) || code.Contains(": IMLoopMetric", StringComparison.Ordinal);

    [Fact]
    public async Task EveryExampleAndEveryDocumentedScriptCompilesAndLoads()
    {
        var loader = new ScriptLoader(Path.Combine(_work, "cache"));
        var failures = new List<string>();
        int index = 0;
        foreach (var (name, code) in Samples())
        {
            var path = Path.Combine(_work, $"sample-{++index}.cs");
            await File.WriteAllTextAsync(path, code);
            int loaded = code.Contains(": IMLoopHook", StringComparison.Ordinal) ? (await loader.LoadScriptAsync<IMLoopHook>(path)).Count
                : code.Contains(": IPreprocessingScript", StringComparison.Ordinal) ? (await loader.LoadScriptAsync<IPreprocessingScript>(path)).Count
                : (await loader.LoadScriptAsync<IMLoopMetric>(path)).Count;
            if (loaded == 0)
                failures.Add(name);
        }

        Assert.True(failures.Count == 0, "These samples do not compile as MLoop scripts: " + string.Join("; ", failures));
    }

    /// <summary>
    /// A guard that reads documents must be shown to see them: a scan whose pattern stopped matching would report a clean tree
    /// either way. The three examples and the documented hook / metric blocks are known sites.
    /// </summary>
    [Fact]
    public void TheGuardSeesTheExamplesAndTheDocumentedScripts()
    {
        var samples = Samples().Select(s => s.Name).ToList();

        Assert.Contains("examples/extensions/hooks/DataValidationHook.cs", samples);
        Assert.Contains("examples/extensions/metrics/ProfitMetric.cs", samples);
        Assert.Contains("examples/extensions/metrics/ChurnCostMetric.cs", samples);
        Assert.True(samples.Count(s => s.StartsWith("examples/preprocessing-scripts/", StringComparison.Ordinal)) >= 7, "the seven preprocessing examples");
        Assert.True(samples.Count(s => s.StartsWith("src/MLoop.Extensibility/README.md", StringComparison.Ordinal)) >= 2, "the package README shows a hook and a metric");
        Assert.True(samples.Count(s => s.StartsWith("docs/ARCHITECTURE.md", StringComparison.Ordinal)) >= 2, "the architecture document shows a hook and a metric");
    }

    /// <summary>
    /// The READMEs <c>mloop init</c> writes into <c>.mloop/scripts/hooks/</c> and <c>.mloop/scripts/metrics/</c> are the first thing a
    /// reader of a new project sees; the hook sample there was the old one. They are generated text, so they are read where they land.
    /// </summary>
    [Fact]
    public async Task TheScriptsInTheReadmesThatInitWritesCompile()
    {
        var original = Directory.GetCurrentDirectory();
        string hooks, metrics;
        try
        {
            Directory.SetCurrentDirectory(_work);
            Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
            hooks = await File.ReadAllTextAsync(Path.Combine(_work, "p", ".mloop", "scripts", "hooks", "README.md"));
            metrics = await File.ReadAllTextAsync(Path.Combine(_work, "p", ".mloop", "scripts", "metrics", "README.md"));
        }
        finally
        {
            Directory.SetCurrentDirectory(original);
        }

        var loader = new ScriptLoader(Path.Combine(_work, "cache3"));
        var hook = CSharpBlock.Matches(hooks).Select(m => m.Groups["code"].Value).Single(IsScript);
        var metric = CSharpBlock.Matches(metrics).Select(m => m.Groups["code"].Value).Single(IsScript);
        var hookPath = Path.Combine(_work, "init-hook.cs");
        var metricPath = Path.Combine(_work, "init-metric.cs");
        await File.WriteAllTextAsync(hookPath, hook);
        await File.WriteAllTextAsync(metricPath, metric);

        Assert.Single(await loader.LoadScriptAsync<IMLoopHook>(hookPath));
        Assert.Single(await loader.LoadScriptAsync<IMLoopMetric>(metricPath));
    }

    [Fact]
    public async Task TheGuardRefusesTheShapeItForbids_AndAcceptsAHealthyOne()
    {
        var loader = new ScriptLoader(Path.Combine(_work, "cache2"));

        // the pre-rewrite sample: old namespace, an interface member that no longer exists, Task<double>
        var stale = Path.Combine(_work, "stale.cs");
        await File.WriteAllTextAsync(stale, "using MLoop.Extensibility;\npublic class M : IMLoopMetric { public string Name => \"x\"; public bool HigherIsBetter => true;\n  public async Task<double> CalculateAsync(MetricContext ctx) => 0; }");
        Assert.Empty(await loader.LoadScriptAsync<IMLoopMetric>(stale));
        Assert.True(IsScript(File.ReadAllText(stale)));

        var healthy = Path.Combine(_work, "healthy.cs");
        await File.WriteAllTextAsync(healthy, "using System.Threading.Tasks;\nusing MLoop.Extensibility.Metrics;\npublic class M : IMLoopMetric { public string Name => \"x\"; public string Description => \"d\";\n  public Task<MetricResult> CalculateAsync(MetricContext ctx) => Task.FromResult(new MetricResult { Name = Name, Value = 1 }); }");
        Assert.Single(await loader.LoadScriptAsync<IMLoopMetric>(healthy));
    }
}
