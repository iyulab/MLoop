using System.CommandLine;
using System.Text.Json;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Embedding;
using MLoop.Tests.Commands;
using MLoop.Tests.Common;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The field names and JSON kinds the CLI's <c>--json</c> output carries, pinned against a recorded baseline
/// (<c>CliJsonShapes.baseline.txt</c>) - the same contract <c>WireShapeContractTests</c> keeps for the HTTP API. A script, an
/// agent bridge or a client library reads these fields by name; a hand-written assertion names only the ones it happens to
/// care about, so a renamed field used to pass every test and was found in a published version.
///
/// <para>The scenarios run on one project in the order a user lives it: the commands that read, then a second training run,
/// <c>compare</c>, <c>promote</c> and <c>rollback</c>, then the commands that change policy. Each records the shape of what the
/// command printed (the event stream of <c>train --json</c> one shape per event kind): every path with the kinds seen at it,
/// never a value. A path gone, renamed or retyped fails, and so does a new one - additive but deliberate: regenerate with
/// <c>MLOOP_WRITE_WIRE_SHAPES=&lt;path&gt;</c> and say so in the CHANGELOG. A command that takes <c>--json</c> and is neither run
/// here nor named in <see cref="Exempt"/> with the class that covers it fails as well.</para>
/// </summary>
public sealed class CliJsonShapeContractTests : IClassFixture<EveryCommandRunsContractTests.TrainedProject>
{
    private const string BaselineFile = "CliJsonShapes.baseline.txt";
    private const string WriteVariable = "MLOOP_WRITE_WIRE_SHAPES";

    private readonly EveryCommandRunsContractTests.TrainedProject _project;

    public CliJsonShapeContractTests(EveryCommandRunsContractTests.TrainedProject project) => _project = project;

    /// <summary>One invocation: the scenario id, the command it exercises (its words joined by a space) and the command line.</summary>
    private sealed record Scenario(string Id, string Command, params string[] Args)
    {
        /// <summary>Runs before the command, to put the project in the state the scenario needs (not recorded).</summary>
        public Func<CliJsonShapeContractTests, Task>? Before { get; init; }
    }

    /// <summary>A command that takes <c>--json</c> and cannot run here, why, and the class that covers its shape.</summary>
    private static readonly Dictionary<string, (string Why, Type CoveredBy)> Exempt = new(StringComparer.Ordinal)
    {
        ["init"] = ("creates a project of its own", typeof(ScaffoldingJsonContractTests)),
        ["embed"] = ("needs an installed image encoder", typeof(EmbedCommandTests)),
        ["encoder install"] = ("downloads an encoder", typeof(EmbedCommandTests)),
        ["analyze series"] = ("needs a time-series file", typeof(AnalyzeCommandTests)),
        ["prep run"] = ("runs what 'prep plan' declared on its own file", typeof(PrepRunCommandTests)),
    };

    private static List<Scenario> Scenarios() =>
    [
        // the commands that read the project as the fixture built it (one experiment, promoted, 400 logged predictions)
        new("list", "list", "list", "--json"),
        new("status", "status", "status", "--json"),
        new("validate", "validate", "validate", "--json"),
        new("logs", "logs", "logs", "--json", "--limit", "2"),
        new("logs-prune", "logs prune", "logs", "prune", "--days", "30", "--dry-run", "--json"),
        new("feedback-list", "feedback list", "feedback", "list", "--json"),
        new("feedback-metrics", "feedback metrics", "feedback", "metrics", "--json"),
        new("feedback-add", "feedback add", "feedback", "add", "--prediction-id", "{prediction}", "--actual-value", "yes", "--json"),
        new("feedback-list-after", "feedback list", "feedback", "list", "--json"),
        new("feedback-metrics-after", "feedback metrics", "feedback", "metrics", "--json"),
        new("trigger-check", "trigger check", "trigger", "check", "--json"),
        new("drift", "drift", "drift", "--json"),
        new("sample-stats", "sample stats", "sample", "stats", "--json"),
        new("sample-create", "sample create", "sample", "create", "--from", "datasets/train.csv", "--rows", "20", "-o", "datasets/s.csv", "--json"),
        new("info", "info", "info", "datasets/train.csv", "--json"),
        new("analyze-profile", "analyze profile", "analyze", "profile", "datasets/train.csv", "--json"),
        new("analyze-correlation", "analyze correlation", "analyze", "correlation", "datasets/train.csv", "--json"),
        new("analyze-importance", "analyze importance", "analyze", "importance", "datasets/train.csv", "--label", "y", "--json"),
        new("analyze-outliers", "analyze outliers", "analyze", "outliers", "datasets/train.csv", "--json"),
        new("analyze-distribution", "analyze distribution", "analyze", "distribution", "datasets/train.csv", "--json"),
        new("detect", "detect", "detect", "datasets/train.csv", "--column", "x", "--json"),
        new("evaluate", "evaluate", "evaluate", "exp-001", "datasets/test.csv", "--json"),
        new("evaluate-missing", "evaluate", "evaluate", "exp-099", "datasets/test.csv", "--json"),
        new("explain", "explain", "explain", "exp-001", "--data", "datasets/test.csv", "--repeats", "2", "--rows", "100", "--json"),
        new("predict", "predict", "predict", "datasets/test.csv", "--json"),
        new("threshold", "threshold", "threshold", "exp-001", "--precision", "0.8", "--data", "datasets/test.csv", "--json"),
        new("threshold-clear", "threshold", "threshold", "exp-001", "--clear", "--json"),
        new("calibrate", "calibrate", "calibrate", "exp-001", "--data", "datasets/test.csv", "--method", "isotonic", "--json"),
        new("calibrate-clear", "calibrate", "calibrate", "exp-001", "--clear", "--json"),
        new("export-onnx", "export onnx", "export", "onnx", "exp-001", "--data", "datasets/train.csv", "-o", "onnx-out", "--json"),
        new("runtime-list", "runtime list", "runtime", "list", "--json"),
        new("encoder-list", "encoder list", "encoder", "list", "--json"),
        new("rollback-none", "rollback", "rollback", "--json"),
        // a second training run, then what a user does with two experiments
        new("train", "train", "train", "datasets/train.csv", "--label", "y", "--time", "15", "--max-trials", "3", "--no-promote", "--json"),
        // The two below are written with metrics that are clearly weaker and clearly better than the trained model's, so the answers
        // (does it beat production? is it better?) do not depend on which model a short search happened to find.
        new("compare", "compare", "compare", "exp-004", "exp-001", "--json") { Before = t => t.SeedAsync(("exp-003", 0.6), ("exp-004", 0.999)) },
        new("promote-weaker", "promote", "promote", "exp-003", "--json"),
        new("promote-better", "promote", "promote", "exp-004", "--json"),
        new("promote-missing", "promote", "promote", "exp-099", "--json"),
        new("rollback", "rollback", "rollback", "--json"),
        // the commands that change policy come last, so the ones above ran on the project as trained
        new("prep-plan", "prep plan", "prep", "plan", "--set", "fill-missing:mean", "--columns", "x", "--json"),
        new("features-select", "features select", "features", "select", "--drop", "z", "--json"),
    ];

    [Fact]
    public async Task EveryJsonCommandPrintsTheRecordedShape()
    {
        var predictionId = await FirstPredictionIdAsync();
        var observed = new List<(string Id, string Header, IReadOnlyDictionary<string, string> Shape)>();
        // The encoder store lives outside any project (a user's installed encoders): an empty one, so the answer does not depend on the machine.
        var noEncoders = Path.Combine(Path.GetTempPath(), "mloop-no-encoders-" + Guid.NewGuid().ToString("N"));
        var before = Environment.GetEnvironmentVariable(EncoderStore.RootVariable);
        Directory.CreateDirectory(noEncoders);
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, noEncoders);
        try
        {
            foreach (var scenario in Scenarios())
            {
                if (scenario.Before is not null) await scenario.Before(this);
                var args = scenario.Args.Select(a => a.Replace("{prediction}", predictionId)).ToArray();
                var seen = await _project.RunAsync(args);
                observed.AddRange(Describe(scenario, seen));
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(EncoderStore.RootVariable, before);
            try { Directory.Delete(noEncoders, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }

        var rendered = WireShape.Render(observed);
        var target = Environment.GetEnvironmentVariable(WriteVariable);
        if (!string.IsNullOrEmpty(target))
        {
            await File.WriteAllTextAsync(target, rendered);
            return;
        }

        var baselinePath = Path.Combine(AppContext.BaseDirectory, BaselineFile);
        Assert.True(File.Exists(baselinePath), BaselineFile + " is copied beside the test assembly");
        // A real model's answers can leave a value out or write it as null (a threshold the model cannot reach); null says nothing of a kind.
        var differences = WireShape.Differences(WireShape.Parse(await File.ReadAllTextAsync(baselinePath)), WireShape.Parse(rendered), ignoreNull: true);
        Assert.True(differences.Count == 0,
            "the output's field names and kinds are a contract - a removed, renamed or retyped field breaks a consumer; a new one is " +
            "additive but deliberate: regenerate with " + WriteVariable + "=<path to tests/MLoop.Tests/Documentation/" + BaselineFile +
            "> and say so in the CHANGELOG.\n" + string.Join('\n', differences));
    }

    [Fact]
    public void EveryCommandThatTakesJsonIsRunHereOrExemptedWithAReason()
    {
        var takesJson = JsonCommands(Program.BuildRootCommand());
        Assert.NotEmpty(takesJson);
        Assert.Contains("promote", takesJson);   // the scan sees the command whose output the shape guard exists for

        var claimed = new HashSet<string>([.. Scenarios().Select(s => s.Command), .. Exempt.Keys], StringComparer.Ordinal);
        Assert.Empty(WireShape.Unclaimed(takesJson, claimed));
        foreach (var (command, (why, coveredBy)) in Exempt)
        {
            Assert.Contains(command, takesJson);   // an exemption for a command that no longer takes --json is stale
            Assert.False(string.IsNullOrWhiteSpace(why), command);
            Assert.Contains(coveredBy.GetMethods(), m => m.GetCustomAttributes(typeof(FactAttribute), true).Length > 0 ||
                                                         m.GetCustomAttributes(typeof(TheoryAttribute), true).Length > 0);
        }
        Assert.Empty(Exempt.Keys.Intersect(Scenarios().Select(s => s.Command)));
    }

    /// <summary>Every command with an action of its own that has a <c>--json</c> option, by its words joined with a space.</summary>
    private static List<string> JsonCommands(Command root)
    {
        var found = new List<string>();
        void Walk(Command parent, string prefix)
        {
            foreach (var child in parent.Subcommands.Where(c => !c.Hidden))
            {
                var path = prefix + child.Name;
                if (child.Action is not null && child.Options.Any(o => o.Name == "--json" || o.Aliases.Contains("--json"))) found.Add(path);
                Walk(child, path + " ");
            }
        }
        Walk(root, "");
        return found;
    }

    /// <summary>What a scenario printed, as one block - or one per event kind for an event stream (one JSON object a line).</summary>
    private static IEnumerable<(string Id, string Header, IReadOnlyDictionary<string, string> Shape)> Describe(
        Scenario scenario, EveryCommandRunsContractTests.Seen seen)
    {
        var text = seen.Out.Trim();
        var header = $"{scenario.Command} -> exit {seen.Exit}";
        if (text.Length == 0) { yield return (scenario.Id, header + " (no output)", new Dictionary<string, string>()); yield break; }

        if (TryDescribe(text, out var whole)) { yield return (scenario.Id, header, whole); yield break; }

        var lines = text.Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        var byEvent = new SortedDictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var line in lines)
        {
            if (!line.StartsWith('{')) continue;   // a text line among the events is not part of the stream's contract
            using var document = JsonDocument.Parse(line);
            var kind = document.RootElement.TryGetProperty("event", out var e) && e.ValueKind == JsonValueKind.String ? e.GetString()! : "(no event)";
            (byEvent.TryGetValue(kind, out var list) ? list : byEvent[kind] = []).Add(line);
        }
        if (byEvent.Count == 0) { yield return (scenario.Id, header + " (not JSON)", new Dictionary<string, string>()); yield break; }

        foreach (var (kind, events) in byEvent)
        {
            // the kinds seen across every event of this kind (an optional field is in the union)
            var union = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
            foreach (var shape in events.Select(WireShape.Describe))
                foreach (var (path, kinds) in shape)
                {
                    if (!union.TryGetValue(path, out var set)) union[path] = set = [];
                    foreach (var k in kinds.Split('|')) set.Add(k);
                }
            yield return (scenario.Id + "#" + kind, header, union.ToDictionary(p => p.Key, p => string.Join('|', p.Value)));
        }
    }

    private static bool TryDescribe(string text, out IReadOnlyDictionary<string, string> shape)
    {
        try { shape = WireShape.Describe(text); return true; }
        catch (JsonException) { shape = new Dictionary<string, string>(); return false; }
    }

    /// <summary>Writes experiments of the project's model that copy the trained model and carry the given metric value in every metric.</summary>
    private async Task SeedAsync(params (string Id, double Value)[] experiments)
    {
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _project.Root);
        var from = store.GetExperimentPath("default", "exp-001");
        foreach (var (id, value) in experiments)
        {
            var path = store.GetExperimentPath("default", id);
            Directory.CreateDirectory(path);
            File.Copy(Path.Combine(from, "model.zip"), Path.Combine(path, "model.zip"));
            var metrics = new[] { "accuracy", "auc", "auprc", "f1_score", "negative_precision", "negative_recall", "precision", "recall" }
                .ToDictionary(name => name, _ => value);
            metrics["ece"] = 1 - value;
            await store.SaveAsync("default", new ExperimentData
            {
                ModelName = "default",
                ExperimentId = id,
                Timestamp = new DateTime(2026, 11, 12, 9, 0, 0, DateTimeKind.Utc),
                Status = "Completed",
                Task = "binary-classification",
                Config = new ExperimentConfig { DataFile = "train.csv", LabelColumn = "y", Metric = "f1_score", TimeLimitSeconds = 60, TestSplit = 0.2 },
                Metrics = metrics,
            }, CancellationToken.None);
        }
    }

    // - the guard asserts that it can see: a rename still fails in the tolerant mode, an absent value does not -

    [Fact]
    public void ARenamedFieldStillFailsWhereNullsAreTolerated()
    {
        var before = WireShape.Parse(WireShape.Render([("s", "x -> exit 0", WireShape.Describe("""{"a":1,"b":"x"}"""))]));
        var after = WireShape.Parse(WireShape.Render([("s", "x -> exit 0", WireShape.Describe("""{"a":1,"c":"x"}"""))]));

        Assert.Equal(["s: removed $.b: string", "s: added $.c: string"], WireShape.Differences(before, after, ignoreNull: true));
    }

    [Fact]
    public void AValueThatIsNullInsteadOfAStringIsNotADifferenceWhereNullsAreTolerated_ButAMissingOneIs()
    {
        var withString = WireShape.Parse(WireShape.Render([("s", "x -> exit 0", WireShape.Describe("""{"a":1,"w":"text","n":null}"""))]));
        var withNull = WireShape.Parse(WireShape.Render([("s", "x -> exit 0", WireShape.Describe("""{"a":1,"w":null,"n":null}"""))]));
        var missing = WireShape.Parse(WireShape.Render([("s", "x -> exit 0", WireShape.Describe("""{"a":1,"n":null}"""))]));

        Assert.Empty(WireShape.Differences(withString, withNull, ignoreNull: true));
        Assert.Empty(WireShape.Differences(withNull, withString, ignoreNull: true));
        Assert.NotEmpty(WireShape.Differences(withString, withNull));   // the strict mode (the API's) still sees it
        Assert.Equal(["s: removed $.w: string"], WireShape.Differences(withString, missing, ignoreNull: true));   // the name is the contract
    }

    private async Task<string> FirstPredictionIdAsync()
    {
        var seen = await _project.RunAsync("logs", "--json", "--limit", "1");
        foreach (var name in new[] { "predictionId", "id" })
        {
            var id = FindString(seen.Json(), name);
            if (id is not null) return id;
        }
        throw new InvalidOperationException("no prediction id in: " + seen.Out);
    }

    private static string? FindString(JsonElement element, string property)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                if (element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String) return value.GetString();
                foreach (var child in element.EnumerateObject())
                    if (FindString(child.Value, property) is { } found) return found;
                return null;
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                    if (FindString(item, property) is { } found) return found;
                return null;
            default:
                return null;
        }
    }
}
