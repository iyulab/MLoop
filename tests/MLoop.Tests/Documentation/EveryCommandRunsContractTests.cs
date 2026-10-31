using System.CommandLine;
using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.CLI;
using MLoop.Tests.Commands;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every command that does work runs once, from the command line, in a project with a trained model, and shows one effect - a file it
/// wrote or a JSON answer that parses. <c>mloop drift</c> had no test that reached it through the command line (its logic is tested on
/// the library side), and it is the command whose new output ships next. A command that is not in <see cref="Runs"/> must be in
/// <see cref="Exempt"/> with the reason it cannot run here and the test class that covers it, so a new command has to declare itself
/// in one place - it cannot pass by printing help, because a group command with no action of its own is not a leaf and only the
/// commands that have an action are asked for.
/// </summary>
[Collection("FileSystem")]
public sealed class EveryCommandRunsContractTests : IClassFixture<EveryCommandRunsContractTests.TrainedProject>
{
    private readonly TrainedProject _project;

    public EveryCommandRunsContractTests(TrainedProject project) => _project = project;

    /// <summary>What a case saw: the exit code and the two streams.</summary>
    internal sealed record Seen(int Exit, string Out, string Err)
    {
        public JsonElement Json() => JsonDocument.Parse(Out).RootElement.Clone();
    }

    /// <summary>
    /// One invocation and the effect that shows it did its work. <paramref name="Then"/> runs after the effect is checked, to put back
    /// what the command changed for the commands that follow (a decision threshold stands in the way of calibration).
    /// </summary>
    internal sealed record Case(string[] Args, Action<Seen, string> Effect, string[]? Then = null, int Exit = 0,
        Action<Seen, string>? OnWorkbook = null);

    private static void Parses(Seen seen, string project) => seen.Json();

    /// <summary>The columns of the project's data, in order - what a command that read the workbook as a table can name.</summary>
    internal static readonly string[] Columns = ["x", "z", "w", "y"];

    private static Action<Seen, string> NamesTheColumns(Func<JsonElement, IEnumerable<string>> columnsOf) => (seen, _) =>
        Assert.Equal(Columns, columnsOf(seen.Json()));

    private static Action<Seen, string> File(string relative) => (_, project) =>
        Assert.True(System.IO.File.Exists(Path.Combine(project, relative)) || Directory.Exists(Path.Combine(project, relative)),
            relative + " was not written");

    /// <summary>The commands the fixture runs while it builds the project - the effect of each is asserted there.</summary>
    internal static readonly string[] SetUp = ["init", "train", "promote", "predict"];

    internal static readonly Dictionary<string, Case> Runs = new(StringComparer.Ordinal)
    {
        ["list"] = new(["list", "--json"], (s, _) => Assert.Contains("exp-001", s.Out)),
        ["status"] = new(["status", "--json"], Parses),
        ["validate"] = new(["validate"], (s, _) => Assert.NotEmpty(s.Out)),
        ["logs"] = new(["logs", "--json"], Parses),
        ["logs prune"] = new(["logs", "prune", "--days", "30", "--dry-run", "--json"], Parses),
        ["feedback list"] = new(["feedback", "list", "--json"], Parses),
        ["feedback metrics"] = new(["feedback", "metrics", "--json"], Parses),
        ["sample create"] = new(["sample", "create", "--from", "datasets/train.csv", "--rows", "50", "-o", "datasets/sample.csv"], File("datasets/sample.csv"),
            OnWorkbook: (seen, project) =>
            {
                File("datasets/sample.csv")(seen, project);
                // the table's rows, not the zip's bytes: a header of its columns, and 50 rows under it
                var lines = System.IO.File.ReadAllLines(Path.Combine(project, "datasets", "sample.csv"));
                Assert.Equal("x,z,w,y", lines[0].TrimStart('\uFEFF'));
                Assert.Equal(51, lines.Length);
            }),
        ["sample stats"] = new(["sample", "stats"], (s, _) => Assert.NotEmpty(s.Out)),
        ["trigger check"] = new(["trigger", "check"], (s, _) => Assert.NotEmpty(s.Out)),
        ["drift"] = new(["drift", "--json"], (s, _) =>
        {
            var json = s.Json();
            Assert.Equal(JsonValueKind.Object, json.GetProperty("drift").ValueKind);
        }),
        ["threshold"] = new(["threshold", "exp-001", "--precision", "0.8", "--data", "datasets/test.csv"], (s, _) => Assert.NotEmpty(s.Out),
            Then: ["threshold", "exp-001", "--clear"]),
        ["calibrate"] = new(["calibrate", "exp-001", "--data", "datasets/test.csv", "--method", "isotonic"], (s, _) => Assert.NotEmpty(s.Out),
            Then: ["calibrate", "exp-001", "--clear"]),
        ["explain"] = new(["explain", "exp-001", "--data", "datasets/test.csv", "--repeats", "2", "--rows", "100", "--json"], Parses),
        ["export onnx"] = new(["export", "onnx", "exp-001", "--data", "datasets/train.csv", "-o", "onnx-out"], File("onnx-out")),
        ["info"] = new(["info", "datasets/train.csv", "--json"], Parses,
            OnWorkbook: NamesTheColumns(j => j.GetProperty("columns").EnumerateArray().Select(c => c.GetProperty("name").GetString()!))),
        ["analyze profile"] = new(["analyze", "profile", "datasets/train.csv", "--json"], Parses,
            OnWorkbook: NamesTheColumns(j => j.GetProperty("data").GetProperty("columns").EnumerateArray().Select(c => c.GetProperty("name").GetString()!))),
        ["analyze correlation"] = new(["analyze", "correlation", "datasets/train.csv", "--json"], Parses),
        ["analyze importance"] = new(["analyze", "importance", "datasets/train.csv", "--label", "y", "--json"], Parses),
        ["analyze outliers"] = new(["analyze", "outliers", "datasets/train.csv", "--json"], Parses),
        ["analyze distribution"] = new(["analyze", "distribution", "datasets/train.csv", "--json"], Parses),
        ["detect"] = new(["detect", "datasets/train.csv", "--column", "x", "--json"], Parses),
        ["evaluate"] = new(["evaluate", "exp-001", "datasets/test.csv", "--json"], Parses),
        ["new hook"] = new(["new", "hook", "pre-train", "pre-train"], File(".mloop/scripts/hooks")),
        ["token"] = new(["token"], (s, _) => Assert.NotEmpty(s.Out.Trim())),
        ["prep plan"] = new(["prep", "plan", "--set", "fill-missing:mean", "--columns", "x", "--json"], Parses),
        ["features select"] = new(["features", "select", "--drop", "z", "--json"], Parses),
        ["docker"] = new(["docker"], File("Dockerfile")),
        ["runtime list"] = new(["runtime", "list", "--json"], Parses),
        ["encoder list"] = new(["encoder", "list", "--json"], Parses),
        ["preprocess"] = new(["preprocess", "datasets/train.csv"], (s, _) => Assert.NotEmpty(s.Out)),
    };

    /// <summary>A command that cannot run in a test, why, and the class that covers it (named by type, so a rename breaks the build).</summary>
    internal static readonly Dictionary<string, (string Reason, Type? Covered)> Exempt = new(StringComparer.Ordinal)
    {
        ["serve"] = ("a server that does not return", typeof(ServeCommandTests)),
        ["update"] = ("asks the network for a newer release", null),
        ["compare"] = ("needs two experiments of one project", typeof(CompareCommandTests)),
        ["rollback"] = ("needs two promotions of one project", typeof(PromoteCommandTests)),
        ["feedback add"] = ("needs the id of a logged prediction", typeof(LogsCommandTests)),
        ["analyze series"] = ("needs a time-series file", typeof(AnalyzeCommandTests)),
        ["prep run"] = ("runs what 'prep plan' declared on its own file", typeof(PrepRunCommandTests)),
        ["embed"] = ("needs an installed image encoder", typeof(EmbedCommandTests)),
        ["runtime install"] = ("downloads a runtime", typeof(RuntimeCommandJsonTests)),
        ["runtime remove"] = ("removes an installed runtime", typeof(RuntimeCommandJsonTests)),
        ["encoder install"] = ("downloads an encoder", typeof(EmbedCommandTests)),
        ["encoder add"] = ("needs an ONNX encoder file", typeof(EmbedCommandTests)),
        ["encoder remove"] = ("removes an installed encoder", typeof(EmbedCommandTests)),
    };

    /// <summary>Every command with an action of its own, by its words joined with a space (<c>feedback add</c>).</summary>
    internal static List<string> Leaves(Command root) => [.. Walk(root, "").Order(StringComparer.Ordinal)];

    private static IEnumerable<string> Walk(Command parent, string prefix)
    {
        foreach (var child in parent.Subcommands.Where(c => !c.Hidden))
        {
            var path = prefix + child.Name;
            if (child.Action is not null) yield return path;
            foreach (var below in Walk(child, path + " ")) yield return below;
        }
    }

    public static TheoryData<string> RunNames()
    {
        var data = new TheoryData<string>();
        foreach (var name in Runs.Keys.Order(StringComparer.Ordinal)) data.Add(name);
        return data;
    }

    [Theory]
    [MemberData(nameof(RunNames))]
    public async Task TheCommandRunsAndShowsItsEffect(string name)
    {
        var plan = Runs[name];
        var seen = await _project.RunAsync(plan.Args);
        Assert.True(seen.Exit == plan.Exit, $"mloop {string.Join(' ', plan.Args)} exited {seen.Exit}\n{seen.Out}\n{seen.Err}");
        plan.Effect(seen, _project.Root);
        if (plan.Then is { } undo) await _project.RunAsync(undo);
    }

    [Fact]
    public void EveryCommandIsRunOrExemptedWithAReason()
    {
        var leaves = Leaves(Program.BuildRootCommand());
        var declared = new HashSet<string>([.. SetUp, .. Runs.Keys, .. Exempt.Keys], StringComparer.Ordinal);

        Assert.True(leaves.Where(l => !declared.Contains(l)).ToList() is { Count: 0 } missing,
            "These commands have no case in Runs and no entry in Exempt (add one of the two):\n" + string.Join('\n', leaves.Where(l => !declared.Contains(l))));
        Assert.Empty(declared.Where(d => !leaves.Contains(d)).Order(StringComparer.Ordinal));
        Assert.Empty(Runs.Keys.Intersect(Exempt.Keys));
    }

    /// <summary>The data files of the project, by name: each exists as a CSV and as a workbook holding the same table.</summary>
    internal static readonly string[] DataFiles = ["train", "test"];

    private static bool IsOutputPath(string[] args, int i) => i > 0 && args[i - 1] is "-o" or "--output";

    /// <summary>The command line with each data file it reads named by its workbook instead of its CSV; an output path is left as it is.</summary>
    internal static string[] OnWorkbooks(string[] args) =>
        [.. args.Select((a, i) => !IsOutputPath(args, i) && DataFiles.Any(f => a == $"datasets/{f}.csv") ? Path.ChangeExtension(a, ".xlsx") : a)];

    /// <summary>The cases that read a data file named on their command line.</summary>
    internal static List<string> DataFileCases() =>
        [.. Runs.OrderBy(r => r.Key, StringComparer.Ordinal).Where(r => !OnWorkbooks(r.Value.Args).SequenceEqual(r.Value.Args)).Select(r => r.Key)];

    public static TheoryData<string> DataFileCommandNames()
    {
        var data = new TheoryData<string>();
        foreach (var name in DataFileCases()) data.Add(name);
        return data;
    }

    /// <summary>
    /// A data file is any table MLoop reads, and a command settles which one it was given by converting it once (<c>TabularDataFile</c>).
    /// <c>sample create</c>, <c>prep run</c>, <c>preprocess</c>, <c>features select</c> and <c>validate</c> each opened the path as lines of text
    /// instead: a workbook became 57 rows of a zip's bytes, or a column name written into <c>mloop.yaml</c>. Each command that reads a data
    /// file named on its command line runs here once more with a workbook of the same table, and must do what it does with the CSV.
    /// </summary>
    [Theory]
    [MemberData(nameof(DataFileCommandNames))]
    public async Task TheCommandReadsAWorkbookAsTheTableItHolds(string name)
    {
        var plan = Runs[name];
        var args = OnWorkbooks(plan.Args);
        Assert.NotEqual(plan.Args, args);

        var seen = await _project.RunAsync(args);

        Assert.True(seen.Exit == plan.Exit, $"mloop {string.Join(' ', args)} exited {seen.Exit}\n{seen.Out}\n{seen.Err}");
        (plan.OnWorkbook ?? plan.Effect)(seen, _project.Root);
        if (plan.Then is { } undo) await _project.RunAsync(undo);
    }

    [Fact]
    public void TheWorkbookRunsSeeTheCommandsThatReadDataFiles_AndAnUnknownCsvIsDeclared()
    {
        var names = DataFileCases();

        // It reads the table it polices: the commands known to read a data file are among its cases.
        foreach (var known in new[] { "info", "evaluate", "explain", "sample create", "analyze profile", "threshold", "calibrate", "export onnx", "detect" })
            Assert.Contains(known, names);

        // A case that names a CSV the fixture has no workbook of reads a data file this test cannot swap, so it must say how.
        var unswapped = Runs.Where(r => r.Value.Args.Where((a, i) => a.EndsWith(".csv", StringComparison.Ordinal) && !IsOutputPath(r.Value.Args, i))
            .Any(a => !DataFiles.Any(f => a == $"datasets/{f}.csv"))).Select(r => r.Key).ToList();
        Assert.True(unswapped.Count == 0, "These cases name a CSV the workbook run does not swap (add it to DataFiles or use a file in it):\n" + string.Join('\n', unswapped));

        // And the swap leaves an output path alone.
        Assert.Equal(["sample", "create", "--from", "datasets/train.xlsx", "--rows", "50", "-o", "datasets/sample.csv"],
            OnWorkbooks(Runs["sample create"].Args));
    }

    [Fact]
    public void TheEnumeration_SeesTheNestedCommandsAndNotTheGroupsThemselves()
    {
        var leaves = Leaves(Program.BuildRootCommand());

        Assert.Contains("feedback add", leaves);
        Assert.Contains("analyze profile", leaves);
        Assert.Contains("logs prune", leaves);
        Assert.Contains("logs", leaves);
        Assert.Contains("drift", leaves);
        Assert.DoesNotContain("feedback", leaves);
        Assert.DoesNotContain("analyze", leaves);
        Assert.True(leaves.Count > 40, "the walk must see the commands it polices; saw " + leaves.Count);

        // A group with no action of its own must not count, and one with both must.
        var child = new Command("child");
        child.SetAction(_ => 0);
        var both = new Command("both") { new Command("sub") };
        both.SetAction(_ => 0);
        both.Subcommands.Single().SetAction(_ => 0);
        var root = new RootCommand { new Command("group") { child }, both };
        Assert.Equal(["both", "both sub", "group child"], Leaves(root));
    }

    /// <summary>A binary project with a promoted model and four hundred logged predictions, built once for the class.</summary>
    public sealed class TrainedProject : IAsyncLifetime
    {
        private readonly string _originalDirectory = Directory.GetCurrentDirectory();
        private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-every-" + Guid.NewGuid().ToString("N"));

        public string Root => Path.Combine(_workspace, "p");

        public async ValueTask InitializeAsync()
        {
            Directory.CreateDirectory(_workspace);
            Directory.SetCurrentDirectory(_workspace);
            try
            {
                Check(await CliRunner.RunAsync("init", "p", "--task", "binary-classification"));
                Directory.SetCurrentDirectory(Root);
                await System.IO.File.WriteAllTextAsync("datasets/train.csv", Rows(300, 1, labelled: true));
                await System.IO.File.WriteAllTextAsync("datasets/test.csv", Rows(150, 2, labelled: true));
                await System.IO.File.WriteAllTextAsync("datasets/predict.csv", Rows(400, 3, labelled: false));
                Check(await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10"));
                Check(await CliRunner.RunAsync("promote", "exp-001"));
                Check(await CliRunner.RunAsync("predict", "datasets/predict.csv", "--log"));
                Assert.True(Directory.EnumerateFiles(Path.Combine(Root, ".mloop", "logs"), "*.jsonl", SearchOption.AllDirectories).Any(),
                    "predict --log wrote no log");

                // The same two tables as workbooks - made last, so the runs above never see them.
                foreach (var name in DataFiles)
                    WorkbookFile.FromCsv(Path.Combine("datasets", name + ".csv"), Path.Combine("datasets", name + ".xlsx"));
            }
            finally
            {
                Directory.SetCurrentDirectory(_originalDirectory);
            }
        }

        public ValueTask DisposeAsync()
        {
            try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
            try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
            return ValueTask.CompletedTask;
        }

        /// <summary>Runs a command line with the project as the current directory, which is where the commands look for it.</summary>
        internal async Task<Seen> RunAsync(params string[] args)
        {
            var before = Directory.GetCurrentDirectory();
            Directory.SetCurrentDirectory(Root);
            try
            {
                var (exit, stdout, stderr) = await CliRunner.RunAsync(args);
                return new Seen(exit, stdout, stderr);
            }
            finally
            {
                Directory.SetCurrentDirectory(before);
            }
        }

        private static void Check((int ExitCode, string Stdout, string Stderr) result) =>
            Assert.True(result.ExitCode == 0, result.Stdout + Environment.NewLine + result.Stderr);

        // x decides the answer (with some noise); z and w are noise.
        private static string Rows(int rows, int seed, bool labelled)
        {
            var csv = new StringBuilder(labelled ? "x,z,w,y\n" : "x,z,w\n");
            var rng = new Random(seed);
            for (var i = 0; i < rows; i++)
            {
                var x = rng.NextDouble();
                var yes = rng.NextDouble() < (x > 0.5 ? 0.92 : 0.08);
                csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{rng.NextDouble():F4}");
                if (labelled) csv.Append(yes ? ",yes" : ",no");
                csv.Append('\n');
            }
            return csv.ToString();
        }
    }
}
