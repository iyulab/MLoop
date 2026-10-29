using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A test that trains through <c>mloop train</c>, <c>POST /train</c> or the training engine stops the trainer
/// search on a trial count, with the clock only as its ceiling.
/// </summary>
/// <remarks>
/// A search ended by a time budget does a different amount of work on a busy machine, and what it did is what the
/// test then sees (a trial recorded without its metric, a failed trial, no trial at all) in suites that asserted
/// nothing about time. <c>--max-trials</c> makes the work the same wherever it runs. The suites of
/// <c>MLoop.Core.Tests</c> have their own guard, and <c>MLoop.Search.Tests</c> measures the budget itself.
/// <para>
/// This reads the test sources. A training call that names a time and no trial limit is flagged unless its task is
/// not a trainer search — a literal task the search does not serve, or a <c>no search</c> comment saying why.
/// </para>
/// </remarks>
public class TrialCountedTrainingContractTests
{
    private static readonly string[] NotATrainerSearch =
    [
        "clustering", "anomaly-detection", "ranking", "recommendation", "forecasting", "time-series-anomaly",
        "image-classification", "object-detection", "image-anomaly", "text-classification", "sentence-similarity", "ner", "question-answering",
        "text-generation",   // a task no module serves: the test is of the refusal
    ];

    private static readonly Regex Initializer = new(@"new TrainingConfig\s*\{(?<body>[^{}]*)\}", RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly Regex CliStatement = new(@"CliRunner\.RunAsync\((?<args>[^;]*);?(?<tail>[^\n]*)", RegexOptions.Compiled);

    private static readonly Regex TrainRequest = new(@"new\s*\{(?<body>[^{}]*\bdataFile\b[^{}]*)\}", RegexOptions.Singleline | RegexOptions.Compiled);

    private static string Flat(string text) => Regex.Replace(text.Trim(), @"\s+", " ");

    private static bool NamesNoSearch(string text) => text.Contains("no search", StringComparison.Ordinal);

    private static bool LiteralTaskIsNotASearch(string text) =>
        NotATrainerSearch.Any(task => Regex.IsMatch(text, $@"\b[Tt]ask\s*=\s*""{Regex.Escape(task)}"""));

    internal static IEnumerable<string> EngineOffenders(string source, string name) =>
        Initializer.Matches(source)
            .Select(m => m.Groups["body"].Value)
            .Where(body => body.Contains("TimeLimitSeconds", StringComparison.Ordinal) && !body.Contains("MaxTrials", StringComparison.Ordinal))
            .Where(body => !LiteralTaskIsNotASearch(body) && !NamesNoSearch(body))
            .Select(body => $"{name}: new TrainingConfig {{ {Flat(body)} }}");

    internal static IEnumerable<string> CliOffenders(string source, string name) =>
        CliStatement.Matches(source)
            .Select(m => (Args: m.Groups["args"].Value, Tail: m.Groups["tail"].Value))
            .Where(c => c.Args.Contains("\"train\"", StringComparison.Ordinal) && c.Args.Contains("\"--time\"", StringComparison.Ordinal))
            .Where(c => !c.Args.Contains("\"--max-trials\"", StringComparison.Ordinal) && !NamesNoSearch(c.Args + c.Tail))
            .Select(c => $"{name}: RunAsync({Flat(c.Args)}");

    internal static IEnumerable<string> RequestOffenders(string source, string name) =>
        TrainRequest.Matches(source)
            .Select(m => m.Groups["body"].Value)
            .Where(body => Regex.IsMatch(body, @"\btask\s*=") && body.Contains("timeLimitSeconds", StringComparison.Ordinal))
            .Where(body => !body.Contains("maxTrials", StringComparison.Ordinal))
            .Where(body => !LiteralTaskIsNotASearch(body) && !NamesNoSearch(body))
            .Select(body => $"{name}: new {{ {Flat(body)} }}");

    private static IEnumerable<(string Name, string Text)> Sources() =>
        RepoSourceTree.SourceFiles(nameof(TrialCountedTrainingContractTests))
            .Where(f => !f.Contains("MLoop.Core.Tests") && !f.Contains("MLoop.Search.Tests"))
            .Select(f => (Path.GetRelativePath(RepoSourceTree.TestsRoot, f), File.ReadAllText(f)));

    [Fact]
    public void ATestThatTrainsStopsTheSearchOnATrialCount()
    {
        var sources = Sources().ToList();

        var offenders = sources.SelectMany(s => EngineOffenders(s.Text, s.Name)
                .Concat(CliOffenders(s.Text, s.Name))
                .Concat(RequestOffenders(s.Text, s.Name)))
            .ToList();

        // The scan has to see what it polices: the calls this suite makes are dozens, not a handful.
        var engines = sources.Sum(s => Initializer.Matches(s.Text).Count);
        var cli = sources.Sum(s => CliStatement.Matches(s.Text).Count(m => m.Groups["args"].Value.Contains("\"train\"", StringComparison.Ordinal)));
        var requests = sources.Sum(s => TrainRequest.Matches(s.Text).Count(m => Regex.IsMatch(m.Groups["body"].Value, @"\btask\s*=") && m.Groups["body"].Value.Contains("timeLimitSeconds", StringComparison.Ordinal)));
        Assert.True(engines >= 10, $"the scan found {engines} TrainingConfig initializers — it should see the ones the engine suites build");
        Assert.True(cli >= 20, $"the scan found {cli} `mloop train` calls — it should see the ones the command suites make");
        Assert.True(requests >= 5, $"the scan found {requests} POST /train requests with a time budget — it should see the ones the API suite makes");
        Assert.Empty(offenders);
    }

    [Fact]
    public void TheScan_FlagsABudgetOnlyTrainingCall_AndLeavesTheCountedOnesAndTheNonSearchTasks()
    {
        const string EngineBudgeted = "var c = new TrainingConfig { Task = \"regression\", TimeLimitSeconds = 10 };";
        const string EngineCounted = "var c = new TrainingConfig { Task = \"regression\", TimeLimitSeconds = 60, MaxTrials = 10 };";
        const string EngineClustering = "var c = new TrainingConfig { Task = \"clustering\", TimeLimitSeconds = 5 };";
        const string EngineVariableTask = "var c = new TrainingConfig { Task = task,   // no search: clustering\n TimeLimitSeconds = 5 };";
        const string CliBudgeted = "var r = await CliRunner.RunAsync(\"train\", \"d.csv\",\n \"--time\", \"5\");";
        const string CliCounted = "var r = await CliRunner.RunAsync(\"train\", \"d.csv\", \"--time\", \"60\", \"--max-trials\", \"10\");";
        const string CliOtherCommand = "var r = await CliRunner.RunAsync(\"evaluate\", \"exp-001\", \"d.csv\", \"--time\", \"5\");";
        const string CliNoSearch = "var r = await CliRunner.RunAsync(\"train\", \"d.csv\", \"--time\", \"5\");   // no search: clustering";
        const string RequestBudgeted = "await client.PostAsJsonAsync(\"/train\", new { dataFile = d, labelColumn = \"y\", task = \"regression\", timeLimitSeconds = 5 });";
        const string RequestCounted = "await client.PostAsJsonAsync(\"/train\", new { dataFile = d, labelColumn = \"y\", task = \"regression\", timeLimitSeconds = 60, maxTrials = 10 });";
        const string ConfigFixture = "File.WriteAllText(p, Serialize(new { dataFile = d, labelColumn = \"y\", timeLimitSeconds = 10 }));";

        Assert.Single(EngineOffenders(EngineBudgeted, "x.cs"));
        Assert.Empty(EngineOffenders(EngineCounted, "x.cs"));
        Assert.Empty(EngineOffenders(EngineClustering, "x.cs"));
        Assert.Empty(EngineOffenders(EngineVariableTask, "x.cs"));
        Assert.Single(CliOffenders(CliBudgeted, "x.cs"));
        Assert.Empty(CliOffenders(CliCounted, "x.cs"));
        Assert.Empty(CliOffenders(CliOtherCommand, "x.cs"));
        Assert.Empty(CliOffenders(CliNoSearch, "x.cs"));
        Assert.Single(RequestOffenders(RequestBudgeted, "x.cs"));
        Assert.Empty(RequestOffenders(RequestCounted, "x.cs"));
        Assert.Empty(RequestOffenders(ConfigFixture, "x.cs"));
    }
}
