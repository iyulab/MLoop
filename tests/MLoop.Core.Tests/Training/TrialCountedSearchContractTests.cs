using System.Text.RegularExpressions;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A test that runs a trainer search stops it on a trial count, with the clock only as its ceiling.
/// </summary>
/// <remarks>
/// A search ended by a time budget does a different amount of work on a busy machine, and what it does is what
/// the test then sees: on the same day a trial recorded without its metric on one runner and a failed trial on
/// another, in suites that asserted nothing about time. A trial-count limit makes the work the same wherever it
/// runs. The suites that measure the budget itself are in <c>MLoop.Search.Tests</c>.
/// <para>
/// This reads the test sources: a <c>TrainingConfig</c> that sets <c>TimeLimitSeconds</c> and not
/// <c>MaxTrials</c> is flagged unless its task is not a trainer search (a literal task that the search
/// does not serve).
/// </para>
/// </remarks>
public class TrialCountedSearchContractTests
{
    private static readonly Regex Initializer = new(@"new TrainingConfig\s*\{(?<body>[^{}]*)\}", RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly string[] NotATrainerSearch =
    [
        "clustering", "anomaly-detection", "ranking", "recommendation", "forecasting", "time-series-anomaly",
        "image-classification", "object-detection", "image-anomaly", "text-classification", "sentence-similarity", "ner", "question-answering",
        "text-generation",   // a task no module serves: the test is of the refusal
    ];

    // Found from the build output, as the other source-reading guards do: [CallerFilePath] is rewritten to a
    // placeholder by a deterministic (CI) build, so it names no directory there.
    private static string TestsRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "MLoop.slnx")))
            dir = dir.Parent;
        return dir is null
            ? throw new InvalidOperationException($"Could not locate MLoop.slnx by walking up from {AppContext.BaseDirectory}.")
            : Path.Combine(dir.FullName, "tests", "MLoop.Core.Tests");
    }

    internal static IEnumerable<string> Offenders(string source, string name) =>
        Initializer.Matches(source)
            .Select(m => m.Groups["body"].Value)
            .Where(body => body.Contains("TimeLimitSeconds", StringComparison.Ordinal) && !body.Contains("MaxTrials", StringComparison.Ordinal))
            .Where(body => !NotATrainerSearch.Any(task => body.Contains($"Task = \"{task}\"", StringComparison.Ordinal)))
            .Select(body => $"{name}: {Regex.Replace(body.Trim(), @"\s+", " ")}");

    [Fact]
    public void ASearchTestStopsOnATrialCount()
    {
        var files = Directory.EnumerateFiles(TestsRoot(), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
            .Where(f => Path.GetFileName(f) != nameof(TrialCountedSearchContractTests) + ".cs")   // its own examples are the shapes it flags
            .ToList();

        var offenders = files.SelectMany(f => Offenders(File.ReadAllText(f), Path.GetRelativePath(TestsRoot(), f))).ToList();
        var seen = files.Sum(f => Initializer.Matches(File.ReadAllText(f)).Count);

        Assert.True(seen >= 10, $"the scan found {seen} TrainingConfig initializers — it should see the dozen this suite builds");
        Assert.Empty(offenders);
    }

    [Fact]
    public void TheScan_FlagsABudgetOnlySearch_AndLeavesTheCountedOneAndANonSearchTask()
    {
        const string Budgeted = "var c = new TrainingConfig { Task = \"regression\", TimeLimitSeconds = 10 };";
        const string Counted = "var c = new TrainingConfig { Task = \"regression\", TimeLimitSeconds = 60, MaxTrials = 10 };";
        const string Clustering = "var c = new TrainingConfig { Task = \"clustering\", TimeLimitSeconds = 5 };";

        Assert.Single(Offenders(Budgeted, "x.cs"));
        Assert.Empty(Offenders(Counted, "x.cs"));
        Assert.Empty(Offenders(Clustering, "x.cs"));
    }
}
