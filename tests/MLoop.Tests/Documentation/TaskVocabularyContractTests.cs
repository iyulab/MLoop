using System.Text.RegularExpressions;
using MLoop.CLI;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The ML task vocabulary is written once. A command that accepts a task, rejects a task, or names
/// the accepted tasks in its help text reads <see cref="TaskTypes"/> rather than carrying its own
/// copy of the list.
/// </summary>
/// <remarks>
/// <para>
/// Three copies existed before this guard: <c>init</c>'s scaffolding allowlist, <c>validate</c>'s
/// config allowlist, and <c>train</c>'s <c>--task</c> help text. They happened to agree, which is
/// the state that makes duplication look harmless — the cost only appears at the next edit, and it
/// appears asymmetrically. A task added to <c>init</c> but not <c>validate</c> scaffolds a project
/// that fails its own validation; one added to both but not the help text is supported and
/// undiscoverable. Neither shows up as a broken build.
/// </para>
/// <para>
/// This is the same convergence <c>ExperimentLayout</c> and <c>TaskMetadata</c> made for filesystem
/// layout and metric knowledge, applied to the third cross-assembly literal set. The guard is a
/// source-text scan rather than an API assertion because the failure it prevents is <i>writing a
/// fourth copy</i>, which no amount of correct behavior in the existing three would catch.
/// </para>
/// </remarks>
public class TaskVocabularyContractTests
{
    /// <summary>
    /// How much of the vocabulary a file may name in list form before that listing counts as a
    /// second copy of it: more than half. The bar is a majority rather than a handful because the
    /// two shapes mean different things. A file listing most of the set is <i>stating which tasks
    /// exist</i>, and that statement belongs in one place. A file listing a few is naming a
    /// <i>subset with its own meaning</i> — the tasks one module handles, the tasks one runtime
    /// serves — and those are governed by the guard for that relationship
    /// (<c>DeepLearningRuntimeCoverageTests</c>), not by this one.
    /// </summary>
    private static readonly int ListedTaskThreshold = TaskTypes.All.Count / 2 + 1;

    private const string TaskAlternation =
        @"binary-classification|multiclass-classification|regression|anomaly-detection"
      + @"|clustering|ranking|forecasting|time-series-anomaly|recommendation"
      + @"|image-classification|object-detection|text-classification|sentence-similarity"
      + @"|ner|question-answering";

    /// <summary>
    /// A line that is nothing but task literals and separators — the shape a collection initializer
    /// takes, one element per line or several. This is deliberately not "any line mentioning a
    /// task": a <c>switch</c> arm and an <c>Equals</c> comparand also name a task, and they are
    /// per-task <i>behavior</i>, which has to name the task it handles and cannot read a list
    /// instead. Only an enumeration of the set is a copy of the set.
    /// </summary>
    private static readonly Regex ListingLine =
        new($@"^\s*(?:""(?:{TaskAlternation})""\s*,?\s*)+$", RegexOptions.Compiled);

    /// <summary>
    /// A single string that spells out the vocabulary — the help-text shape, where the drift is
    /// invisible because nothing but a reader compares it to the real list.
    /// </summary>
    private static readonly Regex TaskInText = new(TaskAlternation, RegexOptions.Compiled);

    private static readonly Regex StringLiteral =
        new("\"[^\"\r\n]*\"", RegexOptions.Compiled);

    /// <summary>
    /// Help text names the whole vocabulary. The failure this catches is a task that the trainer
    /// accepts and no user can find out about — the option's description is the only place the
    /// accepted values are enumerated for someone who has not read the config schema.
    /// </summary>
    [Fact]
    public void TrainHelpNamesEveryAcceptedTask()
    {
        var described = Program.BuildRootCommand()
            .Subcommands.Single(c => c.Name == "train")
            .Options.Single(o => o.Name == "--task")
            .Description ?? string.Empty;

        var missing = TaskTypes.All.Where(t => !described.Contains(t, StringComparison.Ordinal));

        Assert.Empty(missing);
    }

    /// <summary>
    /// A task with a canonical metric must be a task the vocabulary admits. The reverse does not
    /// hold — object detection and the NLP tasks have no thresholded scalar primary and are absent
    /// from <see cref="TaskMetadata"/> by design.
    /// </summary>
    [Fact]
    public void MetricKnowledgeCoversOnlyTasksTheVocabularyAdmits()
    {
        var scored = TaskTypes.All.Where(t => TaskMetadata.PrimaryMetric(t) != null).ToArray();

        Assert.NotEmpty(scored);
        Assert.All(scored, t => Assert.True(TaskTypes.IsValid(t)));
    }

    /// <summary>
    /// No production source file re-lists the vocabulary. Naming a task or two in a comment,
    /// template or example is normal; naming most of the set is a copy that will drift.
    /// </summary>
    [Fact]
    public void NoProductionFileRelistsTheVocabulary()
    {
        var authority = Path.GetFullPath(
            Path.Combine(RepoSourceTree.RepoRoot, "src", "MLoop.Core", "Models", "TaskTypes.cs"));

        var offenders =
            (from file in RepoSourceTree.ProductionSourceFiles("src", "tools")
             where !Path.GetFullPath(file).Equals(authority, StringComparison.OrdinalIgnoreCase)
             let listed = TasksNamedInListForm(File.ReadAllText(file))
             where listed.Length >= ListedTaskThreshold
             select $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)} — lists "
                  + $"{listed.Length} task types ({string.Join(", ", listed)}); read "
                  + "MLoop.Core.Models.TaskTypes instead of re-listing them")
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "The task vocabulary lives in MLoop.Core.Models.TaskTypes:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>
    /// The distinct tasks a file names as a <i>list</i> — as elements of a collection initializer,
    /// or spelled out inside one string. Tasks named as switch arms or comparands are not counted:
    /// those are behavior for a task, not a statement about which tasks exist.
    /// </summary>
    private static string[] TasksNamedInListForm(string source)
    {
        var fromCollections = source
            .Split('\n')
            .Where(line => ListingLine.IsMatch(line.TrimEnd('\r')))
            .SelectMany(line => TaskInText.Matches(line).Select(m => m.Value));

        var fromStrings = StringLiteral.Matches(source)
            .Select(m => m.Value)
            .Where(literal => TaskInText.Matches(literal).Select(m => m.Value)
                                        .Distinct().Count() >= ListedTaskThreshold)
            .SelectMany(literal => TaskInText.Matches(literal).Select(m => m.Value));

        return [.. fromCollections.Concat(fromStrings).Distinct().Order()];
    }
}
