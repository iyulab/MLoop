using System.Text.RegularExpressions;
using MLoop.Core.Storage;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The project-root names have one home. Before they did, <c>"datasets"</c> sat in a private
/// constant plus seven literals across the CLI and the API, and <c>"predictions"</c> had no
/// constant at all — which is how three commands came to write prediction files under three naming
/// conventions while the one command that reads them back knew only the first.
/// </summary>
public class ProjectLayoutContractTests
{
    /// <summary>
    /// A path-shaped literal: the name in quotes, not the word appearing in prose or in a doc
    /// comment. <c>"datasets/train.csv"</c> counts; <c>// the datasets directory</c> does not.
    /// </summary>
    private static readonly Regex DatasetsLiteral =
        new("\"datasets(?:[/\\\\][^\"]*)?\"", RegexOptions.Compiled);

    private static readonly Regex PredictionsLiteral =
        new("\"predictions(?:[/\\\\][^\"]*)?\"", RegexOptions.Compiled);

    /// <summary>
    /// The prediction file conventions, spelled as an interpolation rather than asked for. This is
    /// the shape that let a writer add a fourth convention without the reader hearing about it.
    /// </summary>
    private static readonly Regex PredictionNameLiteral =
        new(@"-(?:predictions|detections|forecast)-", RegexOptions.Compiled);

    /// <summary>
    /// Code only. A comment that talks about the convention is not a use of it, and a doc comment
    /// naming a parameter <c>predictions</c> is not one either — both were reported as offenders by
    /// the first version of this scan.
    /// </summary>
    private static string CodeOf(string path) =>
        string.Join('\n', File.ReadAllLines(path).Where(line =>
        {
            var t = line.TrimStart();
            return !t.StartsWith("//", StringComparison.Ordinal)
                && !t.StartsWith("*", StringComparison.Ordinal)
                && !t.StartsWith("/*", StringComparison.Ordinal);
        }));

    private static IReadOnlyList<string> Scanned()
    {
        var files = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
            .Concat(RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.API")))
            .Concat(RepoSourceTree.ProductionSourceFiles(Path.Combine("src", "MLoop.Core")))
            .Where(f => !Path.GetFileName(f).Equals(nameof(ProjectLayout) + ".cs", StringComparison.Ordinal))
            .ToList();

        // The scan reaches the files it is about. Without this, a directory that stopped resolving
        // would empty the set and report a clean tree.
        Assert.Contains(files, f => Path.GetFileName(f) == "StatusCommand.cs");
        Assert.Contains(files, f => Path.GetFileName(f) == "PredictCommand.cs");
        Assert.Contains(files, f => Path.GetFileName(f) == "Program.cs");

        return files;
    }

    [Fact]
    public void NoProductCodeSpellsTheProjectRootDirectoriesItself()
    {
        var offenders = Scanned()
            .Where(f =>
            {
                var source = CodeOf(f);
                return DatasetsLiteral.IsMatch(source) || PredictionsLiteral.IsMatch(source);
            })
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "The project-root directory names live in " + nameof(ProjectLayout) + " so the API and the "
            + "CLI cannot drift apart about them:\n" + string.Join('\n', offenders));
    }

    [Fact]
    public void NoProductCodeSpellsAPredictionFileConventionItself()
    {
        var offenders = Scanned()
            .Where(f => PredictionNameLiteral.IsMatch(CodeOf(f)))
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "A prediction file's name is asked for (" + nameof(ProjectLayout.PredictionFileName) + ") or "
            + "searched for (" + nameof(ProjectLayout.PredictionSearchPatterns) + "), never spelled — that is "
            + "what keeps every writer's output findable by the one reader:\n" + string.Join('\n', offenders));
    }

    /// <summary>
    /// Companion to both scans. A pattern that has stopped matching reports a clean tree, which is
    /// the same output as a clean tree — and the more confident of the two.
    /// </summary>
    [Fact]
    public void TheScansSeeTheShapesTheyForbidAndAcceptTheOnesTheyWant()
    {
        Assert.Matches(DatasetsLiteral, """var d = Path.Combine(root, "datasets");""");
        Assert.Matches(DatasetsLiteral, """var p = "datasets/train.csv";""");
        Assert.DoesNotMatch(DatasetsLiteral, """var d = Path.Combine(root, ProjectLayout.DatasetsDirectory);""");
        // Prose about the convention is not a use of it.
        Assert.DoesNotMatch(DatasetsLiteral, "// drop your csv in datasets/ and train");

        Assert.Matches(PredictionsLiteral, """var d = Path.Combine(root, "predictions");""");
        Assert.DoesNotMatch(PredictionsLiteral, """var d = Path.Combine(root, ProjectLayout.PredictionsDirectory);""");

        Assert.Matches(PredictionNameLiteral, """$"{model}-predictions-{stamp}.csv" """);
        Assert.Matches(PredictionNameLiteral, """$"{model}-detections-{stamp}.json" """);
        Assert.Matches(PredictionNameLiteral, """var pattern = $"{model}-forecast-*.csv";""");
        Assert.DoesNotMatch(PredictionNameLiteral,
            """ProjectLayout.PredictionFileName(model, ProjectLayout.PredictionKind.Rows, at)""");
    }

    /// <summary>
    /// The comment filter is load-bearing: without it this scan reported four offenders whose only
    /// mention of a name was in prose, including the comment explaining the very defect it guards.
    /// </summary>
    [Fact]
    public void TheCommentFilterKeepsCodeAndDropsProse()
    {
        var file = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(file,
            [
                "// a comment about \"datasets\" and -predictions- files",
                "/// <param name=\"predictions\">scored rows</param>",
                " * continuation of a block comment about \"predictions\"",
                "var real = Path.Combine(root, \"datasets\");",
            ]);

            var code = CodeOf(file);

            Assert.Single(code.Split('\n'));
            Assert.Contains("var real", code);
            Assert.Matches(DatasetsLiteral, code);
            Assert.DoesNotMatch(PredictionNameLiteral, code);
        }
        finally
        {
            File.Delete(file);
        }
    }
}
