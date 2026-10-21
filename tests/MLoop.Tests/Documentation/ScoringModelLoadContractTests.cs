using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Product code loads a model to score with through <c>ScoringModel</c> only. Loading is three things
/// — the task's native runtime, the decision threshold beside the model, the σ-model sibling — and
/// the paths that each did their own subset gave different answers to the same row.
/// </summary>
public class ScoringModelLoadContractTests
{
    // `.Model.Load(` on any MLContext — the call that deserializes a saved model.
    private static readonly Regex DirectLoad = new(@"\.Model\.Load\(", RegexOptions.Compiled);

    private static bool IsDocumentation(string line) => line.TrimStart().StartsWith("//", StringComparison.Ordinal);

    private static IEnumerable<string> Offenders(IEnumerable<(string File, string[] Lines)> sources) =>
        from source in sources
        where Path.GetFileName(source.File) != "ScoringModel.cs"
        from line in source.Lines.Select((Text, Number) => (Text, Number))
        where !IsDocumentation(line.Text) && DirectLoad.IsMatch(line.Text)
        select $"{source.File}:{line.Number + 1}";

    [Fact]
    public void ProductCodeLoadsModelsThroughScoringModelOnly()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Lines: File.ReadAllLines(f))).ToList();
        // It read the tree: the authority is among what it scanned, and it does load a model.
        Assert.Contains(sources, s => Path.GetFileName(s.File) == "ScoringModel.cs" && s.Lines.Any(DirectLoad.IsMatch));
        var offenders = Offenders(sources).ToList();

        Assert.True(offenders.Count == 0,
            "Load a model to score with through MLoop.Core.Models.ScoringModel (runtime · decision threshold · " +
            "σ-model sibling), not MLContext.Model.Load:" + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: it finds the shape it forbids, and lets the authority itself and a comment pass.</summary>
    [Fact]
    public void TheScanSeesADirectLoad()
    {
        var forbidden = ("tools/X.cs", new[] { "        var model = mlContext.Model.Load(path, out _);" });
        var authority = ("src/MLoop.Core/Models/ScoringModel.cs", new[] { "        var model = mlContext.Model.Load(modelPath, out inputSchema);" });
        var comment = ("tools/Y.cs", new[] { "        // never mlContext.Model.Load(path, out _) here" });

        Assert.Equal(["tools/X.cs:1"], Offenders([forbidden, authority, comment]));
    }
}
