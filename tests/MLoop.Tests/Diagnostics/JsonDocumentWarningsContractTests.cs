using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Tests.Documentation;

namespace MLoop.Tests.Diagnostics;

/// <summary>
/// A warning raised under <c>--json</c> reaches the document. In a <see cref="JsonOutputScope"/> the
/// warning's line goes to stderr with the narration, which a consumer of the document does not read,
/// so the scope keeps it (<see cref="JsonOutputScope.Warnings"/>) and the command's document carries it
/// as <c>warnings</c>. Before, only <c>train</c>'s event stream received warnings: <c>evaluate</c>,
/// <c>predict</c>, <c>calibrate</c> and <c>threshold</c> dropped every one from their documents.
/// </summary>
[Collection("FileSystem")]
public class JsonDocumentWarningsContractTests
{
    [Fact]
    public void AWarningUnderTheScope_IsKeptForTheDocument_WithoutMarkup()
    {
        Assert.Null(JsonOutputScope.Warnings);
        using (new JsonOutputScope())
        {
            WarningConsole.Warn("[bold]weights[/] differ");
            WarningConsole.Warn("second");
            Assert.Equal(["weights differ", "second"], JsonOutputScope.Warnings);
        }
        Assert.Null(JsonOutputScope.Warnings);
    }

    [Fact]
    public void WithoutAScope_AWarningIsOnlyALine()
    {
        WarningConsole.Warn("nothing keeps this");
        Assert.Null(JsonOutputScope.Warnings);
    }

    /// <summary>
    /// A command that opens the scope and raises a warning itself carries
    /// <see cref="JsonOutputScope.Warnings"/> in its document. (A warning raised inside a helper the
    /// scan cannot see is covered by the same seam — this guards the direct case, where a new warning
    /// in a command would otherwise be the first sign that its document lacks the field.)
    /// </summary>
    [Fact]
    public void EveryCommandThatWarnsUnderTheScope_CarriesTheScopesWarnings()
    {
        var commands = Path.Combine(RepoSourceTree.RepoRoot, "tools", "MLoop.CLI", "Commands");
        var offenders = Directory.EnumerateFiles(commands, "*.cs")
            .Where(f => Offends(File.ReadAllText(f)))
            .Select(Path.GetFileName)
            .ToList();

        Assert.Empty(offenders);
        // The scan sees the commands it polices.
        Assert.Contains(Directory.EnumerateFiles(commands, "*.cs"), f => f.EndsWith("EvaluateCommand.cs") && Polices(File.ReadAllText(f)));
    }

    [Fact]
    public void TheScan_RefusesACommandThatWarnsWithoutTheField_AndPassesOneThatHasIt()
    {
        const string warns = "using var s = jsonOutput ? new JsonOutputScope() : null; WarningConsole.Warn(\"x\");";
        Assert.True(Offends(warns));
        Assert.False(Offends(warns + " var d = new { warnings = JsonOutputScope.Warnings };"));
        Assert.False(Offends("WarningConsole.Warn(\"no json mode here\");"));
    }

    private static bool Polices(string source) =>
        source.Contains("new JsonOutputScope()", StringComparison.Ordinal)
        && source.Contains("WarningConsole.Warn(", StringComparison.Ordinal);

    private static bool Offends(string source) =>
        Polices(source) && !source.Contains("JsonOutputScope.Warnings", StringComparison.Ordinal);
}
