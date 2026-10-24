using System.Text.RegularExpressions;
using MLoop.Core.Diagnostics;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A warning MLoop.Core writes to a caller's string sink is marked by <see cref="CoreMessage"/>, so the CLI can
/// put it in a <c>--json</c> document's <c>warnings</c> and a run's <c>warning</c> events — before, the marker was
/// spelled in twenty-three places, and the CLI printed every one as narration (<c>mloop info --json</c> left a
/// multi-row header warning out of a document that reported success).
/// </summary>
public class CoreWarningMarkerContractTests
{
    private static readonly Regex SpelledMarker = new(@"""\[Warning\] ", RegexOptions.Compiled);

    [Fact]
    public void CoreWritesTheWarningMarkerOnlyThroughCoreMessage()
    {
        var offenders = RepoSourceTree.ProductionSourceFiles("src")
            .Where(f => !f.EndsWith("CoreMessage.cs", StringComparison.Ordinal))
            .Where(f => SpelledMarker.IsMatch(File.ReadAllText(f)))
            .Select(f => Path.GetRelativePath(RepoSourceTree.RepoRoot, f))
            .ToList();

        Assert.True(offenders.Count == 0, "Mark a warning with CoreMessage.Warning(…): " + string.Join(", ", offenders));
        Assert.Contains(RepoSourceTree.ProductionSourceFiles("src"), f => File.ReadAllText(f).Contains("CoreMessage.Warning(", StringComparison.Ordinal));
    }

    // The CLI's own warnings go through WarningConsole, which puts them in --json documents and run events; a
    // spelled marker printed with Console reached stdout ahead of `validate --json`'s document.
    [Fact]
    public void TheCliSpellsNoWarningMarker()
    {
        var offenders = RepoSourceTree.ProductionSourceFiles("tools")
            .SelectMany(f => File.ReadAllLines(f).Select((line, i) => (f, line, i)))
            .Where(x => !x.line.TrimStart().StartsWith("//", StringComparison.Ordinal) && SpelledMarker.IsMatch(x.line))
            .Select(x => $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, x.f)}:{x.i + 1}")
            .ToList();

        Assert.True(offenders.Count == 0, "Warn with WarningConsole.Warn(…): " + string.Join(", ", offenders));
    }

    [Theory]
    [InlineData("write($\"[Warning] Sparse column '{name}' excluded\");", true)]
    [InlineData("_log(\"[Warning] No object classes were found.\");", true)]
    [InlineData("write(CoreMessage.Warning($\"Sparse column '{name}' excluded\"));", false)]
    public void TheScan_SeesASpelledMarker(string source, bool spelled) =>
        Assert.Equal(spelled, SpelledMarker.IsMatch(source));

    [Fact]
    public void AMarkedLine_ReadsBackAsItsText_AndNarrationIsNotAWarning()
    {
        Assert.True(CoreMessage.TryReadWarning(CoreMessage.Warning("x dropped"), out var text));
        Assert.Equal("x dropped", text);
        Assert.False(CoreMessage.TryReadWarning("[Info] Removed index column(s): id", out _));
    }
}
