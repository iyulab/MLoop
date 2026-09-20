using System.Text.RegularExpressions;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Which commands the guide explains, measured against the commands that exist.
/// </summary>
/// <remarks>
/// <para>The guide explained fewer than half of them. That is not a gap a reader can see — a
/// command absent from the guide reads as a command that does not exist, and nothing in the build
/// said otherwise. It surfaced only when a cycle set out to check the guide's option blocks
/// against the code and found that half of them had no block to check. The first count of how many
/// was itself wrong, taken by grepping command constructors: it swept in subcommands and missed
/// three real commands, which this guard reported on its first run. A number measured by the thing
/// that enforces it does not drift.</para>
/// <para>So the debt is written down rather than left implicit. A command missing from the guide
/// must be named in <see cref="Undocumented"/>; adding a new command without a section fails here,
/// and documenting one fails here too until its name is removed from the list. Both directions are
/// deliberate: the list only moves when someone means to move it.</para>
/// </remarks>
public class GuideCoverageContractTests
{
    /// <summary>
    /// Commands with no section in the guide yet. **This list may only shrink.** Ordered as a
    /// reader would meet them, which is also the order to write them in.
    /// </summary>
    private static readonly HashSet<string> Undocumented = new(StringComparer.Ordinal)
    {
        "detect",     // deliberately last of the three in flight: an unmerged branch changes its
                      // options, and documenting the current ones would be stale on merge
        "feedback",
        "sample",
        "trigger",
        "prep",
        "features",
        "runtime",
        "update",

        // Found by this guard on its first run: the count that motivated it came from grepping
        // command constructors, which swept in subcommands and missed these. (`token`, the third,
        // is documented now.)
        "new",          // scaffolds hooks/metrics/scripts; the Extensibility section shows the
                        // files it writes but never names the command that writes them
        "preprocess",   // runs preprocessing scripts; distinct from `prep`, which is the pipeline
    };

    private static readonly Regex GuideSection = new(@"^### `mloop ([a-z-]+)`", RegexOptions.Multiline);

    private static IReadOnlySet<string> DocumentedCommands()
    {
        var guide = File.ReadAllText(Path.Combine(RepoSourceTree.RepoRoot, "docs", "GUIDE.md"));
        return GuideSection.Matches(guide).Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
    }

    private static IReadOnlySet<string> ExistingCommands() =>
        Program.BuildRootCommand().Subcommands
            .Where(c => !c.Hidden)
            .Select(c => c.Name)
            .ToHashSet(StringComparer.Ordinal);

    [Fact]
    public void EveryCommandIsEitherDocumentedOrListedAsDebt()
    {
        var existing = ExistingCommands();
        var documented = DocumentedCommands();

        // The scan reaches a real command tree and a real guide; an empty either side would make
        // the comparison below pass by saying nothing.
        Assert.Contains("train", existing);
        Assert.Contains("train", documented);

        var missing = existing.Except(documented).Except(Undocumented).Order(StringComparer.Ordinal).ToList();

        Assert.True(missing.Count == 0,
            "These commands exist and the guide does not explain them. Write the section, or add "
            + "the name to " + nameof(Undocumented) + " with the reason:\n"
            + string.Join('\n', missing));
    }

    /// <summary>
    /// The debt list is debt, not a permanent exemption: a name that has been documented has to
    /// leave it, or the list stops describing anything.
    /// </summary>
    [Fact]
    public void TheDebtListHoldsOnlyCommandsThatAreStillMissing()
    {
        var documented = DocumentedCommands();

        var stale = Undocumented.Where(documented.Contains).Order(StringComparer.Ordinal).ToList();

        Assert.True(stale.Count == 0,
            "These are documented now — remove them from " + nameof(Undocumented) + ":\n"
            + string.Join('\n', stale));
    }

    /// <summary>And a name that is not a command at all does not belong there either.</summary>
    [Fact]
    public void TheDebtListNamesRealCommands()
    {
        var existing = ExistingCommands();

        var ghosts = Undocumented.Where(c => !existing.Contains(c)).Order(StringComparer.Ordinal).ToList();

        Assert.True(ghosts.Count == 0,
            "These are in " + nameof(Undocumented) + " but are not commands:\n" + string.Join('\n', ghosts));
    }
}
