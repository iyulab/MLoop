using System.Text.RegularExpressions;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Which commands the README names, measured against the commands that exist. The README is the page a reader opens first, and
/// a command it never mentions reads as a command that does not exist - eleven of them were missing from its list (drift, explain,
/// threshold, calibrate, status, trigger, logs, feedback, sample, preprocess, new) while the guide documented every one. The example
/// parser in <see cref="ReadmeCommandContractTests"/> checks what the README says; it cannot see what the README leaves out.
/// </summary>
public class ReadmeCoverageContractTests
{
    private static string Readme() => File.ReadAllText(Path.Combine(RepoSourceTree.RepoRoot, "README.md"));

    private static IEnumerable<string> ExistingCommands() =>
        Program.BuildRootCommand().Subcommands.Where(c => !c.Hidden).Select(c => c.Name);

    /// <summary>The commands among <paramref name="commands"/> that no <c>mloop &lt;name&gt;</c> in <paramref name="readme"/> names.</summary>
    internal static List<string> NotNamed(string readme, IEnumerable<string> commands) =>
        [.. commands.Where(c => !Regex.IsMatch(readme, @"\bmloop\s+" + Regex.Escape(c) + @"(?![\w-])")).Order(StringComparer.Ordinal)];

    [Fact]
    public void EveryCommandIsNamedInTheReadme()
    {
        var existing = ExistingCommands().ToList();
        Assert.Contains("train", existing);
        Assert.Empty(NotNamed(Readme(), ["train"]));

        var missing = NotNamed(Readme(), existing);

        Assert.True(missing.Count == 0,
            "These commands exist and the README never names them (add a line to Core Commands):\n" + string.Join('\n', missing));
    }

    [Fact]
    public void TheScan_SeesACommandTheReadmeLeavesOut()
    {
        Assert.Equal(["no-such-command"], NotNamed(Readme(), ["train", "no-such-command"]));
        Assert.Equal(["drift"], NotNamed("mloop train data.csv\nmloop driftless", ["train", "drift"]));
    }
}
