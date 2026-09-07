using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A test that drives a command must go through the same dispatch <c>Main</c> uses, not a hand-rolled
/// copy of it.
/// </summary>
/// <remarks>
/// <para>
/// The copy is easy to write — parse the tree, invoke the result — and for a long time every
/// structured-output test had its own. What that omits is everything decided <i>before</i> a command
/// action starts: a command line the parser rejected, and one it accepted that means something the
/// caller did not write. Both are real exits users reach, and neither is reachable through the copy,
/// so each new contract at that layer needed its own test class rather than being covered by the
/// suites that already drive commands.
/// </para>
/// <para>
/// Fixing the ten copies once does not keep them fixed: the next helper is written by copying a
/// neighbour, and nothing about <c>Parse(args).InvokeAsync()</c> looks wrong on its own. This asserts
/// the seam instead of the symptom, so a reintroduced copy fails here rather than silently narrowing
/// what the suite can see.
/// </para>
/// <para>
/// Parsing without invoking stays legitimate — asserting that the command tree accepts or rejects a
/// line is a statement about the tree, not about dispatch — so only the invoking form is refused.
/// </para>
/// </remarks>
public class DispatchSeamContractTests
{
    private static readonly string[] HandRolledDispatch =
    [
        "Parse(args).Invoke()",
        "Parse(args).InvokeAsync()",
    ];

    [Fact]
    public void NoTestDrivesACommandThroughItsOwnDispatch()
    {
        var offenders =
            (from file in TestSourceTree.SourceFiles(
                 excludingFileNamed: nameof(DispatchSeamContractTests))
             from line in File.ReadAllLines(file)
             where HandRolledDispatch.Any(form => line.Contains(form, StringComparison.Ordinal))
             select $"{TestSourceTree.Relative(file)}: {line.Trim()}")
            .ToList();

        Assert.True(
            offenders.Count == 0,
            "These call the command tree's own dispatch instead of " +
            $"{nameof(Program)}.{nameof(Program.ExecuteAsync)}, so they cannot reach the exits that " +
            "happen before a command action starts:" + Environment.NewLine +
            string.Join(Environment.NewLine, offenders));
    }
}
