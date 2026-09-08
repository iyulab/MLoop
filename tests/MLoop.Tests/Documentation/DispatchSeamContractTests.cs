using System.Text.RegularExpressions;
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
/// <para>
/// The match is on the <i>shape</i> rather than on two literal strings. The first version compared
/// against <c>"Parse(args).Invoke()"</c> and <c>"Parse(args).InvokeAsync()"</c> exactly, so a copy
/// that named its variable anything but <c>args</c>, or broke the chain across two lines the way a
/// formatter does, walked straight past it — a guard that reports "no offenders" because it cannot
/// see them is worse than no guard, since it also reports confidence.
/// <c>TheCheckRecognisesTheFormsItForbids</c> is what keeps that from being true again.
/// </para>
/// </remarks>
public class DispatchSeamContractTests
{
    /// <summary>
    /// <c>.Parse(…)</c> chained into <c>.Invoke…</c>, whatever the argument is called and however
    /// the chain is wrapped. Matched over the whole file rather than line by line, because the
    /// wrapped form is the one a formatter produces.
    /// </summary>
    private static readonly Regex HandRolledDispatch =
        new(@"\.Parse\s*\([^()]*\)\s*\.\s*Invoke", RegexOptions.Compiled);

    [Fact]
    public void NoTestDrivesACommandThroughItsOwnDispatch()
    {
        var offenders =
            (from file in RepoSourceTree.SourceFiles(
                 excludingFileNamed: nameof(DispatchSeamContractTests))
             let text = File.ReadAllText(file)
             from match in HandRolledDispatch.Matches(text).Cast<Match>()
             select $"{RepoSourceTree.Relative(file)}: "
                  + Regex.Replace(match.Value, @"\s+", " ").Trim())
            .ToList();

        Assert.True(
            offenders.Count == 0,
            "These call the command tree's own dispatch instead of " +
            $"{nameof(Program)}.{nameof(Program.ExecuteAsync)}, so they cannot reach the exits that " +
            "happen before a command action starts:" + Environment.NewLine +
            string.Join(Environment.NewLine, offenders));
    }

    /// <summary>
    /// The guard can see what it forbids — including the two shapes its first version could not.
    /// A scan that has stopped matching passes silently, and passing is what it is trusted for.
    /// </summary>
    [Theory]
    [InlineData("var exit = Program.BuildRootCommand().Parse(args).Invoke();")]
    [InlineData("var exit = await root.Parse(args).InvokeAsync();")]
    [InlineData("var exit = await root.Parse(commandLine).InvokeAsync();")]
    [InlineData("var exit = root\n            .Parse(argv)\n            .Invoke();")]
    public void TheCheckRecognisesTheFormsItForbids(string source)
    {
        Assert.Matches(HandRolledDispatch, source);
    }

    /// <summary>
    /// And does not refuse the legitimate forms: parsing to assert on the tree, and dispatching
    /// through the seam every command actually goes through.
    /// </summary>
    [Theory]
    [InlineData("var result = Program.BuildRootCommand().Parse(args);")]
    [InlineData("Assert.Single(Program.BuildRootCommand().Parse(\"list --json\").Errors);")]
    [InlineData("var exit = await Program.ExecuteAsync(Program.BuildRootCommand(), args);")]
    public void TheCheckLeavesTheLegitimateFormsAlone(string source)
    {
        Assert.DoesNotMatch(HandRolledDispatch, source);
    }
}
