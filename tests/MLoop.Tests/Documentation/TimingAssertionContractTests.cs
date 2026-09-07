using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// An assertion that bounds elapsed time from above must carry the category that keeps it out of CI.
/// </summary>
/// <remarks>
/// <para>
/// A wall-clock upper bound measures the machine as much as the code. On a developer's idle laptop
/// it states something about the implementation; on a shared runner it states something about that
/// runner's scheduling, and the two are indistinguishable from the assertion's text. This suite has
/// long agreed with that — every such bound but one carried <c>Category=Slow</c>, which the CI and
/// release pipelines filter out. The exception was not a decision; it was an omission, and it sat on
/// the path of every publish until a release actually stopped on a timing failure.
/// </para>
/// <para>
/// The rule is narrow on purpose. A <i>lower</i> bound (<c>ElapsedSeconds &gt; 0</c>) asserts that
/// something was measured at all, which no amount of load can falsify, and stays allowed.
/// </para>
/// </remarks>
public class TimingAssertionContractTests
{
    /// <summary>An elapsed duration compared against a literal upper bound.</summary>
    private static readonly Regex WallClockUpperBound = new(
        @"(Elapsed[A-Za-z]*|TotalMilliseconds|TotalSeconds|TotalMinutes)\s*<=?\s*[0-9_]+",
        RegexOptions.Compiled);

    private static readonly Regex SlowCategory = new(
        @"\[\s*Trait\s*\(\s*""Category""\s*,\s*""Slow""\s*\)\s*\]",
        RegexOptions.Compiled);

    private static readonly Regex TestMethod = new(
        @"^\s*(public|private|internal)\s+(async\s+)?(Task|void)\s+\w+\s*\(",
        RegexOptions.Compiled);

    [Fact]
    public void EveryWallClockUpperBoundIsCategorisedSlow()
    {
        var offenders = new List<string>();

        foreach (var file in RepoSourceTree.SourceFiles(
                     excludingFileNamed: nameof(TimingAssertionContractTests)))
        {
            var lines = File.ReadAllLines(file);
            for (var i = 0; i < lines.Length; i++)
            {
                if (!WallClockUpperBound.IsMatch(lines[i]))
                    continue;

                if (!EnclosingTestIsSlow(lines, i))
                    offenders.Add($"{RepoSourceTree.Relative(file)}:{i + 1} — {lines[i].Trim()}");
            }
        }

        Assert.True(
            offenders.Count == 0,
            "These bound an elapsed time from above without [Trait(\"Category\", \"Slow\")], so they "
            + "run in CI and in the release pipeline, where a loaded runner — not the code — decides "
            + "whether they pass. Mark the timing assertion Slow; if the guarantee also has a "
            + "load-independent half, assert that half separately so it keeps running everywhere:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheCheckActuallyCatchesAnUncategorisedBound()
    {
        // Negative control: the shapes this guard must see, and the lower bound it must not.
        string[] uncategorised =
        [
            "    [Fact]",
            "    public void Something()",
            "    {",
            "        Assert.True(stopwatch.ElapsedMilliseconds < 50);",
            "    }",
        ];
        Assert.False(EnclosingTestIsSlow(uncategorised, 3));

        string[] categorised =
        [
            "    [Fact]",
            "    [Trait(\"Category\", \"Slow\")]",
            "    public void Something()",
            "    {",
            "        Assert.True(stopwatch.ElapsedMilliseconds < 50);",
            "    }",
        ];
        Assert.True(EnclosingTestIsSlow(categorised, 4));

        Assert.Matches(WallClockUpperBound, "Assert.True(sw.ElapsedMilliseconds < 2000, \"slow\")");
        Assert.DoesNotMatch(WallClockUpperBound, "Assert.True(t.ElapsedSeconds > 0, \"zero elapsed\")");
    }

    /// <summary>
    /// Walks up from <paramref name="line"/> to the enclosing method signature, then reads the
    /// attribute block directly above it. Text, not syntax: the rule is about what a reader sees
    /// next to the assertion, and the suite writes its attributes one per line.
    /// </summary>
    private static bool EnclosingTestIsSlow(IReadOnlyList<string> lines, int line)
    {
        var signature = line;
        while (signature >= 0 && !TestMethod.IsMatch(lines[signature]))
            signature--;

        if (signature < 0)
            return false;

        for (var i = signature - 1; i >= 0; i--)
        {
            var text = lines[i].Trim();
            if (text.Length == 0 || text.StartsWith("///", StringComparison.Ordinal))
                continue;
            if (!text.StartsWith('['))
                break;
            if (SlowCategory.IsMatch(text))
                return true;
        }

        return false;
    }
}
