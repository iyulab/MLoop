using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Nothing this repository publishes carries a token that only means something inside the
/// workspace it was written in — an internal work-item id, a path off one machine's disk, a
/// pointer into a private notes tree.
/// </summary>
/// <remarks>
/// <para>
/// This repository is public. A reader who opens the changelog is not on the team, has no
/// tracker to look <c>F-27</c> up in, and cannot follow a link into a directory that exists only
/// on a maintainer's machine — so the token spends the reader's attention and returns nothing.
/// It also leaks the shape of a private workspace, which is the part that does not become
/// harmless with time.
/// </para>
/// <para>
/// The rule is enforced here rather than by reading carefully, because reading carefully has
/// already been tried and measured: two separate passes declared the tree clean, and a drive-letter
/// path survived the first while a hundred-odd work-item ids survived both. A person scanning for
/// something that looks ordinary in every other context does not see it; a regex does.
/// </para>
/// <para>
/// What the guard does <em>not</em> do is decide what should replace a token it finds. Deleting a
/// bare cross-reference is usually right, but a marker that ties several call sites together needs
/// a phrase that says what the tie is. That judgment stays with the author — the guard only refuses
/// to let the decision be skipped.
/// </para>
/// </remarks>
public class InternalTokenContractTests
{
    /// <summary>
    /// Prefixes that look like an internal id but are public vocabulary: character encodings and
    /// standards bodies, this repository's own published ADR numbering, and <c>N</c> as the count
    /// in a formula ("Vector&lt;Single, N-1&gt;"). The experiment ids MLoop mints for users
    /// (<c>exp-001</c>) need no entry — they are lower-case, and the shape below matches only an
    /// upper-case prefix.
    /// </summary>
    private static readonly HashSet<string> PublicPrefixes =
        new(StringComparer.Ordinal) { "UTF", "ISO", "CP", "CVE", "RFC", "ADR", "N" };

    /// <summary>
    /// An internal work-item id: a short upper-case prefix (or the literal <c>cycle</c>) and a
    /// number. Upper-case is the whole discrimination — <c>rule-1</c> and <c>q-2</c> are fixture
    /// data and <c>Latin-1</c> is an encoding, and none of them is a ticket anyone could look up.
    /// Shape-matched rather than enumerated, because the prefixes in use are not a closed
    /// set — <c>F</c>, <c>BUG</c>, <c>TD</c>, <c>PRED</c>, <c>EVAL</c>, <c>IMP</c>, <c>MLOOP</c> and
    /// <c>cycle</c> were all in the tree at once, and a list would have to grow every time someone
    /// invents a prefix. The allow-list above carries the exceptions instead, which is the smaller
    /// and more stable of the two lists.
    /// </summary>
    private static readonly Regex WorkItemId =
        new(@"\b(?<prefix>[A-Z]{1,5}|cycle)-\d+\b", RegexOptions.Compiled);

    /// <summary>
    /// Internal units of work whose name is an ordinary English word, so the shape above cannot see
    /// them: a development cycle, a sprint. Enumerated rather than shape-matched for the opposite
    /// reason the ids above are shape-matched — there are few of these and they do not multiply,
    /// while an "ordinary word followed by a number" pattern would flag half the prose in the tree.
    /// </summary>
    private static readonly Regex InternalWorkUnit =
        new(@"\b(?:[Cc]ycle|[Ss]print)[- ]\d+\b", RegexOptions.Compiled);

    /// <summary>
    /// An absolute path on someone's machine. Any drive letter, not just the one that happened to
    /// leak: the point is that the path is unreachable for every reader, not which volume it names.
    /// </summary>
    /// <remarks>
    /// Applied to prose only. In documentation a drive path is a place someone actually went and
    /// pasted, which is exactly the leak; in C# it is the input a test is about — the tip for a
    /// missing path has to be given a Windows path to have anything to assert on — and forbidding
    /// it there would be forbidding the test rather than the leak.
    /// </remarks>
    private static readonly Regex DriveLetterPath = new(@"\b[A-Za-z]:\\", RegexOptions.Compiled);

    /// <summary>
    /// Names of the private working tree this package is developed in. The bare word "roadmap" is
    /// deliberately absent — this repository publishes a <c>ROADMAP.md</c> of its own, and the
    /// private one is already caught by the directory that contains it.
    /// </summary>
    private static readonly Regex PrivateWorkspaceRef =
        new(@"\b(claudedocs|HANDOFF)\b", RegexOptions.Compiled);

    [Fact]
    public void TheScanReachesAFileThatIsNotAddedYet()
    {
        // Companion to the scan below. The set it walks comes from git, and asking git only for
        // tracked files would exempt the file most likely to be carrying a fresh violation: the one
        // just written. A guard that cannot see a new file reports a clean tree in exactly the voice
        // of a clean tree, and the omission surfaces only in the commit that ends the exemption.
        var probe = Path.Combine(
            Path.GetDirectoryName(typeof(InternalTokenContractTests).Assembly.Location)!,
            "..", "..", "..", "Documentation", $"NotAddedProbe-{Guid.NewGuid():N}.md");
        probe = Path.GetFullPath(probe);

        try
        {
            File.WriteAllText(probe, "a file the working tree has and the index does not");

            Assert.Contains(probe, RepoSourceTree.PublishedTextFiles());
        }
        finally
        {
            File.Delete(probe);
        }
    }

    [Fact]
    public void PublishedTextCarriesNoInternalToken()
    {
        var findings = new List<string>();

        foreach (var file in RepoSourceTree.PublishedTextFiles())
        {
            var relative = RepoSourceTree.RelativeToRoot(file);
            if (relative.EndsWith(nameof(InternalTokenContractTests) + ".cs", StringComparison.Ordinal))
                continue; // the guard names the shapes it forbids

            var isProse = relative.EndsWith(".md", StringComparison.OrdinalIgnoreCase);
            var lines = File.ReadAllLines(file);
            for (var i = 0; i < lines.Length; i++)
            {
                foreach (var token in InternalTokensIn(lines[i], isProse))
                    findings.Add($"{relative}:{i + 1}  {token}");
            }
        }

        Assert.True(findings.Count == 0,
            "Published text may not carry tokens that only mean something inside the private " +
            "workspace. Delete a bare cross-reference; replace a marker that ties call sites " +
            "together with a phrase that says what the tie is; rewrite a machine path as a " +
            "placeholder.\n" + string.Join('\n', findings));
    }

    /// <summary>
    /// The guard can still see. Each shape it forbids is fed a line that contains one, and each
    /// near-miss it must tolerate is fed a line that contains that.
    /// </summary>
    /// <remarks>
    /// A guard whose patterns match nothing passes, and passing is what it looks like when it is
    /// working — so the failure is invisible in exactly the situation the guard exists for. That
    /// happened here: one pattern was written with a control character where an escape was
    /// intended, matched nothing, and reported a tree clean that had two dozen violations in it.
    /// This test is the difference between "found no tokens" and "cannot find tokens".
    /// </remarks>
    [Theory]
    [InlineData("closed the F-27 drift", true, "work-item id")]
    [InlineData("see BUG-48 for the shape", true, "work-item id with a longer prefix")]
    [InlineData("fixed in cycle-93", true, "cycle id")]
    [InlineData("found in the sweep (Cycle 82)", true, "cycle written as a word and a number")]
    [InlineData("found during sprint-35", true, "sprint id")]
    [InlineData(@"open D:\data\project", true, "absolute path on one machine")]
    [InlineData("tracked in claudedocs/plans", true, "private working tree")]
    [InlineData("the next step is in HANDOFF", true, "private working tree")]
    [InlineData("re-encoded as UTF-8 from CP-949", false, "encodings")]
    [InlineData("see CVE-2024-1234 and RFC-4180", false, "public identifiers")]
    [InlineData("described in ADR-001", false, "this repository's own ADR numbering")]
    [InlineData("expected Vector<Single, N-1>", false, "a count in a formula")]
    [InlineData("the experiment exp-001 was promoted", false, "an experiment id MLoop mints")]
    [InlineData("applying rule-1 then rule-2", false, "lower-case fixture data")]
    [InlineData("decoded as Latin-1", false, "an encoding whose name is not upper-case")]
    public void GuardSeesEachShapeItForbidsAndToleratesEachNearMiss(string line, bool isInternal, string what)
    {
        var found = InternalTokensIn(line, isProse: true).ToArray();

        if (isInternal)
            Assert.True(found.Length > 0, $"the guard no longer recognizes a {what}: \"{line}\"");
        else
            Assert.True(found.Length == 0, $"the guard now flags {what}, which is public vocabulary: {string.Join(", ", found)}");
    }

    private static IEnumerable<string> InternalTokensIn(string line, bool isProse)
    {
        foreach (Match match in WorkItemId.Matches(line))
        {
            if (!PublicPrefixes.Contains(match.Groups["prefix"].Value))
                yield return match.Value;
        }

        if (isProse)
        {
            foreach (Match match in DriveLetterPath.Matches(line))
                yield return match.Value + "…  (absolute path on one machine)";
        }

        foreach (Match match in InternalWorkUnit.Matches(line))
            yield return match.Value + "  (internal unit of work)";

        foreach (Match match in PrivateWorkspaceRef.Matches(line))
            yield return match.Value + "  (private working tree)";
    }
}
