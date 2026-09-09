using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A central <c>PackageVersion</c> raised for a security advisory must also be a direct
/// <c>PackageReference</c> of a shipping project — otherwise the floor it enforces stops at this
/// repository's own build and never reaches a consumer.
/// </summary>
/// <remarks>
/// <para>
/// Central Package Management makes a <c>PackageVersion</c> look like a version decision about a
/// package, and for this repository's own restore it is one. It is not part of the dependency graph
/// a project hands to whoever references it: a consumer referencing <c>MLoop.Core</c> by project
/// resolves that package transitively through <c>MLoop.Core</c>'s own manifest, which never
/// mentions the pin. So a floor declared only here is repo-build-scoped, and a consumer silently
/// re-inherits the vulnerable version the pin was written to exclude.
/// </para>
/// <para>
/// This has now happened three times, to three different packages, each time starting from the same
/// assumption. Twice it was resolved by adding a direct <c>PackageReference</c> so the floor rides
/// the package graph; once by advancing the dependency whose own manifest then carried the floor,
/// which is why that override was removed rather than made direct. Prose already stated the rule —
/// <c>docs/EMBEDDING.md</c> §2 says it in as many words — and stated it about one package, so the
/// next package that needed it did not inherit the lesson. This guard states it about the shape
/// instead.
/// </para>
/// <para>
/// The marker is the one the file already uses: a <c>Security:</c> comment above the entry. That is
/// deliberate — an override written for build coherence rather than for an advisory (the Serilog
/// pin, whose comment explains that two project graphs must resolve the same assembly) is
/// legitimately transitive-only and this guard must not refuse it. What separates the two is intent,
/// and the comment is where this repository already records it.
/// </para>
/// </remarks>
public class SecurityFloorReachContractTests
{
    /// <summary>Matches a <c>PackageVersion</c> entry, capturing the package id.</summary>
    private static readonly Regex PackageVersionEntry =
        new(@"<PackageVersion\s+Include\s*=\s*""(?<id>[^""]+)""", RegexOptions.Compiled);

    /// <summary>Matches a <c>PackageReference</c> entry, capturing the package id.</summary>
    private static readonly Regex PackageReferenceEntry =
        new(@"<PackageReference\s+Include\s*=\s*""(?<id>[^""]+)""", RegexOptions.Compiled);

    /// <summary>Opens a comment whose first word declares the entries below it a security floor.</summary>
    private static readonly Regex SecurityComment =
        new(@"<!--\s*Security\s*:", RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Opens any comment — a new one ends the previous one's reach.</summary>
    private static readonly Regex AnyComment = new(@"<!--", RegexOptions.Compiled);

    [Fact]
    public void EverySecurityFloorIsAlsoADirectReferenceSoItReachesConsumers()
    {
        var floors = SecurityFloorsIn(File.ReadAllLines(PackagesPropsPath));
        var direct = DirectlyReferencedPackages();

        var unreachable = floors.Where(f => !direct.Contains(f)).ToList();

        Assert.True(
            unreachable.Count == 0,
            "A Security-marked PackageVersion governs this repository's own restore only. To reach a "
            + "ProjectReference consumer the floor must ride the package graph — either as a direct "
            + "PackageReference of a project under src/, or by advancing the dependency that pulls "
            + "the vulnerable version until its own manifest carries the floor (in which case remove "
            + "the override rather than marking it Security). Unreachable floors:"
            + Environment.NewLine + string.Join(Environment.NewLine, unreachable));
    }

    [Fact]
    public void TheCheckSeesTheSecurityMarkedEntriesThisRepositoryActuallyHas()
    {
        // Negative control: the assertion above also passes when the parse finds nothing at all — a
        // props file it could not read, or a marker whose spelling drifted, would report a clean
        // tree with the same confidence as a genuinely clean one. Pin that it reads the real file
        // and recognizes the entry that is there.
        var lines = File.ReadAllLines(PackagesPropsPath);
        Assert.True(lines.Length > 20, $"Expected the real Directory.Packages.props, read {lines.Length} lines.");

        var floors = SecurityFloorsIn(lines);
        Assert.Contains("Microsoft.Bcl.Memory", floors);

        var direct = DirectlyReferencedPackages();
        Assert.Contains("FilePrepper", direct);
        Assert.True(direct.Count > 10, $"Expected the projects' references, found {direct.Count}.");
    }

    [Fact]
    public void TheCheckReportsASecurityFloorThatIsNotDirectlyReferenced()
    {
        // The exact shape this guard forbids — and the shape this repository actually shipped, until
        // a consumer measured it rather than this suite catching it.
        string[] offending =
        [
            "<Project><ItemGroup>",
            "  <!-- Security: Override transitive dependency via FilePrepper->EPPlus (GHSA-cvvh-rhrc-wg4q). -->",
            "  <PackageVersion Include=\"System.Security.Cryptography.Xml\" Version=\"10.0.10\" />",
            "</ItemGroup></Project>",
        ];

        var floors = SecurityFloorsIn(offending);

        Assert.Contains("System.Security.Cryptography.Xml", floors);
        Assert.DoesNotContain("System.Security.Cryptography.Xml", DirectlyReferencedPackages());
    }

    [Fact]
    public void TheCheckLeavesATransitivePinThatIsNotASecurityFloorAlone()
    {
        // A pin can be transitive-only for a legitimate reason. Refusing those would make the guard a
        // rule about CPM rather than about advisories, and the first person to hit it would delete
        // the guard rather than the pin.
        string[] legitimate =
        [
            "<Project><ItemGroup>",
            "  <!-- Pin the transitive Serilog core so two project graphs resolve the SAME version.",
            "       Without this, the bundled output ships one version where deps.json demands another. -->",
            "  <PackageVersion Include=\"Serilog\" Version=\"4.3.0\" />",
            "</ItemGroup></Project>",
        ];

        Assert.Empty(SecurityFloorsIn(legitimate));
    }

    [Fact]
    public void ASecurityCommentReachesOnlyAsFarAsTheNextComment()
    {
        // The marker sits above the entry, so its reach has to end somewhere. It ends at the next
        // comment — which is how the file is already written, and what keeps a Security note from
        // silently claiming every entry that follows it to the end of the group.
        string[] mixed =
        [
            "  <!-- Security: advisory GHSA-xxxx -->",
            "  <PackageVersion Include=\"Guarded.One\" Version=\"1.0.0\" />",
            "  <!-- Ordinary grouping comment -->",
            "  <PackageVersion Include=\"Unguarded.Two\" Version=\"2.0.0\" />",
        ];

        var floors = SecurityFloorsIn(mixed);

        Assert.Contains("Guarded.One", floors);
        Assert.DoesNotContain("Unguarded.Two", floors);
    }

    private static string PackagesPropsPath =>
        Path.Combine(RepoSourceTree.RepoRoot, "Directory.Packages.props");

    /// <summary>
    /// The package ids declared a security floor by the comment above them, in reading order.
    /// </summary>
    /// <remarks>
    /// A comment's reach ends at the next comment, matching how the file groups its entries. The
    /// parse is deliberately textual rather than an XML read: the marker lives in a comment, which
    /// an element-shaped view drops.
    /// </remarks>
    private static IReadOnlyList<string> SecurityFloorsIn(IReadOnlyList<string> propsLines)
    {
        var floors = new List<string>();
        var underSecurityComment = false;

        foreach (var line in propsLines)
        {
            if (AnyComment.IsMatch(line))
                underSecurityComment = SecurityComment.IsMatch(line);

            var match = PackageVersionEntry.Match(line);
            if (match.Success && underSecurityComment)
                floors.Add(match.Groups["id"].Value);
        }

        return floors;
    }

    /// <summary>
    /// Every package some shipping project references directly — the set a floor must be in to
    /// travel with the package graph.
    /// </summary>
    /// <remarks>
    /// Scoped to <c>src/</c>: a floor added to a test project or to the CLI host does not reach a
    /// consumer that references a library, which is the boundary all three incidents crossed.
    /// </remarks>
    private static HashSet<string> DirectlyReferencedPackages()
    {
        var srcRoot = Path.Combine(RepoSourceTree.RepoRoot, "src");
        var referenced = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var project in Directory.EnumerateFiles(srcRoot, "*.csproj", SearchOption.AllDirectories))
            foreach (Match match in PackageReferenceEntry.Matches(File.ReadAllText(project)))
                referenced.Add(match.Groups["id"].Value);

        return referenced;
    }
}
