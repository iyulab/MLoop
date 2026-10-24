using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A path a user or a client gives is resolved by <c>ProjectPaths.Resolve</c> — the project's root for a relative
/// path, the working directory outside a project — and kept inside the project by <c>ProjectPaths.IsWithin</c>.
/// The rule used to be restated as <c>IsPathRooted(p) ? p : Combine(root, p)</c> in twenty-one places, and the
/// server's containment test was a string prefix that a sibling directory named like the root passed.
/// </summary>
public class ProjectPathResolutionContractTests
{
    // `IsPathRooted(x) ? x : Path.Combine(` / `… : Path.GetFullPath(Path.Combine(` — the restated rule.
    private static readonly Regex Restated = new(
        @"IsPathRooted\(\s*([\w.!]+)\s*\)\s*\?\s*\1\s*:\s*(Path\.GetFullPath\(\s*)?Path\.Combine\(",
        RegexOptions.Compiled);

    // A containment test written as a prefix comparison against a root.
    private static readonly Regex PrefixContainment = new(
        @"\.StartsWith\(\s*\w*[Rr]oot\w*\s*,",
        RegexOptions.Compiled);

    private static IEnumerable<string> Product() =>
        RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Where(f => !f.EndsWith("ProjectPaths.cs", StringComparison.Ordinal));

    [Fact]
    public void NoProductCodeRestatesTheResolutionRule_OrTestsContainmentByPrefix()
    {
        // The whole file, not line by line: the restated rule was as often written across three lines.
        var offenders = Product()
            .SelectMany(f => Offences(File.ReadAllText(f))
                .Select(line => $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, f)}:{line}"))
            .ToList();

        Assert.True(offenders.Count == 0,
            "Resolve a given path with ProjectPaths.Resolve and keep it in the project with ProjectPaths.IsWithin:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    private static IEnumerable<int> Offences(string source) =>
        Restated.Matches(source).Concat(PrefixContainment.Matches(source))
            .Select(m => source.AsSpan(0, m.Index).Count('\n') + 1);

    [Fact]
    public void TheScan_SeesTheRuleWrittenAcrossLines() =>
        Assert.Single(Offences("var p = Path.IsPathRooted(dataFile)\n    ? dataFile\n    : Path.Combine(projectRoot, dataFile);"));

    [Theory]
    [InlineData("var p = Path.IsPathRooted(file) ? file : Path.Combine(root, file);")]
    [InlineData("var p = Path.IsPathRooted(data!) ? data! : Path.GetFullPath(Path.Combine(projectRoot, data!));")]
    [InlineData("if (!resolved.StartsWith(trainProjectRoot, StringComparison.OrdinalIgnoreCase))")]
    public void TheScan_SeesTheShapesItForbids(string line) =>
        Assert.NotEmpty(Offences(line));

    [Theory]
    [InlineData("var p = ProjectPaths.Resolve(file, root);")]
    [InlineData("if (string.IsNullOrEmpty(dataFile) || !Path.IsPathRooted(dataFile))")]
    [InlineData("return relative.StartsWith(\"..\", StringComparison.Ordinal);")]
    public void TheScan_PassesTheShapesItAllows(string line) =>
        Assert.Empty(Offences(line));
}
