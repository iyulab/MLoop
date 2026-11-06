using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A model name somebody supplies becomes a directory, so it goes through <c>ModelName.Resolve</c> / <c>Filter</c> / <c>Require</c> —
/// the one gate that refuses <c>..</c> and a separator. It used to be restated as <c>name.Trim().ToLowerInvariant()</c> in twenty
/// places, each normalizing and none checking, which is how <c>?name=../../other</c> read a directory outside the project and
/// <c>POST /train</c> wrote one (published 0.49.0).
/// </summary>
public class ModelNameGateContractTests
{
    // A model name lower-cased by hand: `modelName.ToLowerInvariant()`, `resolvedModelName.Trim().ToLowerInvariant()`.
    private static readonly Regex Restated = new(
        @"\b\w*[Mm]odel\w*(\?)?\.(Trim\(\)\.)?ToLowerInvariant\(\)",
        RegexOptions.Compiled);

    // The same rule written with the identifier `name` where the surrounding method is about a model: only the shape
    // `IsNullOrWhiteSpace(x) ? Default... : x.Trim().ToLowerInvariant()` is recognizable without knowing what `x` holds.
    private static readonly Regex DefaultOrLowered = new(
        @"IsNullOrWhiteSpace\((\w+)\)\s*\?\s*(ConfigDefaults\.DefaultModelName|null)\s*:\s*\1\.Trim\(\)\.ToLowerInvariant\(\)",
        RegexOptions.Compiled);

    private static IEnumerable<string> Product() =>
        RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Where(f => !f.EndsWith(Path.Combine("Storage", "ModelName.cs"), StringComparison.Ordinal));

    [Fact]
    public void NoProductCodeLowerCasesAModelNameByHand()
    {
        var offenders = Product()
            .SelectMany(f => Offences(File.ReadAllText(f))
                .Select(line => $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, f)}:{line}"))
            .ToList();

        Assert.True(offenders.Count == 0,
            "A model name goes through ModelName.Resolve (or Filter / Require), which checks it as well as normalizing it:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    private static IEnumerable<int> Offences(string source) =>
        Restated.Matches(source).Concat(DefaultOrLowered.Matches(source))
            .Select(m => source.AsSpan(0, m.Index).Count('\n') + 1);

    [Fact]
    public void TheScan_FindsTheKnownShapesInTheTree_BeforeTheyWereRemoved() =>
        // The guard must see what it polices: the shapes below are the ones that stood in the CLI.
        Assert.True(Offences("""
            var a = modelName.Trim().ToLowerInvariant();
            var b = string.IsNullOrWhiteSpace(modelName)
                ? null
                : modelName.Trim().ToLowerInvariant();
            await sampler.SampleAsync(resolvedModelName.ToLowerInvariant());
            """).Distinct().Count() >= 3);

    [Theory]
    [InlineData("var a = modelName.Trim().ToLowerInvariant();")]
    [InlineData("var a = resolvedModelName.ToLowerInvariant();")]
    [InlineData("var a = string.IsNullOrWhiteSpace(n) ? ConfigDefaults.DefaultModelName : n.Trim().ToLowerInvariant();")]
    public void TheScan_SeesTheShapesItForbids(string line) =>
        Assert.NotEmpty(Offences(line));

    [Theory]
    [InlineData("var a = MLoop.Core.Storage.ModelName.Resolve(modelName);")]
    [InlineData("var m = metricName.ToLowerInvariant();")]
    [InlineData("var t = task.Trim().ToLowerInvariant();")]
    [InlineData("var s = sanitized.ToLowerInvariant();")]
    public void TheScan_PassesTheShapesItAllows(string line) =>
        Assert.Empty(Offences(line));
}
