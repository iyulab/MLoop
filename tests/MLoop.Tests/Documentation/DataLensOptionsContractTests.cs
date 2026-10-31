using System.Reflection;
using System.Text.RegularExpressions;
using DataLens;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Documentation;

/// <summary>
/// DataLens runs whatever <see cref="AnalysisOptions"/> says, and a switch it adds in a later version defaults to on. Change points came in
/// that way: every <c>mloop info --analyze</c> and <c>mloop analyze</c> ran them, quadratic in the rows (ten seconds for fifty thousand
/// of them, a quarter of an hour for a million), though nothing in MLoop reads a change point. Each place that built its own options listed
/// ten switches by hand and none of them named the new one. The options are built in one place, <see cref="DataLensAnalyzer.Only"/>, from
/// everything off.
/// </summary>
public class DataLensOptionsContractTests
{
    private static readonly Regex OwnOptions = new(
        @"\bnew\s+(DataLens\.)?AnalysisOptions\b|\bAnalysisOptions\s+\w+\s*=\s*new\s*\(",
        RegexOptions.Compiled);

    private static IEnumerable<PropertyInfo> Switches() =>
        typeof(AnalysisOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.PropertyType == typeof(bool) && p.CanWrite);

    [Fact]
    public void Only_TurnsOnNothingButWhatIsChosen()
    {
        var none = DataLensAnalyzer.Only(_ => { });
        Assert.All(Switches(), p => Assert.False((bool)p.GetValue(none)!, $"{p.Name} is on without being asked for"));

        var correlation = DataLensAnalyzer.Only(o => o.IncludeCorrelation = true);
        Assert.Equal(["IncludeCorrelation"], Switches().Where(p => (bool)p.GetValue(correlation)!).Select(p => p.Name));
    }

    [Fact]
    public void Only_KeepsTheOtherSettingsTheCallerChooses()
    {
        var options = DataLensAnalyzer.Only(o =>
        {
            o.IncludeFeatures = true;
            o.TargetColumn = "price";
        });

        Assert.True(options.IncludeFeatures);
        Assert.Equal("price", options.TargetColumn);
    }

    [Fact]
    public void TheOptionsAreBuiltOnlyInTheAnalyzer()
    {
        var offenders = Cli()
            .Where(f => !f.EndsWith("DataLensAnalyzer.cs", StringComparison.Ordinal))
            .SelectMany(f => Offences(File.ReadAllText(f)).Select(line => $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, f)}:{line}"))
            .ToList();

        Assert.True(offenders.Count == 0,
            "Build DataLens options with DataLensAnalyzer.Only(...), so a switch DataLens adds later stays off until it is asked for:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheScan_SeesTheAnalyzerThatBuildsThem() =>
        Assert.Contains(Cli(), f => f.EndsWith("DataLensAnalyzer.cs", StringComparison.Ordinal) && Offences(File.ReadAllText(f)).Any());

    private static IEnumerable<string> Cli() => RepoSourceTree.ProductionSourceFiles("tools");

    private static IEnumerable<int> Offences(string source) =>
        OwnOptions.Matches(source).Select(m => source.AsSpan(0, m.Index).Count('\n') + 1);

    [Theory]
    [InlineData("var options = new AnalysisOptions { IncludeCorrelation = true };")]
    [InlineData("var result = await dataLens.AnalyzeAsync(file, new AnalysisOptions")]
    [InlineData("var o = new DataLens.AnalysisOptions();")]
    [InlineData("AnalysisOptions options = new();")]
    public void TheScan_SeesTheShapesItForbids(string line) =>
        Assert.NotEmpty(Offences(line));

    [Theory]
    [InlineData("var options = DataLensAnalyzer.Only(o => o.IncludeCorrelation = true);")]
    [InlineData("void Run(AnalysisOptions options) { }")]
    public void TheScan_PassesTheShapesItAllows(string line) =>
        Assert.Empty(Offences(line));
}
