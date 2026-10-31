using System.Globalization;
using DataLens.Models;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// DataLens answers an analysis it cannot run with an empty result and a warning of its own. Without MLoop passing the warning on,
/// <c>analyze correlation</c> on two rows printed "no high-correlation pairs found" - an empty result read as a finding.
/// </summary>
public class DataLensAnalysisLimitTests : IDisposable
{
    private readonly List<string> _files = [];

    public void Dispose() => _files.ForEach(File.Delete);

    private string Csv(IEnumerable<string> rows)
    {
        var path = Path.Combine(Path.GetTempPath(), $"datalens-limit-{Guid.NewGuid():N}.csv");
        File.WriteAllLines(path, rows);
        _files.Add(path);
        return path;
    }

    private static IEnumerable<string> Rows(int count, Func<int, string> row)
    {
        yield return "a,b,c";
        for (var i = 0; i < count; i++) yield return row(i);
    }

    private static string V(Random rng) => rng.NextDouble().ToString("R", CultureInfo.InvariantCulture);

    private static async Task<List<string>> CorrelationWarningsAsync(string csv)
    {
        using var collected = WarningConsole.Collect();
        var result = await new DataLensAnalyzer().AnalyzeAsync(csv, DataLensAnalyzer.Only(o => o.IncludeCorrelation = true));
        Assert.NotNull(result);
        return [.. collected.Warnings];
    }

    [Fact]
    public async Task AnAnalysisThatCouldNotRunForWantOfRows_SaysSo()
    {
        var rng = new Random(1);
        var warnings = await CorrelationWarningsAsync(Csv(Rows(2, _ => $"{V(rng)},{V(rng)},{V(rng)}")));

        var warning = Assert.Single(warnings);
        Assert.Contains("Correlation analysis did not run (too few rows)", warning);
        Assert.Contains("says nothing about the data", warning);
    }

    [Fact]
    public async Task AColumnLeftOutOfAnAnalysis_IsNamed()
    {
        var rng = new Random(2);
        var warnings = await CorrelationWarningsAsync(Csv(Rows(200, _ => $"{V(rng)},{V(rng)},5")));

        Assert.Contains(warnings, w => w.Contains("left out the constant or empty column(s) c"));
    }

    [Fact]
    public async Task AnAnalysisThatRanOnAHealthyFile_RaisesNoWarning()
    {
        var rng = new Random(3);

        Assert.Empty(await CorrelationWarningsAsync(Csv(Rows(200, _ => $"{V(rng)},{V(rng)},{V(rng)}"))));
    }

    [Fact]
    public async Task AWarningAboutTheDataItself_IsNotPassedOn()
    {
        // 200 rows drawn from 150 distinct ones: DataLens notes the repeated rows, which says nothing about whether the analysis ran.
        var warnings = await CorrelationWarningsAsync(Csv(Rows(200, i =>
        {
            var rng = new Random(i % 150);
            return $"{V(rng)},{V(rng)},{V(rng)}";
        })));

        Assert.Empty(warnings);
    }

    [Theory]
    [InlineData(WarningCategory.DuplicateRows)]
    [InlineData(WarningCategory.DuplicateColumns)]
    [InlineData(WarningCategory.MissingnessPattern)]
    [InlineData(WarningCategory.SingularCovariance)]
    [InlineData(WarningCategory.ConstantColumns)]
    [InlineData(WarningCategory.TypeMappingFailed)]
    public void ADescriptionOfTheData_IsNotAnAnalysisLimit(WarningCategory category) =>
        Assert.Null(DataLensAnalyzer.AnalysisLimit(new AnalysisWarning("Correlation", category, "x")));

    [Theory]
    [InlineData(WarningCategory.InsufficientRows)]
    [InlineData(WarningCategory.InsufficientColumns)]
    [InlineData(WarningCategory.InsufficientUsableColumns)]
    [InlineData(WarningCategory.DegenerateColumnsExcluded)]
    [InlineData(WarningCategory.AllPairsUndefined)]
    [InlineData(WarningCategory.ComputationFailed)]
    [InlineData(WarningCategory.UpstreamError)]
    public void AnAnalysisThatDidNotRunOrRanOnLess_IsAnAnalysisLimit(WarningCategory category)
    {
        var text = DataLensAnalyzer.AnalysisLimit(new AnalysisWarning("Outlier", category, "engine said no"));

        Assert.NotNull(text);
        Assert.StartsWith("The Outlier analysis", text);
    }

    [Fact]
    public void AFailedAnalysis_CarriesTheReasonItGave() =>
        Assert.EndsWith("engine said no",
            DataLensAnalyzer.AnalysisLimit(new AnalysisWarning("Features", WarningCategory.ComputationFailed, "engine said no")));
}
