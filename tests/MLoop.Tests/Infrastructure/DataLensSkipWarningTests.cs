using System.Globalization;
using DataLens.Models;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// DataLens leaves out the analyses that are quadratic in the rows when a file has more of them than a threshold, and says so in a warning of its
/// own. Without MLoop passing that on, <c>analyze outliers</c> on a large file would print a count of outliers as if the local outlier factor had
/// voted - the answer changes meaning and nothing says so.
/// </summary>
public class DataLensSkipWarningTests : IDisposable
{
    private readonly string _csv = Path.Combine(Path.GetTempPath(), $"datalens-skip-{Guid.NewGuid():N}.csv");

    public DataLensSkipWarningTests()
    {
        var rng = new Random(3);
        string V() => (rng.NextDouble() * 10).ToString("R", CultureInfo.InvariantCulture);
        File.WriteAllLines(_csv, new[] { "a,b,c" }.Concat(Enumerable.Range(0, 120).Select(_ => $"{V()},{V()},{V()}")));
    }

    public void Dispose() => File.Delete(_csv);

    [Fact]
    public async Task ASkippedAnalysis_IsReportedAsAWarning_AndTheOutlierFactorIsAbsent()
    {
        using var collected = WarningConsole.Collect();

        var result = await new DataLensAnalyzer().AnalyzeAsync(_csv, DataLensAnalyzer.Only(o =>
        {
            o.IncludeOutliers = true;
            o.LofRowThreshold = 50;
        }));

        Assert.NotNull(result);
        Assert.Null(result!.Outliers!.Lof);
        var warning = Assert.Single(collected.Warnings);
        Assert.Contains("local outlier factor was skipped", warning);
        Assert.Contains("isolation forest and Mahalanobis distance both flag", warning);
    }

    [Fact]
    public async Task AnAnalysisThatRan_RaisesNoWarning()
    {
        using var collected = WarningConsole.Collect();

        var result = await new DataLensAnalyzer().AnalyzeAsync(_csv, DataLensAnalyzer.Only(o => o.IncludeOutliers = true));

        Assert.NotNull(result!.Outliers!.Lof);
        Assert.Empty(collected.Warnings);
    }

    [Fact]
    public void AnAnalysisWithoutAMessageOfItsOwn_IsNamedAndSaidToBeSkipped()
    {
        var message = DataLensAnalyzer.SkippedMessage(new AnalysisWarning("Changepoints", WarningCategory.SkippedForSize, "x"));

        Assert.StartsWith("Changepoints was skipped", message);
    }
}
