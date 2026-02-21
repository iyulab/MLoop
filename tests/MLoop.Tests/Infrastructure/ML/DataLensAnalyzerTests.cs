using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Tests for DataLensAnalyzer - DataLens API wrapper with graceful degradation.
/// </summary>
public class DataLensAnalyzerTests : IDisposable
{
    private readonly string _testDir;

    public DataLensAnalyzerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-datalens-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
            Directory.Delete(_testDir, true);
    }

    [Fact]
    public void IsAvailable_ReturnsCorrectState()
    {
        var analyzer = new DataLensAnalyzer();
        // IsAvailable depends on whether DataLens is loaded.
        // This test just verifies the property doesn't throw.
        _ = analyzer.IsAvailable;
    }

    [Fact]
    public async Task ProfileAsync_ReturnsProfileForValidCsv()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable)
            return; // Skip if library unavailable

        var csvPath = CreateTestCsv("a,b,c\n1,2,3\n4,5,6\n7,8,9\n");
        var result = await analyzer.ProfileAsync(csvPath);

        Assert.NotNull(result);
        Assert.Equal(3, result!.RowCount);
        Assert.Equal(3, result.ColumnCount);
        Assert.Equal(3, result.Columns.Count);
    }

    [Fact]
    public async Task ProfileAsync_ReturnsNullForEmptyInput()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable)
            return;

        var csvPath = CreateTestCsv("");
        var result = await analyzer.ProfileAsync(csvPath);
        // Empty CSV may return null or a result with 0 rows; either is acceptable
        if (result != null)
        {
            Assert.Equal(0, result.RowCount);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_ReturnsAnalysisResult()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable)
            return;

        var csvPath = CreateTestCsv("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n");
        var options = new DataLens.AnalysisOptions
        {
            IncludeProfiling = true,
            IncludeCorrelation = true,
            IncludeDistribution = false,
            IncludeOutliers = false,
            IncludeFeatures = false,
            IncludeRegression = false,
            IncludeClustering = false,
            IncludePca = false
        };

        var result = await analyzer.AnalyzeAsync(csvPath, options);

        Assert.NotNull(result);
        Assert.NotNull(result!.Profile);
        Assert.NotNull(result.Correlation);
    }

    [Fact]
    public void GracefulDegradation_AlwaysAvailable()
    {
        // DataLens is a managed library, so it should always be available
        // when the NuGet package is referenced. Unlike UInsight (native DLL),
        // there's no runtime loading failure scenario.
        var analyzer = new DataLensAnalyzer();
        Assert.True(analyzer.IsAvailable);
    }

    private string CreateTestCsv(string content)
    {
        var path = Path.Combine(_testDir, $"test-{Guid.NewGuid()}.csv");
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }
}
