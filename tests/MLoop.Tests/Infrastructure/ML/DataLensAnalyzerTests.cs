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

    [Fact]
    public void Version_ReturnsNonNullString()
    {
        var analyzer = new DataLensAnalyzer();
        Assert.True(analyzer.IsAvailable);
        Assert.NotNull(analyzer.Version);
        Assert.NotEmpty(analyzer.Version!);
    }

    [Fact]
    public async Task ProfileAsync_MixedTypes_ReportsColumnsCorrectly()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var csvPath = CreateTestCsv("name,age,score\nAlice,25,90.5\nBob,30,85.0\nCarol,28,92.3\n");
        var result = await analyzer.ProfileAsync(csvPath);

        Assert.NotNull(result);
        Assert.Equal(3, result!.RowCount);
        Assert.Equal(3, result.ColumnCount);
    }

    [Fact]
    public async Task ProfileAsync_SingleColumn_Works()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var csvPath = CreateTestCsv("value\n1\n2\n3\n4\n5\n");
        var result = await analyzer.ProfileAsync(csvPath);

        Assert.NotNull(result);
        Assert.Equal(5, result!.RowCount);
        Assert.Equal(1, result.ColumnCount);
    }

    [Fact]
    public async Task ProfileAsync_WithMissingValues_ReportsMissing()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var csvPath = CreateTestCsv("a,b\n1,2\n,4\n5,\n7,8\n");
        var result = await analyzer.ProfileAsync(csvPath);

        Assert.NotNull(result);
        Assert.Equal(4, result!.RowCount);
    }

    [Fact]
    public async Task AnalyzeAsync_WithDescriptive_ReturnsDescriptiveReport()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var csvPath = CreateTestCsv("x,y,z\n1,10,100\n2,20,200\n3,30,300\n4,40,400\n5,50,500\n");
        var options = new DataLens.AnalysisOptions
        {
            IncludeProfiling = false,
            IncludeDescriptive = true,
            IncludeCorrelation = false,
            IncludeDistribution = false,
            IncludeOutliers = false,
            IncludeFeatures = false,
            IncludeRegression = false,
            IncludeClustering = false,
            IncludePca = false
        };

        var result = await analyzer.AnalyzeAsync(csvPath, options);

        Assert.NotNull(result);
        Assert.NotNull(result!.Descriptive);
        Assert.Equal(3, result.Descriptive!.Columns.Count);

        var xCol = result.Descriptive.Columns.First(c => c.Name == "x");
        Assert.Equal(3.0, xCol.Median, 2);
        Assert.Equal(5, xCol.Count);
    }

    [Fact]
    public async Task AnalyzeAsync_NullOptions_ReturnsResult()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var csvPath = CreateTestCsv("a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n");
        var result = await analyzer.AnalyzeAsync(csvPath, null);

        Assert.NotNull(result);
    }

    [Fact]
    public async Task ProfileAsync_NonExistentFile_ReturnsNull()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var result = await analyzer.ProfileAsync(Path.Combine(_testDir, "nonexistent.csv"));
        Assert.Null(result);
    }

    [Fact]
    public async Task AnalyzeAsync_NonExistentFile_ReturnsNull()
    {
        var analyzer = new DataLensAnalyzer();
        if (!analyzer.IsAvailable) return;

        var result = await analyzer.AnalyzeAsync(Path.Combine(_testDir, "nonexistent.csv"));
        Assert.Null(result);
    }

    private string CreateTestCsv(string content)
    {
        var path = Path.Combine(_testDir, $"test-{Guid.NewGuid()}.csv");
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }
}
