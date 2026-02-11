using MLoop.Core.DataQuality;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.DataQuality;

public class DataQualityAnalyzerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly DataQualityAnalyzer _analyzer;

    public DataQualityAnalyzerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopDQTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _analyzer = new DataQualityAnalyzer(new TestLogger());
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _analyzer.AnalyzeAsync(Path.Combine(_tempDirectory, "nonexistent.csv")));
    }

    [Fact]
    public async Task AnalyzeAsync_WithCleanData_ReturnsNoIssues()
    {
        var csvPath = CreateTestCsv(
            "id,feature1,feature2,label",
            "1,1.5,3.0,A",
            "2,2.5,4.0,B",
            "3,3.5,5.0,A",
            "4,4.5,6.0,B",
            "5,5.5,7.0,A");

        var issues = await _analyzer.AnalyzeAsync(csvPath, "label");

        // Clean data should have no critical issues
        var criticalIssues = issues.Where(i => i.Severity == IssueSeverity.Critical).ToList();
        Assert.Empty(criticalIssues);
    }

    [Fact]
    public async Task AnalyzeAsync_WithConstantColumn_DetectsIssue()
    {
        var csvPath = CreateTestCsv(
            "id,constant_col,label",
            "1,X,A",
            "2,X,B",
            "3,X,A",
            "4,X,B");

        var issues = await _analyzer.AnalyzeAsync(csvPath, "label");

        var constantIssue = issues.FirstOrDefault(i => i.Type == DataQualityIssueType.ConstantColumn);
        Assert.NotNull(constantIssue);
        Assert.Equal("constant_col", constantIssue.ColumnName);
    }

    [Fact]
    public async Task AnalyzeAsync_WithDuplicateRows_DetectsIssue()
    {
        var csvPath = CreateTestCsv(
            "id,value",
            "1,A",
            "2,B",
            "1,A",
            "3,C",
            "1,A",
            "2,B");

        var issues = await _analyzer.AnalyzeAsync(csvPath);

        var duplicateIssue = issues.FirstOrDefault(i => i.Type == DataQualityIssueType.DuplicateRows);
        Assert.NotNull(duplicateIssue);
        Assert.Contains("duplicate", duplicateIssue.Description, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AnalyzeAsync_WithClassImbalance_DetectsIssue()
    {
        // Create severe class imbalance: 22 A's vs 2 B's (11:1 ratio, exceeds >10 threshold)
        var lines = new List<string> { "feature,label" };
        for (int i = 0; i < 22; i++)
            lines.Add($"{i},A");
        lines.Add("100,B");
        lines.Add("101,B");

        var csvPath = CreateTestCsv(lines.ToArray());

        var issues = await _analyzer.AnalyzeAsync(csvPath, "label");

        var imbalanceIssue = issues.FirstOrDefault(i => i.Type == DataQualityIssueType.ClassImbalance);
        Assert.NotNull(imbalanceIssue);
        Assert.Equal("label", imbalanceIssue.ColumnName);
        Assert.Equal(IssueSeverity.High, imbalanceIssue.Severity); // 11:1 > 10 threshold
    }

    [Fact]
    public async Task AnalyzeAsync_WithBalancedClasses_NoImbalanceIssue()
    {
        var lines = new List<string> { "feature,label" };
        for (int i = 0; i < 10; i++)
            lines.Add($"{i},A");
        for (int i = 0; i < 10; i++)
            lines.Add($"{i + 10},B");

        var csvPath = CreateTestCsv(lines.ToArray());

        var issues = await _analyzer.AnalyzeAsync(csvPath, "label");

        var imbalanceIssue = issues.FirstOrDefault(i => i.Type == DataQualityIssueType.ClassImbalance);
        Assert.Null(imbalanceIssue);
    }

    [Fact]
    public async Task AnalyzeAsync_WithConstantLabelColumn_DetectsCriticalIssue()
    {
        var csvPath = CreateTestCsv(
            "feature,label",
            "1,A",
            "2,A",
            "3,A",
            "4,A");

        var issues = await _analyzer.AnalyzeAsync(csvPath, "label");

        var labelConstant = issues.FirstOrDefault(
            i => i.Type == DataQualityIssueType.ConstantColumn && i.ColumnName == "label");
        Assert.NotNull(labelConstant);
        Assert.Equal(IssueSeverity.Critical, labelConstant.Severity);
    }

    [Fact]
    public async Task NeedsPreprocessingAsync_WithCleanData_ReturnsFalse()
    {
        var csvPath = CreateTestCsv(
            "feature1,feature2,label",
            "1,2,A",
            "3,4,B",
            "5,6,A",
            "7,8,B");

        var needsPreprocessing = await _analyzer.NeedsPreprocessingAsync(csvPath, "label");

        Assert.False(needsPreprocessing);
    }

    [Fact]
    public async Task NeedsPreprocessingAsync_WithSevereImbalance_ReturnsTrue()
    {
        var lines = new List<string> { "feature,label" };
        for (int i = 0; i < 50; i++)
            lines.Add($"{i},A");
        lines.Add("100,B");

        var csvPath = CreateTestCsv(lines.ToArray());

        var needsPreprocessing = await _analyzer.NeedsPreprocessingAsync(csvPath, "label");

        Assert.True(needsPreprocessing);
    }

    private string CreateTestCsv(params string[] lines)
    {
        var path = Path.Combine(_tempDirectory, $"test_{Guid.NewGuid()}.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    private class TestLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
