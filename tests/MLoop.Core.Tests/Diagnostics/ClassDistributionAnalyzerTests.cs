using MLoop.Core.Data;
using MLoop.Core.Diagnostics;

namespace MLoop.Core.Tests.Diagnostics;

/// <summary>
/// Tests for ClassDistributionAnalyzer - class balance analysis for classification.
/// </summary>
public class ClassDistributionAnalyzerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly CsvHelperImpl _csvHelper;
    private readonly ClassDistributionAnalyzer _analyzer;

    public ClassDistributionAnalyzerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _csvHelper = new CsvHelperImpl();
        _analyzer = new ClassDistributionAnalyzer(_csvHelper);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_BalancedBinaryClasses_ReturnsBalanced()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "balanced.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,A\n3,A\n4,B\n5,B\n6,B");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Equal(2, result.ClassCount);
        Assert.Equal(ClassBalanceLevel.Balanced, result.BalanceLevel);
        Assert.Equal(1.0, result.ImbalanceRatio);
        Assert.False(result.NeedsAttention);
    }

    [Fact]
    public async Task AnalyzeAsync_SlightlyImbalanced_ReturnsSlightlyImbalanced()
    {
        // Arrange - ratio ~2:1
        var csvPath = Path.Combine(_tempDirectory, "slight.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,A\n3,A\n4,A\n5,B\n6,B");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Equal(ClassBalanceLevel.SlightlyImbalanced, result.BalanceLevel);
        Assert.Equal(2.0, result.ImbalanceRatio);
    }

    [Fact]
    public async Task AnalyzeAsync_SeverelyImbalanced_ReturnsSeverelyImbalanced()
    {
        // Arrange - ratio 100:1
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 100; i++) lines.Add($"{i},Majority");
        lines.Add("100,Minority");

        var csvPath = Path.Combine(_tempDirectory, "severe.csv");
        await File.WriteAllTextAsync(csvPath, string.Join("\n", lines));

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Equal(ClassBalanceLevel.SeverelyImbalanced, result.BalanceLevel);
        Assert.True(result.NeedsAttention);
        Assert.NotEmpty(result.Warnings);
        Assert.NotEmpty(result.SuggestedStrategies);
    }

    [Fact]
    public async Task AnalyzeAsync_WithMissingLabels_DetectsEmptyClass()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "missing.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,\n3,A\n4,B\n5,");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.True(result.ClassDistribution.ContainsKey("(empty)"));
        Assert.Equal(2, result.ClassDistribution["(empty)"]);
        Assert.Contains(result.Warnings, w => w.Contains("empty/missing labels"));
    }

    [Fact]
    public async Task AnalyzeAsync_MultipleClasses_CorrectlyCounts()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "multi.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,Cat\n2,Dog\n3,Bird\n4,Cat\n5,Dog");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Equal(3, result.ClassCount);
        Assert.Equal(5, result.TotalRows);
        Assert.Equal(2, result.ClassDistribution["Cat"]);
        Assert.Equal(2, result.ClassDistribution["Dog"]);
        Assert.Equal(1, result.ClassDistribution["Bird"]);
    }

    [Fact]
    public async Task AnalyzeAsync_RareClasses_WarnsAboutRareClasses()
    {
        // Arrange - class with fewer than 10 samples
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 50; i++) lines.Add($"{i},Common");
        for (int i = 0; i < 5; i++) lines.Add($"{50 + i},Rare");

        var csvPath = Path.Combine(_tempDirectory, "rare.csv");
        await File.WriteAllTextAsync(csvPath, string.Join("\n", lines));

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Contains(result.Warnings, w => w.Contains("fewer than 10 samples"));
    }

    [Fact]
    public async Task AnalyzeAsync_SingleSampleClass_WarnsCannotTrain()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "single.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,A\n3,A\n4,B");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Contains(result.Warnings, w => w.Contains("only 1 sample"));
    }

    [Fact]
    public async Task AnalyzeAsync_NonExistentColumn_ReturnsError()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "test.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,B");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "NonExistent");

        // Assert
        Assert.NotNull(result.Error);
        Assert.Contains("not found", result.Error);
    }

    [Fact]
    public async Task AnalyzeAsync_EmptyFile_ReturnsError()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "empty.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.NotNull(result.Error);
        Assert.Contains("Empty", result.Error);
    }

    [Fact]
    public async Task AnalyzeAsync_GeneratesVisualization()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "viz.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,Apple\n2,Apple\n3,Apple\n4,Banana\n5,Banana");

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.NotEmpty(result.DistributionVisualization);
        Assert.Contains("Apple", result.DistributionVisualization);
        Assert.Contains("Banana", result.DistributionVisualization);
        Assert.Contains("█", result.DistributionVisualization);
    }

    [Fact]
    public async Task AnalyzeAsync_ModerateImbalance_SuggestsClassWeights()
    {
        // Arrange - ratio ~5:1
        var csvPath = Path.Combine(_tempDirectory, "moderate.csv");
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 50; i++) lines.Add($"{i},Majority");
        for (int i = 0; i < 10; i++) lines.Add($"{50 + i},Minority");
        await File.WriteAllTextAsync(csvPath, string.Join("\n", lines));

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.True(result.BalanceLevel >= ClassBalanceLevel.ModeratelyImbalanced);
        Assert.Contains(SamplingStrategy.ClassWeights, result.SuggestedStrategies);
    }

    [Fact]
    public async Task AnalyzeAsync_HighImbalance_SuggestsSMOTE()
    {
        // Arrange - ratio ~20:1
        var csvPath = Path.Combine(_tempDirectory, "high.csv");
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 100; i++) lines.Add($"{i},Majority");
        for (int i = 0; i < 5; i++) lines.Add($"{100 + i},Minority");
        await File.WriteAllTextAsync(csvPath, string.Join("\n", lines));

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.True(result.BalanceLevel >= ClassBalanceLevel.HighlyImbalanced);
        Assert.Contains(SamplingStrategy.SMOTE, result.SuggestedStrategies);
    }

    [Fact]
    public async Task AnalyzeAsync_SingleClass_ReturnsSeverelyImbalanced()
    {
        // Arrange - all rows have the same label
        var csvPath = Path.Combine(_tempDirectory, "single_class.csv");
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 100; i++) lines.Add($"{i},Normal");
        await File.WriteAllTextAsync(csvPath, string.Join("\n", lines));

        // Act
        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        // Assert
        Assert.Equal(1, result.ClassCount);
        Assert.Equal(ClassBalanceLevel.SeverelyImbalanced, result.BalanceLevel);
        Assert.Contains("Only 1 class found", result.Summary);
        Assert.Contains(result.Warnings, w => w.Contains("only one unique value"));
        Assert.Contains(result.Suggestions, s => s.Contains("correct label column"));
    }
}
