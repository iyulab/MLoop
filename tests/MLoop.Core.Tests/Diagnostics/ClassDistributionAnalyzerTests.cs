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
    public async Task ANumericLabelWrittenTwoWays_IsOneClass_AsTheTrainersReadIt()
    {
        var csvPath = Path.Combine(_tempDirectory, "spellings.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,3\n2,3.00\n3,3\n4,3.0");

        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        Assert.Equal(1, result.ClassCount);
        Assert.Equal(4, Assert.Single(result.ClassDistribution).Value);
        Assert.Equal("3", Assert.Single(result.ClassDistribution).Key);   // the spelling it was most often written with
    }

    [Fact]
    public async Task NumericClassesWrittenDifferently_AreCountedByValue()
    {
        var csvPath = Path.Combine(_tempDirectory, "numbers.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,0\n2,1\n3,1.0\n4,0.0\n5,1\n6,");

        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        Assert.Equal(3, result.ClassCount);   // 0, 1, and the empty cell
        Assert.Equal(3, result.ClassDistribution["1"]);
        Assert.Equal(2, result.ClassDistribution["0"]);
        Assert.Equal(1, result.ClassDistribution["(empty)"]);
    }

    [Fact]
    public async Task ALabelWithAValueThatIsNotANumber_StaysText()
    {
        var csvPath = Path.Combine(_tempDirectory, "mixed.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,3\n2,3.00\n3,high");

        var result = await _analyzer.AnalyzeAsync(csvPath, "Label");

        Assert.Equal(3, result.ClassCount);
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
    public void GenerateVisualization_DrawsTheLargestClassesAndCountsTheRest()
    {
        // a label of 50,000 values: every one of them is a class, and the picture is a screenful — not 50,000 lines
        var distribution = Enumerable.Range(0, 50_000).ToDictionary(i => $"value{i:D5}", i => 50_000 - i);

        var picture = ClassDistributionAnalyzer.GenerateVisualization(distribution, distribution.Values.Sum());

        var lines = picture.Split(Environment.NewLine);
        Assert.Equal(ClassDistributionAnalyzer.MaxClassesDrawn + 1, lines.Length);
        Assert.Contains("value00000", lines[0], StringComparison.Ordinal);
        Assert.Contains("… and 49,980 more classes (1 to 49,980 rows each)", lines[^1], StringComparison.Ordinal);
    }

    [Fact]
    public void GenerateVisualization_FewClasses_AreAllDrawnWithNoRemainderLine()
    {
        var picture = ClassDistributionAnalyzer.GenerateVisualization(new Dictionary<string, int> { ["a"] = 3, ["b"] = 1 }, 4);

        Assert.Equal(2, picture.Split(Environment.NewLine).Length);
        Assert.DoesNotContain("more classes", picture, StringComparison.Ordinal);
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
