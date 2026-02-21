using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Tests for DataBalancer class that handles minority class oversampling
/// </summary>
public class DataBalancerTests : IDisposable
{
    private readonly string _testDir;

    public DataBalancerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-balancer-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Balance_WithHighImbalance_OversamplesMinorityClass()
    {
        // Arrange - Create imbalanced dataset (50:1)
        var dataFile = CreateImbalancedDataset(50, 1);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", "auto");

        // Assert
        Assert.True(result.Applied);
        Assert.NotNull(result.BalancedFilePath);
        Assert.Equal(50.0, result.OriginalRatio);
        Assert.True(result.NewRatio <= DataBalancer.DefaultTargetRatio);
        Assert.Contains("Oversampled", result.Message);
    }

    [Fact]
    public void Balance_WithAcceptableImbalance_DoesNotOversample()
    {
        // Arrange - Create mildly imbalanced dataset (5:1)
        var dataFile = CreateImbalancedDataset(50, 10);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", "auto");

        // Assert
        Assert.False(result.Applied);
        Assert.Contains("within acceptable range", result.Message);
    }

    [Fact]
    public void Balance_WithNoneOption_DoesNotOversample()
    {
        // Arrange
        var dataFile = CreateImbalancedDataset(50, 1);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", "none");

        // Assert
        Assert.False(result.Applied);
        Assert.Equal("Balancing disabled", result.Message);
    }

    [Fact]
    public void Balance_WithNullOption_DoesNotOversample()
    {
        // Arrange
        var dataFile = CreateImbalancedDataset(50, 1);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", null);

        // Assert
        Assert.False(result.Applied);
    }

    [Fact]
    public void Balance_WithCustomRatio_UsesSpecifiedRatio()
    {
        // Arrange - Create 20:1 imbalanced dataset
        var dataFile = CreateImbalancedDataset(100, 5);
        var balancer = new DataBalancer();

        // Act - Request 2:1 ratio
        var result = balancer.Balance(dataFile, "Label", "2");

        // Assert
        Assert.True(result.Applied);
        Assert.True(result.NewRatio <= 2.0);
        Assert.True(result.NewMinorityCount >= 50); // 100/2 = 50
    }

    [Fact]
    public void Balance_ProducesValidCsvFile()
    {
        // Arrange
        var dataFile = CreateImbalancedDataset(20, 2);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", "auto");

        // Assert - If balancing was applied, verify CSV structure
        if (result.Applied && result.BalancedFilePath != null)
        {
            var lines = File.ReadAllLines(result.BalancedFilePath);

            // Has header and data
            Assert.True(lines.Length > 1);

            // Header matches original
            var originalHeader = File.ReadAllLines(dataFile)[0];
            Assert.Equal(originalHeader, lines[0]);

            // Has more rows than original (oversampled)
            var originalLines = File.ReadAllLines(dataFile);
            Assert.True(lines.Length >= originalLines.Length);
        }
    }

    [Fact]
    public void Balance_WithMissingLabelColumn_ReturnsError()
    {
        // Arrange
        var dataFile = CreateImbalancedDataset(10, 2);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "NonExistentColumn", "auto");

        // Assert
        Assert.False(result.Applied);
        Assert.Contains("not found", result.Message);
    }

    [Fact]
    public void Balance_WithSingleClass_ReturnsNoBalancingNeeded()
    {
        // Arrange - Create dataset with only one class
        var dataFile = CreateSingleClassDataset(20);
        var balancer = new DataBalancer();

        // Act
        var result = balancer.Balance(dataFile, "Label", "auto");

        // Assert
        Assert.False(result.Applied);
        Assert.Contains("Only one class found", result.Message);
    }

    [Fact]
    public void Balance_WithWhitespaceOption_DoesNotOversample()
    {
        var dataFile = CreateImbalancedDataset(50, 1);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "  ");

        Assert.False(result.Applied);
    }

    [Fact]
    public void Balance_WithInvalidOption_ReturnsError()
    {
        var dataFile = CreateImbalancedDataset(10, 2);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "invalid");

        Assert.False(result.Applied);
        Assert.Contains("Invalid balance option", result.Message);
    }

    [Fact]
    public void Balance_NoDataRows_ReturnsNotApplied()
    {
        var filePath = Path.Combine(_testDir, "empty.csv");
        File.WriteAllText(filePath, "Feature1,Label\n");
        var balancer = new DataBalancer();

        var result = balancer.Balance(filePath, "Label", "auto");

        Assert.False(result.Applied);
        Assert.Contains("No data rows", result.Message);
    }

    [Fact]
    public void Balance_CustomRatio_AlreadyMet_DoesNotApply()
    {
        // 10:5 = 2:1, target ratio 5:1 - already within target
        var dataFile = CreateImbalancedDataset(10, 5);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "5");

        Assert.False(result.Applied);
        Assert.Contains("already meets", result.Message);
    }

    [Fact]
    public void Balance_TracksOriginalAndNewCounts()
    {
        var dataFile = CreateImbalancedDataset(50, 3);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "auto");

        Assert.True(result.Applied);
        Assert.Equal(3, result.OriginalMinorityCount);
        Assert.True(result.NewMinorityCount > 3);
        Assert.True(result.OriginalRatio > result.NewRatio);
    }

    private string CreateImbalancedDataset(int majorityCount, int minorityCount)
    {
        var filePath = Path.Combine(_testDir, $"imbalanced_{majorityCount}_{minorityCount}.csv");
        var lines = new List<string> { "Feature1,Feature2,Feature3,Label" };

        var random = new Random(42);

        // Add majority class
        for (int i = 0; i < majorityCount; i++)
        {
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},{random.NextDouble():F4},0");
        }

        // Add minority class
        for (int i = 0; i < minorityCount; i++)
        {
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},{random.NextDouble():F4},1");
        }

        File.WriteAllLines(filePath, lines);
        return filePath;
    }

    [Fact]
    public void Balance_Multiclass_OversamplesAllMinorityClasses()
    {
        // 100:3:2 → target = ceil(100/10) = 10, so B(3) and C(2) both need oversampling
        var dataFile = CreateMulticlassImbalancedDataset(100, 3, 2);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "auto");

        Assert.True(result.Applied);
        Assert.NotNull(result.BalancedFilePath);

        var lines = File.ReadAllLines(result.BalancedFilePath!);
        var classCounts = lines.Skip(1)
            .Select(l => CsvFieldParser.ParseFields(l))
            .Where(f => f.Length >= 4)
            .GroupBy(f => f[3].Trim())
            .ToDictionary(g => g.Key, g => g.Count());

        // With the bug, only C (smallest) gets oversampled; B stays at 3
        Assert.True(classCounts["B"] >= 10, $"Class B count {classCounts["B"]} should be >= 10 (intermediate minority must also be oversampled)");
        Assert.True(classCounts["C"] >= 10, $"Class C count {classCounts["C"]} should be >= 10");
    }

    [Fact]
    public void Balance_Multiclass_MajorityClassUnchanged()
    {
        var dataFile = CreateMulticlassImbalancedDataset(100, 3, 2);
        var balancer = new DataBalancer();

        var result = balancer.Balance(dataFile, "Label", "auto");

        Assert.True(result.Applied);
        var lines = File.ReadAllLines(result.BalancedFilePath!);
        var classCounts = lines.Skip(1)
            .Select(l => CsvFieldParser.ParseFields(l))
            .Where(f => f.Length >= 4)
            .GroupBy(f => f[3].Trim())
            .ToDictionary(g => g.Key, g => g.Count());

        Assert.Equal(100, classCounts["A"]);
    }

    private string CreateMulticlassImbalancedDataset(int classACount, int classBCount, int classCCount)
    {
        var filePath = Path.Combine(_testDir, $"multiclass_{classACount}_{classBCount}_{classCCount}.csv");
        var lines = new List<string> { "Feature1,Feature2,Feature3,Label" };
        var random = new Random(42);

        for (int i = 0; i < classACount; i++)
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},{random.NextDouble():F4},A");
        for (int i = 0; i < classBCount; i++)
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},{random.NextDouble():F4},B");
        for (int i = 0; i < classCCount; i++)
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},{random.NextDouble():F4},C");

        File.WriteAllLines(filePath, lines);
        return filePath;
    }

    private string CreateSingleClassDataset(int count)
    {
        var filePath = Path.Combine(_testDir, $"single_class_{count}.csv");
        var lines = new List<string> { "Feature1,Feature2,Label" };

        var random = new Random(42);
        for (int i = 0; i < count; i++)
        {
            lines.Add($"{random.NextDouble():F4},{random.NextDouble():F4},0");
        }

        File.WriteAllLines(filePath, lines);
        return filePath;
    }
}
