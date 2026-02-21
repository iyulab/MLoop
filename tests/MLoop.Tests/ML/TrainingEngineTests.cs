using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class TrainingEngineTests : IDisposable
{
    private readonly string _tempDir;

    public TrainingEngineTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-te-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public void Constructor_NullFileSystem_Throws()
    {
        var fs = new FileSystemManager();
        var discovery = new ProjectDiscovery(fs);
        var experimentStore = new ExperimentStore(fs, discovery, _tempDir);

        Assert.Throws<ArgumentNullException>(() =>
            new TrainingEngine(null!, experimentStore));
    }

    [Fact]
    public void Constructor_NullExperimentStore_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TrainingEngine(new FileSystemManager(), null!));
    }

    [Fact]
    public void Constructor_ValidArgs_DoesNotThrow()
    {
        var fs = new FileSystemManager();
        var discovery = new ProjectDiscovery(fs);
        var store = new ExperimentStore(fs, discovery, _tempDir);

        var engine = new TrainingEngine(fs, store);

        Assert.NotNull(engine);
    }

    #region CollectDataStats

    [Fact]
    public void CollectDataStats_BasicCsv_ReturnsCorrectCounts()
    {
        var csv = "Feature1,Feature2,Label\n1,2,A\n3,4,B\n5,6,C\n";
        var path = CreateCsv("basic.csv", csv);

        var (rowCount, colCount, hasText, classCount) = TrainingEngine.CollectDataStats(path, "Label", "multiclass-classification");

        Assert.Equal(3, rowCount);
        Assert.Equal(3, colCount);
        Assert.False(hasText);
        Assert.Equal(3, classCount); // A, B, C
    }

    [Fact]
    public void CollectDataStats_BinaryClassification_CountsClasses()
    {
        var csv = "Feature1,Label\n1,Positive\n2,Negative\n3,Positive\n";
        var path = CreateCsv("binary.csv", csv);

        var (_, _, _, classCount) = TrainingEngine.CollectDataStats(path, "Label", "binary-classification");

        Assert.Equal(2, classCount); // Positive, Negative
    }

    [Fact]
    public void CollectDataStats_Regression_ClassCountIsZero()
    {
        var csv = "Feature1,Label\n1,0.5\n2,1.5\n3,2.5\n";
        var path = CreateCsv("regression.csv", csv);

        var (_, _, _, classCount) = TrainingEngine.CollectDataStats(path, "Label", "regression");

        Assert.Equal(0, classCount); // regression has no classes
    }

    [Fact]
    public void CollectDataStats_TextFeatures_DetectsTextColumn()
    {
        // Long text with spaces in non-label columns triggers hasTextFeatures
        var longText = "This is a very long text description with multiple words";
        var csv = $"Feature1,Label\n\"{longText}\",A\n\"{longText}\",B\n";
        var path = CreateCsv("text.csv", csv);

        var (_, _, hasText, _) = TrainingEngine.CollectDataStats(path, "Label", "binary-classification");

        Assert.True(hasText);
    }

    [Fact]
    public void CollectDataStats_EmptyFile_ReturnsZeros()
    {
        var path = CreateCsv("empty.csv", "");

        var (rowCount, colCount, hasText, classCount) = TrainingEngine.CollectDataStats(path, "Label", "binary-classification");

        Assert.Equal(0, rowCount);
        Assert.Equal(0, colCount);
        Assert.False(hasText);
        Assert.Equal(0, classCount);
    }

    [Fact]
    public void CollectDataStats_HeaderOnly_ReturnsZeroRows()
    {
        var path = CreateCsv("header-only.csv", "Feature1,Feature2,Label\n");

        var (rowCount, colCount, _, _) = TrainingEngine.CollectDataStats(path, "Label", "binary-classification");

        Assert.Equal(0, rowCount);
        Assert.Equal(3, colCount);
    }

    #endregion

    #region GetPrimaryMetricValue

    [Fact]
    public void GetPrimaryMetricValue_ExactMatch_ReturnsValue()
    {
        var metrics = new Dictionary<string, double> { { "accuracy", 0.92 } };

        var result = TrainingEngine.GetPrimaryMetricValue(metrics, "accuracy", "binary-classification");

        Assert.Equal(0.92, result);
    }

    [Fact]
    public void GetPrimaryMetricValue_BinaryFallback_ReturnsAccuracy()
    {
        var metrics = new Dictionary<string, double> { { "accuracy", 0.88 } };

        var result = TrainingEngine.GetPrimaryMetricValue(metrics, "nonexistent", "binary-classification");

        Assert.Equal(0.88, result);
    }

    [Fact]
    public void GetPrimaryMetricValue_MulticlassFallback_ReturnsMacroAccuracy()
    {
        var metrics = new Dictionary<string, double> { { "macro_accuracy", 0.75 } };

        var result = TrainingEngine.GetPrimaryMetricValue(metrics, "nonexistent", "multiclass-classification");

        Assert.Equal(0.75, result);
    }

    [Fact]
    public void GetPrimaryMetricValue_RegressionFallback_ReturnsRSquared()
    {
        var metrics = new Dictionary<string, double> { { "r_squared", 0.82 } };

        var result = TrainingEngine.GetPrimaryMetricValue(metrics, "nonexistent", "regression");

        Assert.Equal(0.82, result);
    }

    [Fact]
    public void GetPrimaryMetricValue_MissingMetric_ReturnsZero()
    {
        var metrics = new Dictionary<string, double> { { "rmse", 0.5 } };

        var result = TrainingEngine.GetPrimaryMetricValue(metrics, "nonexistent", "binary-classification");

        Assert.Equal(0.0, result);
    }

    #endregion

    private string CreateCsv(string fileName, string content)
    {
        var path = Path.Combine(_tempDir, fileName);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }
}
