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

    #region LooksLikeText

    [Fact]
    public void LooksLikeText_HighUniqueRatio_ReturnsTrue()
    {
        // 10 unique values in 10 lines = 100% unique ratio (> 50%)
        var lines = Enumerable.Range(0, 10)
            .Select(i => $"value_{i}")
            .ToArray();

        Assert.True(TrainingEngine.LooksLikeText(0, lines, 10));
    }

    [Fact]
    public void LooksLikeText_ShortCategoricalCodes_ReturnsFalse()
    {
        // Short codes like OK/NG/HIGH/LOW — few unique, short strings, single token
        var values = new[] { "OK", "NG", "OK", "HIGH", "LOW", "OK", "NG", "OK", "LOW", "OK" };
        var lines = values.Select(v => v).ToArray();

        Assert.False(TrainingEngine.LooksLikeText(0, lines, 4));
    }

    [Fact]
    public void LooksLikeText_LongLogMessages_ReturnsTrue()
    {
        // Log messages with 3+ tokens and long strings — should be text
        var lines = new[]
        {
            "\"Application started successfully on port 8080\",INFO",
            "\"Database connection pool initialized with 20 connections\",INFO",
            "\"User authentication failed for admin@example.com\",ERROR",
            "\"Memory usage threshold exceeded at 95 percent\",WARNING",
            "\"Application started successfully on port 8080\",INFO",
            "\"Database connection pool initialized with 20 connections\",INFO",
            "\"Cache invalidation completed for session store\",INFO",
            "\"Application started successfully on port 8080\",INFO",
        };

        // 5 unique values in 8 lines = 62.5% unique — but even with lower unique ratio
        // the text heuristic (avgTokens >= 3) should catch it
        Assert.True(TrainingEngine.LooksLikeText(0, lines, 5));
    }

    [Fact]
    public void LooksLikeText_RepeatedLogMessages_LowUniqueRatio_ReturnsTrue()
    {
        // Simulates sim-17 scenario: log messages with low unique ratio (~32%)
        // but still text-like (long strings, multiple tokens)
        var templates = new[]
        {
            "[arp_up] _> start /clusterplex/script/lin/mysql/arp_up.sh",
            "[arp_up] ... ARPING 10.68.40.1 from 10.68.40.173 ens192",
            "[Data_Mirror] Synchronization completed successfully"
        };
        var lines = new string[20];
        for (int i = 0; i < 20; i++)
            lines[i] = templates[i % templates.Length];

        // 3 unique in 20 = 15% unique ratio — below 50% threshold
        // but avgTokens >= 3 should trigger text classification
        Assert.True(TrainingEngine.LooksLikeText(0, lines, 3));
    }

    [Fact]
    public void LooksLikeText_NumericIds_ReturnsFalse()
    {
        // Numeric-like IDs that happen to be strings
        var lines = new[] { "001", "002", "001", "003", "002", "001", "003", "001", "002", "001" };

        Assert.False(TrainingEngine.LooksLikeText(0, lines, 3));
    }

    [Fact]
    public void LooksLikeText_EmptyLines_ReturnsFalse()
    {
        Assert.False(TrainingEngine.LooksLikeText(0, Array.Empty<string>(), 0));
    }

    [Fact]
    public void LooksLikeText_HighCardinality_ModerateUniqueRatio_ReturnsTrue()
    {
        // 250 unique values in 1000 lines = 25% unique ratio
        // uniqueCount > 200 AND uniqueRatio > 10% → criterion 4
        var lines = new string[1000];
        for (int i = 0; i < 1000; i++)
            lines[i] = $"log entry {i % 250}";

        Assert.True(TrainingEngine.LooksLikeText(0, lines, 250));
    }

    #endregion

    private string CreateCsv(string fileName, string content)
    {
        var path = Path.Combine(_tempDir, fileName);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }
}
