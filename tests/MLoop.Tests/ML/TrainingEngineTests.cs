using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Runtime;

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
    public void CollectDataStats_MinorityClassBeyondFirst1000Rows_CountsAllClasses()
    {
        // D2: sorted data whose minority class appears only after row 1000. Class counting
        // must scan the full file, not just the first 1000 rows — otherwise a binary label is
        // misdetected as single-class (Label=["OK"]), corrupting time estimation and the
        // promotion quality gate's 1/N threshold.
        var sb = new System.Text.StringBuilder();
        sb.Append("Feature1,Label\n");
        for (int i = 0; i < 1200; i++) sb.Append($"{i},OK\n");
        for (int i = 0; i < 30; i++) sb.Append($"{i},NG\n");
        var path = CreateCsv("sorted-tail-minority.csv", sb.ToString());

        var (rowCount, _, _, classCount) = TrainingEngine.CollectDataStats(path, "Label", "binary-classification");

        Assert.Equal(1230, rowCount);
        Assert.Equal(2, classCount); // OK + NG — NG must not be missed despite being past row 1000
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

    #region CollectCompleteCategoricalValues

    [Fact]
    public void CollectCompleteCategoricalValues_LabelColumn_ScansFullFileNotHeadSample()
    {
        // D2: the label's categorical values/count must reflect the full file. The 1000-row
        // head sample sees only "OK", so the label schema is seeded single-class; the complete
        // scan must correct it to {OK, NG} once "NG" appears later in sorted data — otherwise
        // the promotion quality gate derives a wrong 1/N threshold from UniqueValueCount.
        var sb = new System.Text.StringBuilder();
        sb.Append("Feature1,Label\n");
        for (int i = 0; i < 1200; i++) sb.Append($"{i},OK\n");
        for (int i = 0; i < 30; i++) sb.Append($"{i},NG\n");
        var path = CreateCsv("label-complete.csv", sb.ToString());

        var columns = new List<ColumnSchema>
        {
            new() { Name = "Feature1", DataType = "Numeric", Purpose = "Feature" },
            // Simulates the head-sample result: label seen as single-class "OK".
            new() { Name = "Label", DataType = "Categorical", Purpose = "Label",
                    CategoricalValues = new List<string> { "OK" }, UniqueValueCount = 1 },
        };

        TrainingEngine.CollectCompleteCategoricalValues(path, columns, new[] { "Feature1", "Label" });

        var label = columns.Single(c => c.Name == "Label");
        Assert.Equal(2, label.UniqueValueCount);
        Assert.Equal(new[] { "NG", "OK" }, label.CategoricalValues);
    }

    [Fact]
    public void CollectCompleteCategoricalValues_FeatureColumn_StillScannedFully()
    {
        // Regression guard: extending the scan to the label must not stop it covering
        // categorical feature columns (the original BUG-R2-06 behavior).
        var sb = new System.Text.StringBuilder();
        sb.Append("Zone,Label\n");
        for (int i = 0; i < 1200; i++) sb.Append($"A,OK\n");
        sb.Append("B,NG\n");
        var path = CreateCsv("feature-complete.csv", sb.ToString());

        var columns = new List<ColumnSchema>
        {
            new() { Name = "Zone", DataType = "Categorical", Purpose = "Feature",
                    CategoricalValues = new List<string> { "A" }, UniqueValueCount = 1 },
            new() { Name = "Label", DataType = "Categorical", Purpose = "Label",
                    CategoricalValues = new List<string> { "OK" }, UniqueValueCount = 1 },
        };

        TrainingEngine.CollectCompleteCategoricalValues(path, columns, new[] { "Zone", "Label" });

        var zone = columns.Single(c => c.Name == "Zone");
        Assert.Equal(2, zone.UniqueValueCount);
        Assert.Equal(new[] { "A", "B" }, zone.CategoricalValues);
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

    #region Image classification (directory bypass)

    [Fact]
    public async Task TrainAsync_ImageClassification_BypassesCsvAndReachesRuntimeGate()
    {
        // With the TF runtime installed, training would actually run (slow) — skip then.
        var runtime = RuntimeRegistry.GetRequiredByTask("image-classification");
        if (runtime != null && new RuntimeManager().IsInstalled(runtime))
            return;

        var imageDir = Path.Combine(_tempDir, "images");
        CreateImageClass(imageDir, "OK", 4);
        CreateImageClass(imageDir, "NG", 4);

        var engine = NewEngine();
        var config = new TrainingConfig
        {
            ModelName = "img",
            DataFile = imageDir,
            LabelColumn = "Label",
            Task = "image-classification",
            TimeLimitSeconds = 5
        };

        // Reaching the runtime gate proves the CSV preprocessing pipeline was bypassed
        // and the image directory loaded successfully, rather than failing on CSV parsing.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.TrainAsync(config));

        Assert.Contains("runtime", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("install", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("CSV", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TrainAsync_ImageClassification_MissingDirectory_FailsWithDirectoryMessage()
    {
        var engine = NewEngine();
        var config = new TrainingConfig
        {
            ModelName = "img",
            DataFile = Path.Combine(_tempDir, "no-such-image-dir"),
            LabelColumn = "Label",
            Task = "image-classification",
            TimeLimitSeconds = 5
        };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.TrainAsync(config));

        Assert.Contains("directory", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BuildDirectoryInputSchema_ImageClassification_UsesCanonicalVocabulary()
    {
        // BUG-42: the directory schema must use MLoop's canonical dataType vocabulary
        // (Categorical/Text/Numeric/Boolean), NOT the raw .NET "String". A classification label
        // typed "String" falls through PredictionEngine's type-override default and is read as
        // Single, which then breaks the model's MapValueToKey at predict time. The label must be
        // categorical (→ DataKind.String) and the image path a text feature.
        var schema = TrainingEngine.BuildDirectoryInputSchema("Label", "image-classification");

        var imagePath = schema.Columns.Single(c => c.Name == "ImagePath");
        var label = schema.Columns.Single(c => c.Name == "Label");

        Assert.Equal("Text", imagePath.DataType);
        Assert.Equal("Feature", imagePath.Purpose);
        Assert.Equal("Categorical", label.DataType);
        Assert.Equal("Label", label.Purpose);
        Assert.DoesNotContain(schema.Columns, c => c.DataType == "String");
    }

    [Fact]
    public void BuildDirectoryInputSchema_WithClassCount_PopulatesLabelUniqueValueCount()
    {
        // BUG-46: the promotion quality gate derives its 1/N threshold from the label's
        // UniqueValueCount. Directory-based tasks must populate it (folder count) so the gate
        // can reject non-converged image models (micro_accuracy < 1/N) instead of silently
        // promoting them.
        var schema = TrainingEngine.BuildDirectoryInputSchema("Label", "image-classification", classCount: 6);

        var label = schema.Columns.Single(c => c.Name == "Label");
        Assert.Equal(6, label.UniqueValueCount);
    }

    [Fact]
    public void BuildDirectoryInputSchema_WithoutClassCount_LeavesUniqueValueCountNull()
    {
        var schema = TrainingEngine.BuildDirectoryInputSchema("Label", "image-classification");

        var label = schema.Columns.Single(c => c.Name == "Label");
        Assert.Null(label.UniqueValueCount);
    }

    [Fact]
    public void BuildDirectoryInputSchema_ObjectDetection_AddsBoundingBoxVector()
    {
        // Object detection carries a categorical label vector plus a float bounding-box vector.
        var schema = TrainingEngine.BuildDirectoryInputSchema("Label", "object-detection");

        var label = schema.Columns.Single(c => c.Name == "Label");
        var bbox = schema.Columns.Single(c => c.Name == "BoundingBoxes");

        Assert.Equal("Categorical", label.DataType);
        Assert.Equal("Single", bbox.DataType);
        Assert.Equal("Label", bbox.Purpose);
    }

    #endregion

    #region Object detection (COCO directory bypass)

    [Fact]
    public async Task TrainAsync_ObjectDetection_BypassesCsvAndReachesRuntimeGate()
    {
        // With the TorchSharp runtime installed, training would actually run (slow) — skip then.
        var runtime = RuntimeRegistry.GetRequiredByTask("object-detection");
        if (runtime != null && new RuntimeManager().IsInstalled(runtime))
            return;

        var cocoDir = Path.Combine(_tempDir, "coco");
        CreateCocoDataset(cocoDir);

        var engine = NewEngine();
        var config = new TrainingConfig
        {
            ModelName = "od",
            DataFile = cocoDir,
            LabelColumn = "Label",
            Task = "object-detection",
            TimeLimitSeconds = 5
        };

        // Reaching the runtime gate proves the CSV preprocessing pipeline was bypassed and the
        // COCO annotations loaded successfully, rather than failing on CSV parsing.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.TrainAsync(config));

        Assert.Contains("runtime", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("install", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("CSV", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task TrainAsync_ObjectDetection_MissingDirectory_FailsWithDirectoryMessage()
    {
        var engine = NewEngine();
        var config = new TrainingConfig
        {
            ModelName = "od",
            DataFile = Path.Combine(_tempDir, "no-such-coco-dir"),
            LabelColumn = "Label",
            Task = "object-detection",
            TimeLimitSeconds = 5
        };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.TrainAsync(config));

        Assert.Contains("directory", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    private static void CreateCocoDataset(string dir)
    {
        Directory.CreateDirectory(dir);
        for (var i = 1; i <= 4; i++)
            File.WriteAllBytes(Path.Combine(dir, $"img{i}.jpg"), new byte[] { 0xFF, 0xD8, 0xFF });

        var images = string.Join(",", Enumerable.Range(1, 4).Select(i => $"{{\"id\":{i},\"file_name\":\"img{i}.jpg\"}}"));
        var annotations = string.Join(",", Enumerable.Range(1, 4).Select(i => $"{{\"image_id\":{i},\"category_id\":{(i % 2) + 1},\"bbox\":[0,0,10,10]}}"));
        var json = $"{{\"images\":[{images}],\"annotations\":[{annotations}]," +
                   "\"categories\":[{\"id\":1,\"name\":\"car\"},{\"id\":2,\"name\":\"person\"}]}";
        File.WriteAllText(Path.Combine(dir, "annotations.json"), json);
    }

    private TrainingEngine NewEngine()
    {
        var fs = new FileSystemManager();
        var discovery = new ProjectDiscovery(fs);
        var store = new ExperimentStore(fs, discovery, _tempDir);
        return new TrainingEngine(fs, store);
    }

    private static void CreateImageClass(string root, string label, int count)
    {
        var dir = Path.Combine(root, label);
        Directory.CreateDirectory(dir);
        for (var i = 0; i < count; i++)
            File.WriteAllBytes(Path.Combine(dir, $"{label}_{i}.jpg"), new byte[] { 0xFF, 0xD8, 0xFF });
    }

    private string CreateCsv(string fileName, string content)
    {
        var path = Path.Combine(_tempDir, fileName);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }
}
