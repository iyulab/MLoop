using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Tests for prediction output CSV format.
/// Validates that prediction output contains only the expected columns without duplication.
/// </summary>
public class PredictionOutputFormatTests : IDisposable
{
    private readonly string _testDir;
    private readonly MLContext _mlContext;

    public PredictionOutputFormatTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-prediction-format-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
        _mlContext = new MLContext(seed: 42);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    /// <summary>
    /// Input data class for binary classification test
    /// </summary>
    public class BinaryClassificationInput
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public float Feature3 { get; set; }
        public bool Label { get; set; }
    }

    /// <summary>
    /// Output class for binary classification prediction
    /// </summary>
    public class BinaryClassificationOutput
    {
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    [Fact]
    public void SaveAsText_WithPredictionColumns_OutputsCleanCsv()
    {
        // Arrange - Create test prediction data
        var predictionData = new[]
        {
            new BinaryClassificationOutput { PredictedLabel = false, Score = -2.5f, Probability = 0.08f },
            new BinaryClassificationOutput { PredictedLabel = true, Score = 1.2f, Probability = 0.77f },
            new BinaryClassificationOutput { PredictedLabel = false, Score = -1.8f, Probability = 0.14f }
        };

        var dataView = _mlContext.Data.LoadFromEnumerable(predictionData);
        var outputPath = Path.Combine(_testDir, "predictions.csv");

        // Act - Save with schema: false for clean output
        using (var fileStream = File.Create(outputPath))
        {
            _mlContext.Data.SaveAsText(dataView, fileStream, separatorChar: ',', headerRow: true, schema: false);
        }

        // Assert - Read and verify the output
        var lines = File.ReadAllLines(outputPath);
        Assert.True(lines.Length >= 4, "Should have header + 3 data rows");

        // Verify header format (should be clean column names without schema info)
        var header = lines[0];
        Assert.Contains("PredictedLabel", header);
        Assert.Contains("Score", header);
        Assert.Contains("Probability", header);

        // Verify no duplicate columns
        var headerColumns = header.Split(',');
        Assert.Equal(3, headerColumns.Length);

        // Verify data rows
        var dataRow1 = lines[1].Split(',');
        Assert.Equal(3, dataRow1.Length); // Should have exactly 3 values
    }

    [Fact]
    public void SelectColumns_OnlyOutputsPredictionColumns()
    {
        // Arrange - Create combined input+output data (simulating ML.NET Transform output)
        var combinedData = new[]
        {
            new { Feature1 = 1.0f, Feature2 = 2.0f, Feature3 = 3.0f, PredictedLabel = false, Score = -2.5f, Probability = 0.08f },
            new { Feature1 = 4.0f, Feature2 = 5.0f, Feature3 = 6.0f, PredictedLabel = true, Score = 1.2f, Probability = 0.77f }
        };

        var dataView = _mlContext.Data.LoadFromEnumerable(combinedData);

        // Act - Select only prediction columns (mimics PredictionEngine behavior)
        var predictionColumns = new List<string>();
        foreach (var col in dataView.Schema)
        {
            if (col.Name == "PredictedLabel" || col.Name == "Score" || col.Name == "Probability")
            {
                predictionColumns.Add(col.Name);
            }
        }

        var outputData = _mlContext.Transforms.SelectColumns(predictionColumns.ToArray())
            .Fit(dataView)
            .Transform(dataView);

        var outputPath = Path.Combine(_testDir, "filtered_predictions.csv");
        using (var fileStream = File.Create(outputPath))
        {
            _mlContext.Data.SaveAsText(outputData, fileStream, separatorChar: ',', headerRow: true, schema: false);
        }

        // Assert
        var lines = File.ReadAllLines(outputPath);
        var header = lines[0];
        var headerColumns = header.Split(',');

        // Should only have 3 prediction columns, not 6 (3 features + 3 predictions)
        Assert.Equal(3, headerColumns.Length);
        Assert.DoesNotContain("Feature1", header);
        Assert.DoesNotContain("Feature2", header);
        Assert.DoesNotContain("Feature3", header);
        Assert.Contains("PredictedLabel", header);
        Assert.Contains("Score", header);
        Assert.Contains("Probability", header);
    }

    [Fact]
    public void PredictionOutput_NoFeatureDuplication()
    {
        // Arrange - Create data that simulates the old buggy output (features appearing twice)
        var inputFeatures = new[]
        {
            new { pH_mean = 10.35f, pH_std = 0.44f, Temp_mean = 44.68f },
            new { pH_mean = 10.27f, pH_std = 0.42f, Temp_mean = 45.35f }
        };

        var inputData = _mlContext.Data.LoadFromEnumerable(inputFeatures);

        // Simulate prediction output by adding prediction columns
        var predictionOutput = _mlContext.Transforms
            .Expression("PredictedLabel", "(x) => false", "pH_mean")
            .Append(_mlContext.Transforms.Expression("Score", "(x) => x * -0.5f", "pH_mean"))
            .Append(_mlContext.Transforms.Expression("Probability", "(x) => 0.1f", "pH_mean"))
            .Fit(inputData)
            .Transform(inputData);

        // Select only prediction columns
        var predictionOnlyData = _mlContext.Transforms.SelectColumns("PredictedLabel", "Score", "Probability")
            .Fit(predictionOutput)
            .Transform(predictionOutput);

        var outputPath = Path.Combine(_testDir, "no_duplicate.csv");
        using (var fileStream = File.Create(outputPath))
        {
            _mlContext.Data.SaveAsText(predictionOnlyData, fileStream, separatorChar: ',', headerRow: true, schema: false);
        }

        // Assert
        var lines = File.ReadAllLines(outputPath);
        var header = lines[0];

        // Should NOT contain any feature columns
        Assert.DoesNotContain("pH_mean", header);
        Assert.DoesNotContain("pH_std", header);
        Assert.DoesNotContain("Temp_mean", header);

        // Should only have prediction columns
        var headerColumns = header.Split(',');
        Assert.Equal(3, headerColumns.Length);

        // Verify data row has exactly 3 values
        if (lines.Length > 1)
        {
            var dataRow = lines[1].Split(',');
            Assert.Equal(3, dataRow.Length);
        }
    }

    [Fact]
    public void RegressionOutput_ContainsScoreOnly()
    {
        // Arrange - Regression prediction only has Score column
        var regressionOutput = new[]
        {
            new { Score = 42.5f },
            new { Score = 38.2f },
            new { Score = 55.1f }
        };

        var dataView = _mlContext.Data.LoadFromEnumerable(regressionOutput);
        var outputPath = Path.Combine(_testDir, "regression_predictions.csv");

        // Act
        using (var fileStream = File.Create(outputPath))
        {
            _mlContext.Data.SaveAsText(dataView, fileStream, separatorChar: ',', headerRow: true, schema: false);
        }

        // Assert
        var lines = File.ReadAllLines(outputPath);
        var header = lines[0];

        Assert.Contains("Score", header);
        Assert.DoesNotContain("PredictedLabel", header);
        Assert.DoesNotContain("Probability", header);

        var headerColumns = header.Split(',');
        Assert.Single(headerColumns);
    }
}
