using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tests;

public class PreprocessingScriptGeneratorTests
{
    private readonly PreprocessingScriptGenerator _generator = new();

    [Fact]
    public void GenerateScripts_CleanData_GeneratesMinimalScripts()
    {
        // Arrange - Clean data with no quality issues
        var report = CreateCleanReport();

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Scripts);
        // Should have at least categorical encoding and numeric normalization
        Assert.True(result.Scripts.Count >= 2);
        Assert.Contains(result.Scripts, s => s.Name == "encode_categorical");
        Assert.Contains(result.Scripts, s => s.Name == "normalize_numeric");
    }

    [Fact]
    public void GenerateScripts_WithDuplicates_GeneratesRemoveDuplicatesScript()
    {
        // Arrange
        var report = CreateReportWithDuplicates(duplicateCount: 50);

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var duplicateScript = result.Scripts.FirstOrDefault(s => s.Name == "remove_duplicates");
        Assert.NotNull(duplicateScript);
        Assert.Equal(1, duplicateScript.Sequence); // Should be first
        Assert.Contains("RemoveDuplicatesScript", duplicateScript.SourceCode);
        Assert.Contains("IPreprocessingScript", duplicateScript.SourceCode);
    }

    [Fact]
    public void GenerateScripts_WithConstantColumns_GeneratesRemoveConstantScript()
    {
        // Arrange
        var report = CreateReportWithConstantColumns(new[] { "status", "region" });

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var constantScript = result.Scripts.FirstOrDefault(s => s.Name == "remove_constant");
        Assert.NotNull(constantScript);
        Assert.Contains("RemoveConstantColumnsScript", constantScript.SourceCode);
        Assert.Contains("status", constantScript.SourceCode);
        Assert.Contains("region", constantScript.SourceCode);
    }

    [Fact]
    public void GenerateScripts_WithMissingValues_GeneratesHandleMissingScript()
    {
        // Arrange
        var report = CreateReportWithMissingValues(
            numericColumns: new[] { "age", "income" },
            categoricalColumns: new[] { "city" });

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var missingScript = result.Scripts.FirstOrDefault(s => s.Name == "handle_missing");
        Assert.NotNull(missingScript);
        Assert.Contains("HandleMissingValuesScript", missingScript.SourceCode);
        Assert.Contains("age", missingScript.SourceCode);
        Assert.Contains("income", missingScript.SourceCode);
        Assert.Contains("city", missingScript.SourceCode);
        Assert.Contains("median", missingScript.SourceCode, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GenerateScripts_WithOutliers_GeneratesHandleOutliersScript()
    {
        // Arrange
        var report = CreateReportWithOutliers(new[] { "price", "quantity" });

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var outlierScript = result.Scripts.FirstOrDefault(s => s.Name == "handle_outliers");
        Assert.NotNull(outlierScript);
        Assert.Contains("HandleOutliersScript", outlierScript.SourceCode);
        Assert.Contains("price", outlierScript.SourceCode);
        Assert.Contains("quantity", outlierScript.SourceCode);
        Assert.Contains("IQR", outlierScript.SourceCode);
    }

    [Fact]
    public void GenerateScripts_WithCategoricalFeatures_GeneratesEncodingScript()
    {
        // Arrange
        var report = CreateReportWithCategoricalFeatures(
            lowCardinality: new[] { "status", "priority" },
            highCardinality: new[] { "customer_id" });

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var encodingScript = result.Scripts.FirstOrDefault(s => s.Name == "encode_categorical");
        Assert.NotNull(encodingScript);
        Assert.Contains("EncodeCategoricalScript", encodingScript.SourceCode);
        Assert.Contains("OneHotEncode", encodingScript.SourceCode);
        Assert.Contains("LabelEncode", encodingScript.SourceCode);
    }

    [Fact]
    public void GenerateScripts_WithNumericFeatures_GeneratesNormalizationScript()
    {
        // Arrange
        var report = CreateReportWithNumericFeatures(new[] { "age", "income", "score" });

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        var normalizeScript = result.Scripts.FirstOrDefault(s => s.Name == "normalize_numeric");
        Assert.NotNull(normalizeScript);
        Assert.Contains("NormalizeNumericScript", normalizeScript.SourceCode);
        Assert.Contains("age", normalizeScript.SourceCode);
        Assert.Contains("income", normalizeScript.SourceCode);
        Assert.Contains("min-max", normalizeScript.Description, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GenerateScripts_CorrectSequenceOrder()
    {
        // Arrange - Report with all quality issues
        var report = CreateComplexReport();

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert - Verify correct ordering
        var scriptNames = result.Scripts.Select(s => s.Name).ToList();

        // Duplicates should be removed first
        var duplicatesIndex = scriptNames.IndexOf("remove_duplicates");
        Assert.True(duplicatesIndex == 0 || duplicatesIndex == -1);

        // Constants should be removed early
        var constantsIndex = scriptNames.IndexOf("remove_constant");
        var missingIndex = scriptNames.IndexOf("handle_missing");
        if (constantsIndex >= 0 && missingIndex >= 0)
        {
            Assert.True(constantsIndex < missingIndex);
        }

        // Encoding should come after cleaning
        var encodingIndex = scriptNames.IndexOf("encode_categorical");
        if (encodingIndex >= 0 && missingIndex >= 0)
        {
            Assert.True(encodingIndex > missingIndex);
        }

        // Normalization should be last
        var normalizeIndex = scriptNames.IndexOf("normalize_numeric");
        Assert.Equal(result.Scripts.Count - 1, normalizeIndex);
    }

    [Fact]
    public void GenerateScripts_ExcludesTargetColumn()
    {
        // Arrange - Report with categorical target
        var report = CreateReportWithCategoricalTarget();

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert - Target should not be in encoding script
        var encodingScript = result.Scripts.FirstOrDefault(s => s.Name == "encode_categorical");
        if (encodingScript != null)
        {
            Assert.DoesNotContain("Churned", encodingScript.SourceCode); // Target column
        }
    }

    [Fact]
    public void GenerateScripts_IncludesSummary()
    {
        // Arrange
        var report = CreateComplexReport();

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert
        Assert.NotEmpty(result.Summary);
        Assert.Contains("preprocessing scripts", result.Summary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Target column", result.Summary);
        Assert.Contains("Problem type", result.Summary);
    }

    [Fact]
    public async Task SaveScriptsAsync_CreatesFilesInDirectory()
    {
        // Arrange
        var report = CreateCleanReport();
        var result = _generator.GenerateScripts(report);
        var tempDir = Path.Combine(Path.GetTempPath(), $"mloop_test_{Guid.NewGuid()}");

        try
        {
            // Act
            await _generator.SaveScriptsAsync(result, tempDir);

            // Assert
            Assert.True(Directory.Exists(tempDir));
            var files = Directory.GetFiles(tempDir, "*.cs");
            Assert.Equal(result.Scripts.Count, files.Length);

            foreach (var script in result.Scripts)
            {
                var filePath = Path.Combine(tempDir, script.FileName);
                Assert.True(File.Exists(filePath));
                var content = await File.ReadAllTextAsync(filePath);
                Assert.Equal(script.SourceCode, content);
            }
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void GeneratedScripts_ImplementIPreprocessingScript()
    {
        // Arrange
        var report = CreateComplexReport();

        // Act
        var result = _generator.GenerateScripts(report);

        // Assert - All scripts should implement the interface
        foreach (var script in result.Scripts)
        {
            Assert.Contains("IPreprocessingScript", script.SourceCode);
            Assert.Contains("ExecuteAsync", script.SourceCode);
            Assert.Contains("PreprocessContext", script.SourceCode);
        }
    }

    // Helper methods
    private DataAnalysisReport CreateCleanReport()
    {
        return new DataAnalysisReport
        {
            FilePath = "test.csv",
            RowCount = 1000,
            ColumnCount = 4,
            Columns = new List<ColumnAnalysis>
            {
                new() { Name = "target", InferredType = DataType.Categorical,
                       NonNullCount = 1000, NullCount = 0, UniqueCount = 2,
                       CategoricalStats = new CategoricalStatistics
                       {
                           MostFrequentValue = "Yes",
                           MostFrequentCount = 600,
                           ValueCounts = new Dictionary<string, int> { ["Yes"] = 600, ["No"] = 400 }
                       }
                },
                new() { Name = "age", InferredType = DataType.Numeric,
                       NonNullCount = 1000, NullCount = 0, UniqueCount = 50,
                       NumericStats = new NumericStatistics { Mean = 35, Median = 34,
                           StandardDeviation = 10, Variance = 100, Min = 18, Max = 65, Q1 = 28, Q3 = 42 }
                },
                new() { Name = "income", InferredType = DataType.Numeric,
                       NonNullCount = 1000, NullCount = 0, UniqueCount = 800,
                       NumericStats = new NumericStatistics { Mean = 50000, Median = 48000,
                           StandardDeviation = 15000, Variance = 225000000, Min = 20000, Max = 120000, Q1 = 40000, Q3 = 60000 }
                },
                new() { Name = "status", InferredType = DataType.Categorical,
                       NonNullCount = 1000, NullCount = 0, UniqueCount = 3,
                       CategoricalStats = new CategoricalStatistics
                       {
                           MostFrequentValue = "Active",
                           MostFrequentCount = 700,
                           ValueCounts = new Dictionary<string, int>
                           {
                               ["Active"] = 700, ["Inactive"] = 200, ["Pending"] = 100
                           }
                       }
                }
            },
            RecommendedTarget = new TargetRecommendation
            {
                ColumnName = "target",
                ProblemType = MLProblemType.BinaryClassification,
                Confidence = 0.9,
                Reason = "Binary target with 2 classes"
            },
            QualityIssues = new DataQualityIssues(),
            MLReadiness = new MLReadinessAssessment
            {
                IsReady = true,
                ReadinessScore = 0.95,
                BlockingIssues = new List<string>(),
                Warnings = new List<string>(),
                Recommendations = new List<string>()
            }
        };
    }

    private DataAnalysisReport CreateReportWithDuplicates(int duplicateCount)
    {
        var report = CreateCleanReport();
        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                DuplicateRowCount = duplicateCount
            },
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithConstantColumns(string[] constantColumns)
    {
        var report = CreateCleanReport();
        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                ConstantColumns = constantColumns.ToList()
            },
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithMissingValues(
        string[] numericColumns,
        string[] categoricalColumns)
    {
        var report = CreateCleanReport();
        var columns = new List<ColumnAnalysis>();

        // Add target column
        columns.Add(report.Columns.First(c => c.Name == "target"));

        // Add numeric columns with missing values
        foreach (var col in numericColumns)
        {
            columns.Add(new ColumnAnalysis
            {
                Name = col,
                InferredType = DataType.Numeric,
                NonNullCount = 900,
                NullCount = 100,
                UniqueCount = 50,
                NumericStats = new NumericStatistics
                {
                    Mean = 35,
                    Median = 34,
                    StandardDeviation = 10,
                    Variance = 100,
                    Min = 18,
                    Max = 65,
                    Q1 = 28,
                    Q3 = 42
                }
            });
        }

        // Add categorical columns with missing values
        foreach (var col in categoricalColumns)
        {
            columns.Add(new ColumnAnalysis
            {
                Name = col,
                InferredType = DataType.Categorical,
                NonNullCount = 950,
                NullCount = 50,
                UniqueCount = 5,
                CategoricalStats = new CategoricalStatistics
                {
                    MostFrequentValue = "Unknown",
                    MostFrequentCount = 300,
                    ValueCounts = new Dictionary<string, int>()
                }
            });
        }

        var missingColumns = numericColumns.Concat(categoricalColumns).ToList();

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = columns.Count,
            Columns = columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                ColumnsWithMissingValues = missingColumns
            },
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithOutliers(string[] outlierColumns)
    {
        var report = CreateCleanReport();
        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                ColumnsWithOutliers = outlierColumns.ToList()
            },
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithCategoricalFeatures(
        string[] lowCardinality,
        string[] highCardinality)
    {
        var report = CreateCleanReport();
        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                HighCardinalityColumns = highCardinality.ToList()
            },
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithNumericFeatures(string[] numericColumns)
    {
        return CreateCleanReport(); // Already has numeric columns
    }

    private DataAnalysisReport CreateReportWithCategoricalTarget()
    {
        var report = CreateCleanReport();
        // Add 'Churned' as the target
        var updatedColumns = report.Columns.ToList();
        updatedColumns[0] = new ColumnAnalysis
        {
            Name = "Churned",
            InferredType = DataType.Categorical,
            NonNullCount = 1000,
            NullCount = 0,
            UniqueCount = 2,
            CategoricalStats = new CategoricalStatistics
            {
                MostFrequentValue = "Yes",
                MostFrequentCount = 600,
                ValueCounts = new Dictionary<string, int> { ["Yes"] = 600, ["No"] = 400 }
            }
        };

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = updatedColumns,
            RecommendedTarget = new TargetRecommendation
            {
                ColumnName = "Churned",
                ProblemType = MLProblemType.BinaryClassification,
                Confidence = 0.9,
                Reason = "Binary target"
            },
            QualityIssues = report.QualityIssues,
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateComplexReport()
    {
        var report = CreateCleanReport();
        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = new DataQualityIssues
            {
                DuplicateRowCount = 25,
                ConstantColumns = new List<string> { "const_col" },
                ColumnsWithMissingValues = new List<string> { "age", "income" },
                ColumnsWithOutliers = new List<string> { "income" },
                HighCardinalityColumns = new List<string>()
            },
            MLReadiness = report.MLReadiness
        };
    }
}
