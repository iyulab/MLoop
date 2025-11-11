using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tests;

public class DataAnalyzerTests
{
    private readonly DataAnalyzer _analyzer = new();

    [Fact]
    public async Task AnalyzeAsync_CSV_WithNumericData_CalculatesStatisticsCorrectly()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"age,income,score
25,50000,85.5
30,60000,90.0
35,70000,88.5
40,80000,92.0
28,55000,87.0");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            Assert.Equal(5, report.RowCount);
            Assert.Equal(3, report.ColumnCount);

            var ageColumn = report.Columns.First(c => c.Name == "age");
            Assert.Equal(DataType.Numeric, ageColumn.InferredType);
            Assert.NotNull(ageColumn.NumericStats);
            Assert.Equal(31.6, ageColumn.NumericStats.Mean, 1);
            Assert.Equal(30, ageColumn.NumericStats.Median);
            Assert.True(ageColumn.NumericStats.StandardDeviation > 0);

            var scoreColumn = report.Columns.First(c => c.Name == "score");
            Assert.Equal(DataType.Numeric, scoreColumn.InferredType);
            Assert.NotNull(scoreColumn.NumericStats);
            Assert.Equal(88.6, scoreColumn.NumericStats.Mean, 1);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithCategoricalData_CalculatesFrequencies()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"category,status
A,active
B,inactive
A,active
C,active
A,inactive");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var categoryColumn = report.Columns.First(c => c.Name == "category");
            Assert.Equal(DataType.Categorical, categoryColumn.InferredType);
            Assert.NotNull(categoryColumn.CategoricalStats);
            Assert.Equal(3, categoryColumn.CategoricalStats.ValueCounts.Count);
            Assert.Equal("A", categoryColumn.CategoricalStats.MostFrequentValue);
            Assert.Equal(3, categoryColumn.CategoricalStats.MostFrequentCount);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithMissingValues_DetectsMissingData()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"name,age,city
Alice,25,NYC
Bob,,LA
Charlie,30,
David,35,Chicago");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var ageColumn = report.Columns.First(c => c.Name == "age");
            Assert.Equal(1, ageColumn.NullCount);
            Assert.Equal(3, ageColumn.NonNullCount);

            var cityColumn = report.Columns.First(c => c.Name == "city");
            Assert.Equal(1, cityColumn.NullCount);
            Assert.Equal(3, cityColumn.NonNullCount);

            Assert.Contains(report.QualityIssues.ColumnsWithMissingValues, col => col == "age");
            Assert.Contains(report.QualityIssues.ColumnsWithMissingValues, col => col == "city");
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithOutliers_DetectsOutliers()
    {
        // Arrange - Create data with clear outliers
        var csvPath = CreateTempCsv(@"value
10
12
11
13
12
11
100
10
12
11");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var valueColumn = report.Columns.First(c => c.Name == "value");
            Assert.Equal(DataType.Numeric, valueColumn.InferredType);
            Assert.NotNull(valueColumn.NumericStats);
            Assert.True(valueColumn.NumericStats.OutlierCount > 0, "Should detect outlier (100)");
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithConstantColumn_DetectsConstant()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"name,status
Alice,active
Bob,active
Charlie,active
David,active");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            Assert.Contains(report.QualityIssues.ConstantColumns, col => col == "status");
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithDuplicateRows_DetectsDuplicates()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"name,age
Alice,25
Bob,30
Alice,25
Charlie,35");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            Assert.True(report.QualityIssues.DuplicateRowCount > 0);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_WithHighCardinality_DetectsHighCardinality()
    {
        // Arrange - Create categorical data with >50 unique values but <10% cardinality ratio
        // Need total rows to be much more than unique values to keep cardinality ratio < 0.1
        var rows = new List<string> { "category" };
        for (int i = 1; i <= 60; i++)
        {
            rows.Add($"Category_{i % 51}"); // 51 unique categories, repeated to reach total rows
        }
        // Add more rows with same categories to reduce cardinality ratio
        for (int i = 0; i < 600; i++)
        {
            rows.Add($"Category_{i % 51}");
        }
        var csvPath = CreateTempCsv(string.Join("\n", rows));

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var categoryColumn = report.Columns.First();
            Assert.NotNull(categoryColumn.CategoricalStats);
            Assert.True(categoryColumn.CategoricalStats.IsHighCardinality);
            Assert.Contains(report.QualityIssues.HighCardinalityColumns, col => col == "category");
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_CSV_RecommendsTargetColumn()
    {
        // Arrange - Create data where binary categorical column comes first
        var csvPath = CreateTempCsv(@"label,feature1,feature2
ClassA,1.5,2.3
ClassB,2.1,3.4
ClassA,1.8,2.9
ClassB,2.4,3.1");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            Assert.NotNull(report.RecommendedTarget);
            // Should recommend label as binary classification target (2 unique values)
            Assert.Equal("label", report.RecommendedTarget.ColumnName);
            Assert.Equal(2, report.Columns.First(c => c.Name == "label").UniqueCount);
            Assert.Equal(MLProblemType.BinaryClassification, report.RecommendedTarget.ProblemType);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_JSON_ParsesCorrectly()
    {
        // Arrange
        var jsonPath = CreateTempJson(@"[
  {""name"": ""Alice"", ""age"": 25, ""city"": ""NYC""},
  {""name"": ""Bob"", ""age"": 30, ""city"": ""LA""},
  {""name"": ""Charlie"", ""age"": 35, ""city"": ""Chicago""}
]");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(jsonPath);

            // Assert
            Assert.Equal(3, report.RowCount);
            Assert.Equal(3, report.ColumnCount);
            Assert.Contains(report.Columns, c => c.Name == "name");
            Assert.Contains(report.Columns, c => c.Name == "age");
            Assert.Contains(report.Columns, c => c.Name == "city");
        }
        finally
        {
            File.Delete(jsonPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_JSONL_ParsesCorrectly()
    {
        // Arrange
        var jsonlPath = CreateTempJsonl(@"{""name"": ""Alice"", ""age"": 25}
{""name"": ""Bob"", ""age"": 30}
{""name"": ""Charlie"", ""age"": 35}");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(jsonlPath);

            // Assert
            Assert.Equal(3, report.RowCount);
            Assert.Equal(2, report.ColumnCount);
        }
        finally
        {
            File.Delete(jsonlPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_InfersBooleanType()
    {
        // Arrange
        var csvPath = CreateTempCsv(@"is_active
true
false
true
true
false");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var column = report.Columns.First();
            Assert.Equal(DataType.Boolean, column.InferredType);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_InfersTextType()
    {
        // Arrange - Create 50+ unique text values to trigger Text type
        var rows = new List<string> { "description" };
        for (int i = 0; i < 60; i++)
        {
            rows.Add($"This is a unique long description number {i} with multiple words");
        }
        var csvPath = CreateTempCsv(string.Join("\n", rows));

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            var column = report.Columns.First();
            Assert.Equal(DataType.Text, column.InferredType);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_AssessesMLReadiness()
    {
        // Arrange - Create data with missing values to trigger recommendations
        var csvPath = CreateTempCsv(@"feature1,feature2,target
1.5,2.3,A
2.1,,B
1.8,2.9,
2.4,3.1,B
,3.5,A");

        try
        {
            // Act
            var report = await _analyzer.AnalyzeAsync(csvPath);

            // Assert
            Assert.NotNull(report.MLReadiness);
            Assert.True(report.MLReadiness.IsReady); // Still ready despite warnings
            Assert.NotEmpty(report.MLReadiness.Warnings);
            Assert.NotEmpty(report.MLReadiness.Recommendations);
        }
        finally
        {
            File.Delete(csvPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_UnsupportedFormat_ThrowsException()
    {
        // Arrange
        var txtPath = Path.GetTempFileName() + ".txt";
        File.WriteAllText(txtPath, "unsupported data");

        try
        {
            // Act & Assert
            await Assert.ThrowsAsync<NotSupportedException>(
                () => _analyzer.AnalyzeAsync(txtPath));
        }
        finally
        {
            File.Delete(txtPath);
        }
    }

    [Fact]
    public async Task AnalyzeAsync_NonExistentFile_ThrowsException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _analyzer.AnalyzeAsync("nonexistent.csv"));
    }

    // Helper methods
    private string CreateTempCsv(string content)
    {
        var path = Path.GetTempFileName() + ".csv";
        File.WriteAllText(path, content);
        return path;
    }

    private string CreateTempJson(string content)
    {
        var path = Path.GetTempFileName() + ".json";
        File.WriteAllText(path, content);
        return path;
    }

    private string CreateTempJsonl(string content)
    {
        var path = Path.GetTempFileName() + ".jsonl";
        File.WriteAllText(path, content);
        return path;
    }
}
