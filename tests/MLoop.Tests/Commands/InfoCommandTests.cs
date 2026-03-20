using Microsoft.ML.AutoML;
using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class InfoCommandTests : IDisposable
{
    private readonly string _tempDir;

    public InfoCommandTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-info-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    #region GetColumnPurpose

    [Fact]
    public void GetColumnPurpose_LabelColumn_ReturnsLabel()
    {
        var colInfo = new ColumnInformation { LabelColumnName = "target" };

        var result = InfoCommand.GetColumnPurpose("target", colInfo, "Numeric");

        Assert.Contains("Label", result);
    }

    [Fact]
    public void GetColumnPurpose_IgnoredColumn_ReturnsIgnored()
    {
        var colInfo = new ColumnInformation();
        colInfo.IgnoredColumnNames.Add("id");

        var result = InfoCommand.GetColumnPurpose("id", colInfo, "Numeric");

        Assert.Contains("Ignored", result);
    }

    [Fact]
    public void GetColumnPurpose_TextDataType_ReturnsTextFeature()
    {
        var colInfo = new ColumnInformation();

        var result = InfoCommand.GetColumnPurpose("description", colInfo, "Text");

        Assert.Contains("Text Feature", result);
    }

    [Fact]
    public void GetColumnPurpose_CategoricalColumn_ReturnsCategorical()
    {
        var colInfo = new ColumnInformation();
        colInfo.CategoricalColumnNames.Add("color");

        var result = InfoCommand.GetColumnPurpose("color", colInfo, "Text/Categorical");

        Assert.Contains("Categorical Feature", result);
    }

    [Fact]
    public void GetColumnPurpose_NumericColumn_ReturnsNumeric()
    {
        var colInfo = new ColumnInformation();
        colInfo.NumericColumnNames.Add("x1");

        var result = InfoCommand.GetColumnPurpose("x1", colInfo, "Numeric");

        Assert.Contains("Numeric Feature", result);
    }

    [Fact]
    public void GetColumnPurpose_BooleanType_ReturnsNumeric()
    {
        var colInfo = new ColumnInformation();

        var result = InfoCommand.GetColumnPurpose("flag", colInfo, "Boolean");

        Assert.Contains("Numeric Feature", result);
    }

    [Fact]
    public void GetColumnPurpose_UnknownType_ReturnsFeature()
    {
        var colInfo = new ColumnInformation();

        var result = InfoCommand.GetColumnPurpose("unknown", colInfo, "SomeOtherType");

        Assert.Contains("Feature", result);
    }

    #endregion

    #region CalculateColumnStats

    [Fact]
    public void CalculateColumnStats_BasicCsv_CountsMissingValues()
    {
        var csvFile = CreateTempCsv("x1,x2,label\n1.0,2.0,A\n,3.0,B\n4.0,,C\n");
        var columns = new[] { "x1", "x2", "label" };

        var (stats, _) = InfoCommand.CalculateColumnStats(csvFile, columns, null);

        Assert.Equal(1, stats["x1"].MissingCount);
        Assert.Equal(1, stats["x2"].MissingCount);
        Assert.Equal(0, stats["label"].MissingCount);
    }

    [Fact]
    public void CalculateColumnStats_CountsUniqueValues()
    {
        var csvFile = CreateTempCsv("x1,label\n1.0,A\n2.0,B\n1.0,A\n3.0,C\n");
        var columns = new[] { "x1", "label" };

        var (stats, _) = InfoCommand.CalculateColumnStats(csvFile, columns, null);

        Assert.Equal(3, stats["x1"].UniqueCount); // 1.0, 2.0, 3.0
        Assert.Equal(3, stats["label"].UniqueCount); // A, B, C
    }

    [Fact]
    public void CalculateColumnStats_WithLabel_TracksDistribution()
    {
        var csvFile = CreateTempCsv("x1,label\n1.0,OK\n2.0,NG\n3.0,OK\n4.0,OK\n");
        var columns = new[] { "x1", "label" };

        var (_, labelDist) = InfoCommand.CalculateColumnStats(csvFile, columns, "label");

        Assert.NotNull(labelDist);
        Assert.Equal(3, labelDist["OK"]);
        Assert.Equal(1, labelDist["NG"]);
    }

    [Fact]
    public void CalculateColumnStats_NoLabel_ReturnsNullDistribution()
    {
        var csvFile = CreateTempCsv("x1,x2\n1.0,2.0\n3.0,4.0\n");
        var columns = new[] { "x1", "x2" };

        var (_, labelDist) = InfoCommand.CalculateColumnStats(csvFile, columns, null);

        Assert.Null(labelDist);
    }

    [Fact]
    public void CalculateColumnStats_EmptyLabelValues_CountedAsEmpty()
    {
        var csvFile = CreateTempCsv("x1,label\n1.0,OK\n2.0,\n3.0,OK\n");
        var columns = new[] { "x1", "label" };

        var (_, labelDist) = InfoCommand.CalculateColumnStats(csvFile, columns, "label");

        Assert.NotNull(labelDist);
        Assert.Equal(2, labelDist["OK"]);
        Assert.Equal(1, labelDist["(empty)"]);
    }

    [Fact]
    public void CalculateColumnStats_AllMissing_CorrectCount()
    {
        var csvFile = CreateTempCsv("x1,x2\n,\n,\n,\n");
        var columns = new[] { "x1", "x2" };

        var (stats, _) = InfoCommand.CalculateColumnStats(csvFile, columns, null);

        Assert.Equal(3, stats["x1"].MissingCount);
        Assert.Equal(3, stats["x2"].MissingCount);
    }

    [Fact]
    public void CalculateColumnStats_MaxUniqueRows_LimitsUniqueTracking()
    {
        // Create CSV with more rows than maxUniqueRows
        var lines = new System.Text.StringBuilder("x1\n");
        for (int i = 0; i < 15; i++)
            lines.AppendLine(i.ToString());

        var csvFile = CreateTempCsv(lines.ToString());
        var columns = new[] { "x1" };

        // With maxUniqueRows=5, only first 5 unique values tracked
        var (stats, _) = InfoCommand.CalculateColumnStats(csvFile, columns, null, maxUniqueRows: 5);

        Assert.Equal(5, stats["x1"].UniqueCount);
    }

    #endregion

    private string CreateTempCsv(string content)
    {
        var path = Path.Combine(_tempDir, $"test-{Guid.NewGuid()}.csv");
        File.WriteAllText(path, content);
        return path;
    }
}
