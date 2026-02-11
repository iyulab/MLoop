using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class CategoricalMapperTests : IDisposable
{
    private readonly CategoricalMapper _mapper = new();
    private readonly string _tempDir;

    public CategoricalMapperTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-cm-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string CreateCsv(string content)
    {
        var path = Path.Combine(_tempDir, $"data_{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }

    private static InputSchemaInfo CreateSchema(params ColumnSchema[] columns)
    {
        return new InputSchemaInfo
        {
            Columns = columns.ToList(),
            CapturedAt = DateTime.UtcNow
        };
    }

    private static ColumnSchema CategoricalFeature(string name, params string[] values)
    {
        return new ColumnSchema
        {
            Name = name,
            DataType = "Categorical",
            Purpose = "Feature",
            CategoricalValues = values.ToList(),
            UniqueValueCount = values.Length
        };
    }

    private static ColumnSchema NumericFeature(string name)
    {
        return new ColumnSchema
        {
            Name = name,
            DataType = "Numeric",
            Purpose = "Feature"
        };
    }

    #region EmptyFile

    [Fact]
    public void PreprocessPredictionData_EmptyFile_ReturnsFailure()
    {
        var path = CreateCsv("");
        var schema = CreateSchema(CategoricalFeature("Color", "Red", "Blue"));

        var result = _mapper.PreprocessPredictionData(path, schema);

        Assert.False(result.Success);
        Assert.Contains("empty", result.ErrorMessage!, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    #region NoCategoricalColumns

    [Fact]
    public void PreprocessPredictionData_NoCategoricalColumns_ReturnsOriginalPath()
    {
        var csv = "Feature1,Feature2\n1.0,2.0\n3.0,4.0\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(NumericFeature("Feature1"), NumericFeature("Feature2"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.UseMostFrequent);

        Assert.True(result.Success);
        Assert.Equal(path, result.TempFilePath);
    }

    #endregion

    #region KnownValues

    [Fact]
    public void PreprocessPredictionData_AllKnownValues_Success()
    {
        var csv = "Color,Size\nRed,Small\nBlue,Large\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue", "Green"),
            CategoricalFeature("Size", "Small", "Medium", "Large"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.UseMostFrequent);

        Assert.True(result.Success);
        Assert.Empty(result.UnknownValues);
    }

    #endregion

    #region UnknownValues - Error Strategy

    [Fact]
    public void PreprocessPredictionData_UnknownValues_ErrorStrategy_ReturnsFailure()
    {
        var csv = "Color,Value\nYellow,10\nRed,20\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.Error);

        Assert.False(result.Success);
        Assert.NotEmpty(result.UnknownValues);
        Assert.Contains("Yellow", result.UnknownValues[0]);
    }

    #endregion

    #region UnknownValues - UseMostFrequent

    [Fact]
    public void PreprocessPredictionData_UnknownValues_UseMostFrequent_Replaces()
    {
        var csv = "Color,Value\nYellow,10\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.UseMostFrequent);

        Assert.True(result.Success);
        Assert.NotNull(result.TempFilePath);

        // Verify the temp file has the replacement
        var lines = File.ReadAllLines(result.TempFilePath!);
        Assert.Equal(2, lines.Length); // header + 1 data row
        Assert.Contains("Red", lines[1]); // Replaced with first categorical value
    }

    #endregion

    #region UnknownValues - UseMissing

    [Fact]
    public void PreprocessPredictionData_UnknownValues_UseMissing_ClearsValue()
    {
        var csv = "Color,Value\nUnknown,10\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.UseMissing);

        Assert.True(result.Success);
        Assert.NotNull(result.TempFilePath);
    }

    #endregion

    #region Auto Strategy

    [Fact]
    public void PreprocessPredictionData_AutoStrategy_LowUnknownRatio_UsesMostFrequent()
    {
        // 1 unknown out of 21 ≈ 4.76% (< 5%) → UseMostFrequent
        var lines = new List<string> { "Color,Value" };
        for (int i = 0; i < 20; i++) lines.Add("Red,10");
        lines.Add("Yellow,20"); // 1 unknown out of 21 categorical values
        var path = CreateCsv(string.Join("\n", lines));
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.Auto);

        Assert.True(result.Success);
        Assert.Equal(CategoricalMapper.UnknownValueStrategy.UseMostFrequent, result.AppliedStrategy);
    }

    [Fact]
    public void PreprocessPredictionData_AutoStrategy_HighUnknownRatio_ReturnsError()
    {
        // All unknown = 100% → Error
        var lines = new List<string> { "Color,Value" };
        for (int i = 0; i < 10; i++) lines.Add("Unknown,10");
        var path = CreateCsv(string.Join("\n", lines));
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.Auto);

        Assert.False(result.Success);
        Assert.Equal(CategoricalMapper.UnknownValueStrategy.Error, result.AppliedStrategy);
    }

    #endregion

    #region ValidateCategoricalValues

    [Fact]
    public void ValidateCategoricalValues_AllValid_ReturnsSuccess()
    {
        var csv = "Color,Value\nRed,10\nBlue,20\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.ValidateCategoricalValues(path, schema);

        Assert.True(result.Success);
    }

    [Fact]
    public void ValidateCategoricalValues_HasUnknown_ReturnsFailure()
    {
        var csv = "Color,Value\nPurple,10\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            NumericFeature("Value"));

        var result = _mapper.ValidateCategoricalValues(path, schema);

        Assert.False(result.Success);
    }

    #endregion

    #region UnknownValuesByColumn Tracking

    [Fact]
    public void PreprocessPredictionData_TracksUnknownsByColumn()
    {
        var csv = "Color,Shape\nYellow,Star\nPurple,Circle\n";
        var path = CreateCsv(csv);
        var schema = CreateSchema(
            CategoricalFeature("Color", "Red", "Blue"),
            CategoricalFeature("Shape", "Circle", "Square"));

        var result = _mapper.PreprocessPredictionData(path, schema,
            CategoricalMapper.UnknownValueStrategy.Error);

        Assert.False(result.Success);
        Assert.True(result.UnknownValuesByColumn.ContainsKey("Color"));
        Assert.True(result.UnknownValuesByColumn.ContainsKey("Shape"));
        Assert.Contains("Yellow", result.UnknownValuesByColumn["Color"]);
        Assert.Contains("Star", result.UnknownValuesByColumn["Shape"]);
    }

    #endregion
}
