using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class CsvStreamBuilderTests
{
    private static InputSchemaInfo CreateSchema(params (string name, string dataType, string purpose)[] cols)
    {
        return new InputSchemaInfo
        {
            Columns = cols.Select(c => new ColumnSchema
            {
                Name = c.name, DataType = c.dataType, Purpose = c.purpose
            }).ToList(),
            CapturedAt = DateTime.UtcNow
        };
    }

    private static string ReadStream(MemoryStream stream)
    {
        stream.Position = 0;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    [Fact]
    public void Build_SimpleNumericRows_ProducesValidCsv()
    {
        var schema = CreateSchema(
            ("Temperature", "Single", "Feature"),
            ("Pressure", "Single", "Feature"));

        var rows = new[]
        {
            new Dictionary<string, object> { ["Temperature"] = 23.5, ["Pressure"] = 101.3 },
            new Dictionary<string, object> { ["Temperature"] = 25.0, ["Pressure"] = 99.8 }
        };

        using var stream = CsvStreamBuilder.Build(rows, schema);
        var csv = ReadStream(stream);
        var lines = csv.TrimEnd().Split('\n').Select(l => l.TrimEnd('\r')).ToArray();

        Assert.Equal(3, lines.Length);
        Assert.Equal("Temperature,Pressure", lines[0]);
        Assert.Equal("23.5,101.3", lines[1]);
        Assert.Equal("25,99.8", lines[2]);
    }

    [Fact]
    public void Build_ExcludedColumns_AreOmitted()
    {
        var schema = CreateSchema(
            ("Sensor1", "Single", "Feature"),
            ("Timestamp", "DateTime", "Exclude"),
            ("Sensor2", "Single", "Feature"));

        var rows = new[]
        {
            new Dictionary<string, object>
            {
                ["Sensor1"] = 1.0,
                ["Timestamp"] = "2026-01-01",
                ["Sensor2"] = 2.0
            }
        };

        using var stream = CsvStreamBuilder.Build(rows, schema);
        var csv = ReadStream(stream);
        var lines = csv.TrimEnd().Split('\n').Select(l => l.TrimEnd('\r')).ToArray();

        Assert.Equal("Sensor1,Sensor2", lines[0]);
        Assert.DoesNotContain("Timestamp", csv);
    }

    [Fact]
    public void Build_CommaInValue_QuotedCorrectly()
    {
        var schema = CreateSchema(
            ("Name", "String", "Feature"),
            ("Value", "Single", "Feature"));

        var rows = new[]
        {
            new Dictionary<string, object> { ["Name"] = "Hello, World", ["Value"] = 42 }
        };

        using var stream = CsvStreamBuilder.Build(rows, schema);
        var csv = ReadStream(stream);
        var lines = csv.TrimEnd().Split('\n').Select(l => l.TrimEnd('\r')).ToArray();

        Assert.Equal("Name,Value", lines[0]);
        Assert.Equal("\"Hello, World\",42", lines[1]);
    }

    [Fact]
    public void Build_WithDummyLabel_InjectsColumn()
    {
        var schema = CreateSchema(
            ("Feature1", "Single", "Feature"),
            ("Quality", "String", "Label"));

        var rows = new[]
        {
            new Dictionary<string, object> { ["Feature1"] = 10.0 }
        };

        using var stream = CsvStreamBuilder.Build(rows, schema, injectDummyLabel: true);
        var csv = ReadStream(stream);
        var lines = csv.TrimEnd().Split('\n').Select(l => l.TrimEnd('\r')).ToArray();

        Assert.Equal("Feature1,Quality", lines[0]);
        Assert.Equal("10,0", lines[1]);
    }

    [Fact]
    public void Build_MissingFeature_UsesEmptyString()
    {
        var schema = CreateSchema(
            ("A", "Single", "Feature"),
            ("B", "Single", "Feature"),
            ("C", "Single", "Feature"));

        var rows = new[]
        {
            new Dictionary<string, object> { ["A"] = 1.0, ["C"] = 3.0 }
        };

        using var stream = CsvStreamBuilder.Build(rows, schema);
        var csv = ReadStream(stream);
        var lines = csv.TrimEnd().Split('\n').Select(l => l.TrimEnd('\r')).ToArray();

        Assert.Equal("A,B,C", lines[0]);
        Assert.Equal("1,,3", lines[1]);
    }
}
