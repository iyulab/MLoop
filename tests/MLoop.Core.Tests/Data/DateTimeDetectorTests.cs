using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class DateTimeDetectorTests
{
    [Theory]
    [InlineData("datetime", true)]
    [InlineData("timestamp", true)]
    [InlineData("date", true)]
    [InlineData("time", true)]
    [InlineData("created_date", true)]
    [InlineData("updated_time", true)]
    [InlineData("date_created", true)]
    [InlineData("time_stamp", true)]
    [InlineData("DateTime", true)]
    [InlineData("TIMESTAMP", true)]
    [InlineData("name", false)]
    [InlineData("value", false)]
    [InlineData("datevalue", false)]
    [InlineData("update", false)]
    public void IsDateTimeColumnName_ReturnsExpected(string columnName, bool expected)
    {
        Assert.Equal(expected, DateTimeDetector.IsDateTimeColumnName(columnName));
    }

    [Fact]
    public void IsDateTimeByValues_WithDateValues_ReturnsTrue()
    {
        var values = new[] { "2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30" };
        Assert.True(DateTimeDetector.IsDateTimeByValues(values));
    }

    [Fact]
    public void IsDateTimeByValues_WithNonDateValues_ReturnsFalse()
    {
        var values = new[] { "hello", "world", "123", "abc" };
        Assert.False(DateTimeDetector.IsDateTimeByValues(values));
    }

    [Fact]
    public void IsDateTimeByValues_Below80Percent_ReturnsFalse()
    {
        var values = new[] { "2024-01-15", "not-a-date", "also-not", "nope", "2024-03-25" };
        // Only 2/5 = 40% are dates
        Assert.False(DateTimeDetector.IsDateTimeByValues(values));
    }

    [Fact]
    public void IsDateTimeByValues_EmptyValues_ReturnsFalse()
    {
        var values = new[] { "", "  ", "" };
        Assert.False(DateTimeDetector.IsDateTimeByValues(values));
    }

    [Fact]
    public void IsDateTimeByValues_WithSomeEmpty_StillDetects()
    {
        var values = new[] { "2024-01-15", "", "2024-03-25", "  ", "2024-05-30" };
        // 3/3 non-empty are dates = 100%
        Assert.True(DateTimeDetector.IsDateTimeByValues(values));
    }

    [Fact]
    public void IsDateTimeColumn_NameMatch_NoValuesNeeded()
    {
        Assert.True(DateTimeDetector.IsDateTimeColumn("created_date"));
        Assert.True(DateTimeDetector.IsDateTimeColumn("created_date", null));
    }

    [Fact]
    public void IsDateTimeColumn_NoNameMatch_UsesValues()
    {
        var dateValues = new[] { "2024-01-15", "2024-02-20" };
        Assert.True(DateTimeDetector.IsDateTimeColumn("some_column", dateValues));
    }

    [Fact]
    public void IsDateTimeColumn_NoNameMatch_NoValues_ReturnsFalse()
    {
        Assert.False(DateTimeDetector.IsDateTimeColumn("some_column"));
    }
}
