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
    [InlineData("STD_DT", true)]
    [InlineData("MFG_DT", true)]
    [InlineData("PROC_DT", true)]
    [InlineData("std_dt", true)]
    [InlineData("name", false)]
    [InlineData("value", false)]
    [InlineData("datevalue", false)]
    [InlineData("update", false)]
    [InlineData("PRODUCT", false)]
    [InlineData("RESULT", false)]
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
    public void IsDateTimeColumn_StrongName_NoValuesNeeded()
    {
        // Strong names (datetime, timestamp) don't need value confirmation
        Assert.True(DateTimeDetector.IsDateTimeColumn("datetime"));
        Assert.True(DateTimeDetector.IsDateTimeColumn("created_timestamp", null));
    }

    [Fact]
    public void IsDateTimeColumn_WeakName_RequiresValueConfirmation()
    {
        // Weak names (_date, _time) require value confirmation
        // to avoid false positives on columns like "Cycle_Time", "Spray_Time"
        Assert.False(DateTimeDetector.IsDateTimeColumn("created_date"));
        Assert.False(DateTimeDetector.IsDateTimeColumn("created_date", null));

        // With DateTime values, weak names return true
        var dateValues = new[] { "2024-01-15", "2024-02-20" };
        Assert.True(DateTimeDetector.IsDateTimeColumn("created_date", dateValues));
    }

    [Fact]
    public void IsDateTimeColumn_WeakName_WithNumericValues_ReturnsFalse()
    {
        // Numeric durations like "Cycle_Time" should NOT be detected as DateTime
        var numericValues = new[] { "20.7", "0.044", "7.8", "0.7" };
        Assert.False(DateTimeDetector.IsDateTimeColumn("Cycle_Time", numericValues));
        Assert.False(DateTimeDetector.IsDateTimeColumn("Spray_Time", numericValues));
        Assert.False(DateTimeDetector.IsDateTimeColumn("Melting_Furnace_Temp", numericValues));
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

    [Fact]
    public void IsDateTimeColumn_DtSuffix_RequiresValueConfirmation()
    {
        // _DT suffix is a weak pattern — requires value confirmation
        Assert.False(DateTimeDetector.IsDateTimeColumn("STD_DT"));
        Assert.False(DateTimeDetector.IsDateTimeColumn("MFG_DT", null));

        // With DateTime values, _DT columns return true
        var dateValues = new[] { "2024-01-15 08:30:00", "2024-02-20 14:15:00" };
        Assert.True(DateTimeDetector.IsDateTimeColumn("STD_DT", dateValues));
        Assert.True(DateTimeDetector.IsDateTimeColumn("MFG_DT", dateValues));
        Assert.True(DateTimeDetector.IsDateTimeColumn("PROC_DT", dateValues));
    }

    [Fact]
    public void IsDateTimeColumn_DtSuffix_WithNumericValues_ReturnsFalse()
    {
        // _DT columns with numeric values should not be detected as DateTime
        var numericValues = new[] { "100", "200", "300", "400" };
        Assert.False(DateTimeDetector.IsDateTimeColumn("STD_DT", numericValues));
    }
}
