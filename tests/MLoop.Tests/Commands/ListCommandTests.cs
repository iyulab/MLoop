using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

public class ListCommandTests
{
    #region FormatRelativeTime

    [Fact]
    public void FormatRelativeTime_JustNow_LessThanMinute()
    {
        var timestamp = DateTime.Now.AddSeconds(-30);
        var result = ListCommand.FormatRelativeTime(timestamp);

        Assert.Contains("just now", result);
    }

    [Fact]
    public void FormatRelativeTime_MinutesAgo()
    {
        var timestamp = DateTime.Now.AddMinutes(-15);
        var result = ListCommand.FormatRelativeTime(timestamp);

        Assert.Contains("15m ago", result);
    }

    [Fact]
    public void FormatRelativeTime_HoursAgo()
    {
        var timestamp = DateTime.Now.AddHours(-5);
        var result = ListCommand.FormatRelativeTime(timestamp);

        Assert.Contains("5h ago", result);
    }

    [Fact]
    public void FormatRelativeTime_DaysAgo()
    {
        var timestamp = DateTime.Now.AddDays(-3);
        var result = ListCommand.FormatRelativeTime(timestamp);

        Assert.Contains("3d ago", result);
    }

    [Fact]
    public void FormatRelativeTime_OlderThanWeek_ShowsDate()
    {
        var timestamp = DateTime.Now.AddDays(-30);
        var result = ListCommand.FormatRelativeTime(timestamp);

        Assert.Contains(timestamp.ToString("yyyy-MM-dd"), result);
    }

    [Fact]
    public void FormatRelativeTime_UtcTimestamp_ConvertsToLocal()
    {
        var utcTimestamp = DateTime.UtcNow.AddMinutes(-10);
        var result = ListCommand.FormatRelativeTime(utcTimestamp);

        Assert.Contains("10m ago", result);
    }

    #endregion

    #region FormatMetric

    [Fact]
    public void FormatMetric_NoValue_ReturnsDash()
    {
        var exp = new ExperimentSummary
        {
            ModelName = "test",
            ExperimentId = "exp-001",
            Timestamp = DateTime.Now,
            Status = "Completed",
            BestMetric = null
        };

        var result = ListCommand.FormatMetric(exp);
        Assert.Contains("-", result);
    }

    [Fact]
    public void FormatMetric_WithValue_FormatsToFourDecimals()
    {
        var exp = new ExperimentSummary
        {
            ModelName = "test",
            ExperimentId = "exp-001",
            Timestamp = DateTime.Now,
            Status = "Completed",
            BestMetric = 0.9567
        };

        var result = ListCommand.FormatMetric(exp);
        Assert.Contains("0.9567", result);
    }

    [Fact]
    public void FormatMetric_WithMetricName_IncludesNameInParens()
    {
        var exp = new ExperimentSummary
        {
            ModelName = "test",
            ExperimentId = "exp-001",
            Timestamp = DateTime.Now,
            Status = "Completed",
            BestMetric = 0.85,
            MetricName = "accuracy"
        };

        var result = ListCommand.FormatMetric(exp);
        Assert.Contains("accuracy", result);
    }

    [Fact]
    public void FormatMetric_WithoutMetricName_NoParens()
    {
        var exp = new ExperimentSummary
        {
            ModelName = "test",
            ExperimentId = "exp-001",
            Timestamp = DateTime.Now,
            Status = "Completed",
            BestMetric = 0.85,
            MetricName = null
        };

        var result = ListCommand.FormatMetric(exp);
        Assert.DoesNotContain("(", result);
    }

    #endregion
}
