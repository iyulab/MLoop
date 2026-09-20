using MLoop.CLI.Infrastructure.Display;

namespace MLoop.Tests.Infrastructure.Display;

/// <summary>
/// `mloop list` and `mloop status` each had their own version of "how old is this", and the two
/// disagreed on the cutoff to a date, on whether a fresh value reads "just now", and on the zone —
/// so the same file could be shown with two different calendar dates depending on which command
/// you asked.
/// </summary>
public class TimestampRelativeTests
{
    [Fact]
    public void UnderAMinute_ReadsJustNow()
        => Assert.Equal("just now", TimestampDisplay.Relative(DateTimeOffset.Now.AddSeconds(-20)));

    /// <summary>
    /// A file stamped a moment into the future — clock skew, or a writer on another machine — is an
    /// age of zero, not a negative one.
    /// </summary>
    [Fact]
    public void SlightlyInTheFuture_ReadsJustNow()
        => Assert.Equal("just now", TimestampDisplay.Relative(DateTimeOffset.Now.AddSeconds(30)));

    [Theory]
    [InlineData(5, "5m ago")]
    [InlineData(59, "59m ago")]
    public void UnderAnHour_CountsMinutes(int minutes, string expected)
        => Assert.Equal(expected, TimestampDisplay.Relative(DateTimeOffset.Now.AddMinutes(-minutes)));

    [Theory]
    [InlineData(1, "1h ago")]
    [InlineData(23, "23h ago")]
    public void UnderADay_CountsHours(int hours, string expected)
        => Assert.Equal(expected, TimestampDisplay.Relative(DateTimeOffset.Now.AddHours(-hours)));

    [Theory]
    [InlineData(1, "1d ago")]
    [InlineData(29, "29d ago")]
    public void UnderAMonth_CountsDays(int days, string expected)
        => Assert.Equal(expected, TimestampDisplay.Relative(DateTimeOffset.Now.AddDays(-days)));

    [Theory]
    [InlineData(31, "1mo ago")]
    [InlineData(200, "6mo ago")]
    public void PastAMonth_CountsMonths(int days, string expected)
        => Assert.Equal(expected, TimestampDisplay.Relative(DateTimeOffset.Now.AddDays(-days)));

    [Theory]
    [InlineData(400, "1y ago")]
    [InlineData(1100, "3y ago")]
    public void PastAYear_CountsYears(int days, string expected)
        => Assert.Equal(expected, TimestampDisplay.Relative(DateTimeOffset.Now.AddDays(-days)));

    /// <summary>
    /// The reason there is no date fallback. A rendered age names no clock, so the defect that
    /// motivated this authority — two commands showing different calendar dates for one file —
    /// has nowhere left to occur.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(45)]
    [InlineData(800)]
    public void NoAgeEverRendersACalendarDate(int daysAgo)
    {
        var rendered = TimestampDisplay.Relative(DateTimeOffset.Now.AddDays(-daysAgo));

        Assert.DoesNotContain("-", rendered);
        Assert.EndsWith("ago", rendered);
    }

    /// <summary>
    /// A <see cref="DateTimeKind.Utc"/> value and the same instant as a <see cref="DateTimeOffset"/>
    /// are the same age — the overload a caller happens to hold must not change the answer.
    /// </summary>
    [Fact]
    public void BothOverloadsAgreeOnTheSameInstant()
    {
        var utc = DateTime.UtcNow.AddHours(-5);

        Assert.Equal(
            TimestampDisplay.Relative(new DateTimeOffset(utc, TimeSpan.Zero)),
            TimestampDisplay.Relative(utc));
    }

    [Fact]
    public void LocalDate_IsTheReadersCalendarDateWithNoOffset()
    {
        var instant = DateTimeOffset.Now.AddDays(-3);

        var rendered = TimestampDisplay.LocalDate(instant);

        Assert.Equal(instant.ToLocalTime().ToString("yyyy-MM-dd"), rendered);
        Assert.DoesNotContain("+", rendered);
    }
}
