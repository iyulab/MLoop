using MLoop.CLI.Infrastructure;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// Pins that user-supplied date filters are interpreted as the local calendar day, matching the
/// CLI's local-time display (F-31). The prior code forced UTC midnight, shifting the window by the
/// machine's UTC offset. Assertions compare against the machine's own local offset so they hold in
/// any timezone (including a UTC build agent).
/// </summary>
public class DateRangeFilterTests
{
    [Fact]
    public void ToFilterRange_FromDate_IsLocalMidnight_NotUtc()
    {
        var date = new DateTime(2026, 6, 29);
        var expectedOffset = TimeZoneInfo.Local.GetUtcOffset(date);

        var (from, _) = DateRangeFilter.ToFilterRange(date, null);

        Assert.NotNull(from);
        Assert.Equal(expectedOffset, from!.Value.Offset);
        // Local wall-clock start of the requested day.
        Assert.Equal(new DateTime(2026, 6, 29, 0, 0, 0), from.Value.DateTime);
    }

    [Fact]
    public void ToFilterRange_ToDate_CoversWholeLocalDay()
    {
        var date = new DateTime(2026, 6, 29);

        var (_, to) = DateRangeFilter.ToFilterRange(null, date);

        Assert.NotNull(to);
        Assert.Equal(TimeZoneInfo.Local.GetUtcOffset(date), to!.Value.Offset);
        // Last tick of 2026-06-29 local time.
        Assert.Equal(new DateTime(2026, 6, 30, 0, 0, 0).AddTicks(-1), to.Value.DateTime);
    }

    [Fact]
    public void ToFilterRange_NullInputs_ReturnNull()
    {
        var (from, to) = DateRangeFilter.ToFilterRange(null, null);

        Assert.Null(from);
        Assert.Null(to);
    }
}
