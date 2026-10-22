using System.Globalization;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

public class TimestampKeysTests
{
    private static T Under<T>(string culture, Func<T> read)
    {
        var before = CultureInfo.CurrentCulture;
        CultureInfo.CurrentCulture = new CultureInfo(culture);
        try { return read(); }
        finally { CultureInfo.CurrentCulture = before; }
    }

    [Fact]
    public void ANameIsWrittenInTheGregorianCalendar_UnderACultureWithAnother()
    {
        var at = new DateTimeOffset(2026, 10, 22, 16, 15, 0, TimeSpan.Zero);

        Assert.Equal("2026-10-22", Under("th-TH", () => TimestampKeys.Day(at)));
        Assert.Equal("20261022-161500", Under("th-TH", () => TimestampKeys.Stamp(at)));
        Assert.Equal("20261022-161500", Under("ar-SA", () => TimestampKeys.Stamp(at.UtcDateTime)));
    }

    [Fact]
    public void ADayFileIsItsUtcDay_AndReadsBack()
    {
        var lateInSeoul = new DateTimeOffset(2026, 10, 23, 8, 0, 0, TimeSpan.FromHours(9));   // 2026-10-22 23:00 UTC

        Assert.Equal("2026-10-22", TimestampKeys.Day(lateInSeoul));
        Assert.True(Under("th-TH", () => TimestampKeys.TryParseDay("2026-10-22", out var day) ? day : default) == new DateOnly(2026, 10, 22));
        Assert.False(TimestampKeys.TryParseDay("notes", out _));
    }

    [Fact]
    public void ADayFileCoversItsDay_AndAFileNamedOtherwiseIsRead()
    {
        var from = new DateTimeOffset(2026, 10, 22, 12, 0, 0, TimeSpan.Zero);
        var to = new DateTimeOffset(2026, 10, 23, 1, 0, 0, TimeSpan.Zero);

        Assert.True(TimestampKeys.DayFileCovers("logs/m/2026-10-22.jsonl", from, to));
        Assert.True(TimestampKeys.DayFileCovers("logs/m/2026-10-23.jsonl", from, to));
        Assert.False(TimestampKeys.DayFileCovers("logs/m/2026-10-21.jsonl", from, to));
        Assert.False(TimestampKeys.DayFileCovers("logs/m/2026-10-24.jsonl", from, to));
        Assert.True(TimestampKeys.DayFileCovers("logs/m/backup.jsonl", from, to));
    }
}
