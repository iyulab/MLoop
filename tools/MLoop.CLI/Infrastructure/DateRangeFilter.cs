namespace MLoop.CLI.Infrastructure;

/// <summary>
/// Converts user-supplied <c>--from</c>/<c>--to</c> date options into a timestamp filter range.
/// The CLI stores timestamps in UTC but displays them in local time, so a bare date the user types
/// means their <i>local</i> calendar day. This was previously forced to UTC midnight (via
/// <c>new DateTimeOffset(date, TimeSpan.Zero)</c>) in three separate call sites — <c>logs</c>,
/// <c>feedback list</c>, <c>feedback metrics</c> — so a <c>--from</c> filter was off by the machine's
/// UTC offset (e.g. 9 hours in KST: the first 9 hours of the requested local day were excluded and
/// the tail of the previous day leaked in), contradicting the local-time display (F-31). Centralized
/// here so the three cannot drift again.
/// </summary>
internal static class DateRangeFilter
{
    /// <summary>
    /// <paramref name="from"/> becomes the start of that local day; <paramref name="to"/> becomes the
    /// end of that local day (inclusive of the whole day). A <see cref="DateTime"/> with
    /// <see cref="DateTimeKind.Unspecified"/> (what System.CommandLine yields for <c>yyyy-MM-dd</c>) is
    /// interpreted as local time.
    /// </summary>
    public static (DateTimeOffset? From, DateTimeOffset? To) ToFilterRange(DateTime? from, DateTime? to)
    {
        DateTimeOffset? fromOffset = from.HasValue
            ? new DateTimeOffset(from.Value)
            : null;
        DateTimeOffset? toOffset = to.HasValue
            ? new DateTimeOffset(to.Value.AddDays(1).AddTicks(-1))
            : null;
        return (fromOffset, toOffset);
    }
}
