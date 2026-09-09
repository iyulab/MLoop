namespace MLoop.CLI.Infrastructure.Display;

/// <summary>
/// The single place an instant becomes text a person reads.
/// </summary>
/// <remarks>
/// <para>
/// The convention this encodes has one axis, and it is not "file name versus screen" — it is
/// <b>who reads the value</b>:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <b>A person reads it</b> — render in the reader's own zone and say which zone that is.
///       Their local time is the only clock they can check the number against, and the offset is
///       what makes the number checkable rather than merely plausible.
///     </description>
///   </item>
///   <item>
///     <description>
///       <b>A machine reads it</b> — storage keys, log partition names, backup directory names —
///       UTC, because those are sorted, compared, and merged across machines that do not share a
///       zone.
///     </description>
///   </item>
///   <item>
///     <description>
///       <b>A person looks for it in a file listing</b> — the names of files the product writes for
///       the user to find are local, for the same reason the screen is: someone hunting for the
///       export they made after lunch should not have to convert.
///     </description>
///   </item>
/// </list>
/// <para>
/// The failure that motivated writing it down was a single command's output disagreeing with
/// itself: the promotion summary printed a local timestamp while the backup directory it had just
/// created was named in UTC, so one screen showed two times nine hours apart with nothing to say
/// they were the same kind of thing. Neither value was wrong; nothing said which was which.
/// </para>
/// <para>
/// The offset is written as a number (<c>+09:00</c>), never an abbreviation. .NET has no dependable
/// cross-platform source of zone abbreviations, and one baked in as a literal would report the
/// author's machine to every reader on a different continent.
/// </para>
/// </remarks>
internal static class TimestampDisplay
{
    private const string WithZone = "yyyy-MM-dd HH:mm:ss zzz";
    private const string WithZoneToMinutes = "yyyy-MM-dd HH:mm zzz";
    private const string WithoutZone = "yyyy-MM-dd HH:mm:ss";

    /// <summary>
    /// The instant this <see cref="DateTime"/> names, in the reader's zone.
    /// </summary>
    /// <remarks>
    /// A <see cref="DateTimeKind.Utc"/> value is converted; anything else is taken to be local
    /// already, which is what an unspecified value has always meant everywhere it is rendered here.
    /// Converting through <see cref="DateTimeOffset"/> rather than formatting the
    /// <see cref="DateTime"/> directly is what makes the printed offset a fact about the value
    /// instead of a fact about the process that printed it.
    /// </remarks>
    internal static DateTimeOffset AsLocal(DateTime value) =>
        value.Kind == DateTimeKind.Utc
            ? new DateTimeOffset(value).ToLocalTime()
            : new DateTimeOffset(value);

    /// <summary>A full timestamp in the reader's zone, to the second.</summary>
    internal static string Local(DateTime value) => AsLocal(value).ToString(WithZone);

    /// <inheritdoc cref="Local(DateTime)"/>
    internal static string Local(DateTimeOffset value) => value.ToLocalTime().ToString(WithZone);

    /// <summary>
    /// The same, to the minute — for a side-by-side table, where seconds cost a column's width and
    /// tell the reader nothing they are comparing on.
    /// </summary>
    internal static string LocalToMinutes(DateTime value) => AsLocal(value).ToString(WithZoneToMinutes);

    /// <summary>
    /// A timestamp with the zone left off, for a listing that names the zone in its column heading
    /// instead — see <see cref="ZoneHeading"/>.
    /// </summary>
    /// <remarks>
    /// Repeating the offset down a column of many rows spends six characters per row to say the
    /// same thing every time, and in a listing that has to fit an eighty-column terminal it is the
    /// difference between one line per row and two. Measured: adding the offset inline to the
    /// prediction log wrapped every row onto a second line. Every row of one listing is rendered in
    /// the same zone, so the heading is the honest place to say which.
    /// </remarks>
    internal static string LocalWithoutZone(DateTimeOffset value) =>
        value.ToLocalTime().ToString(WithoutZone);

    /// <summary>
    /// A column heading that names the zone its rows are in, e.g. <c>Timestamp (+09:00)</c>.
    /// </summary>
    internal static string ZoneHeading(string caption) =>
        $"{caption} ({DateTimeOffset.Now.ToString("zzz")})";
}
