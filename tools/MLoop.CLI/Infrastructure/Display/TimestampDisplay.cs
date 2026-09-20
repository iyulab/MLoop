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

    /// <summary>How old this instant is, in words — <c>just now</c>, <c>12m ago</c>, <c>3h ago</c>,
    /// <c>9d ago</c>, <c>4mo ago</c>, <c>2y ago</c>.</summary>
    /// <remarks>
    /// <para>Two commands had grown their own version and the two disagreed on four things: the
    /// cutoff to an absolute date (7 days against 30), whether a value under a minute reads "just
    /// now" or "0m ago", what colour it takes, and — the one that made them contradict each other —
    /// the zone. One took its fallback date through <see cref="AsLocal"/>, the other formatted the
    /// raw UTC value, so <c>mloop list</c> and <c>mloop status</c> could print different calendar
    /// dates for the same file.</para>
    /// <para><b>There is no date fallback, and that is the point.</b> An age stated as an age has no
    /// zone to get wrong, so the disagreement that motivated this cannot come back in a new form.
    /// The first version of this method did fall back to a local date past a month, which put both
    /// columns under the <see cref="ZoneHeading"/> pairing — and on the narrower of the two that
    /// heading wrapped to three lines, spending a permanent column of width on a branch almost no
    /// row takes. The exact instant is still one command away: <c>--json</c> carries it, and so does
    /// the experiment report.</para>
    /// <para>Colour is deliberately not decided here: how urgent an age looks belongs to the table
    /// showing it, while what the age <i>is</i> does not.</para>
    /// </remarks>
    internal static string Relative(DateTime value) => Relative(AsLocal(value));

    /// <inheritdoc cref="Relative(DateTime)"/>
    internal static string Relative(DateTimeOffset value)
    {
        var elapsed = DateTimeOffset.Now - value.ToLocalTime();

        // A clock skew or a file stamped slightly in the future reads as "just now" rather than as
        // a negative age.
        if (elapsed < TimeSpan.FromMinutes(1))
            return "just now";
        if (elapsed < TimeSpan.FromHours(1))
            return $"{(int)elapsed.TotalMinutes}m ago";
        if (elapsed < TimeSpan.FromDays(1))
            return $"{(int)elapsed.TotalHours}h ago";
        if (elapsed.TotalDays < DaysPerMonth)
            return $"{(int)elapsed.TotalDays}d ago";
        if (elapsed.TotalDays < DaysPerYear)
            return $"{(int)(elapsed.TotalDays / DaysPerMonth)}mo ago";

        return $"{(int)(elapsed.TotalDays / DaysPerYear)}y ago";
    }

    /// <summary>
    /// Just the calendar date this instant falls on in the reader's zone, with no offset — for a
    /// line that names the zone once for several dates at a time, the way a column heading does for
    /// several rows.
    /// </summary>
    internal static string LocalDate(DateTimeOffset value) => value.ToLocalTime().ToString(DateOnly);

    // Nominal lengths, for turning an elapsed span into a word. An age is an approximation by the
    // time it is being said in months, and a reader comparing "4mo ago" against "5mo ago" is not
    // counting calendar months.
    private const double DaysPerMonth = 30.0;
    private const double DaysPerYear = 365.0;

    private const string DateOnly = "yyyy-MM-dd";

    /// <summary>
    /// A column heading that names the zone its rows are in, e.g. <c>Timestamp (+09:00)</c>.
    /// </summary>
    internal static string ZoneHeading(string caption) =>
        $"{caption} ({DateTimeOffset.Now.ToString("zzz")})";
}
