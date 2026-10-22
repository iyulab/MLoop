using System.Globalization;

namespace MLoop.Core.Storage;

/// <summary>
/// The times written into names a program reads back — day files, backup directories, output files, job ids — in
/// one place, in the invariant culture and the Gregorian calendar whatever the machine's. (The times a person reads
/// on screen are <c>TimestampDisplay</c>'s, in the CLI.) A name written through the current culture under a culture
/// with another calendar (th-TH counts Buddhist years) came out as <c>2569-…</c>, and the retention pruner — reading
/// names with the invariant culture — took a day file for one five centuries ahead and never pruned it. The day
/// format lived in three files, writers and readers, and the pruner could only work while all three agreed.
/// </summary>
/// <remarks>
/// Which clock a name uses is the caller's, by who reads it: a key a program sorts or prunes by is UTC; a file a
/// person looks for in a listing is local.
/// </remarks>
public static class TimestampKeys
{
    /// <summary>The name of a day file — one file of log or feedback entries per UTC day.</summary>
    public const string DayFormat = "yyyy-MM-dd";

    /// <summary>A time to the second in a name (<c>20261022-161500</c>).</summary>
    public const string StampFormat = "yyyyMMdd-HHmmss";

    /// <summary>The day file name (without extension) for an entry at <paramref name="at"/>: its UTC day.</summary>
    public static string Day(DateTimeOffset at) => at.UtcDateTime.ToString(DayFormat, CultureInfo.InvariantCulture);

    /// <summary>The day a day file name (without extension) stands for, when it is one.</summary>
    public static bool TryParseDay(string name, out DateOnly day)
        => DateOnly.TryParseExact(name, DayFormat, CultureInfo.InvariantCulture, DateTimeStyles.None, out day);

    /// <summary>
    /// Whether the day file at <paramref name="path"/> may hold entries between <paramref name="from"/> and
    /// <paramref name="to"/> — true for a file whose name is not a day, so it is read rather than skipped.
    /// </summary>
    public static bool DayFileCovers(string path, DateTimeOffset from, DateTimeOffset to)
    {
        if (!TryParseDay(Path.GetFileNameWithoutExtension(path), out var day))
            return true;
        var start = new DateTimeOffset(day.ToDateTime(TimeOnly.MinValue), TimeSpan.Zero);
        return start <= to && start.AddDays(1) > from;
    }

    /// <summary>A time to the second in a name, as given — UTC or local is the caller's.</summary>
    public static string Stamp(DateTime at) => at.ToString(StampFormat, CultureInfo.InvariantCulture);

    /// <inheritdoc cref="Stamp(DateTime)"/>
    public static string Stamp(DateTimeOffset at) => at.ToString(StampFormat, CultureInfo.InvariantCulture);
}
