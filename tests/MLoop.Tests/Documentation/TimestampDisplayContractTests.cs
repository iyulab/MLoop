using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.Display;

namespace MLoop.Tests.Documentation;

/// <summary>
/// An instant a person reads is rendered in their zone and says which zone that is, and exactly one
/// place decides how.
/// </summary>
/// <remarks>
/// <para>
/// Nine call sites each carried their own copy of the format string, and separately from them the
/// paths that name directories and partitions used UTC. Both halves were defensible; nothing
/// recorded that they were different halves, so one command printed a local time beside a UTC
/// directory name it had just created and the two disagreed by the length of a time zone.
/// </para>
/// <para>
/// The assertions below check the shape of an offset, never its value. A test that expected a
/// particular offset would encode the machine that ran it, and the suite has already paid for that
/// once with a percent sign that was green on every developer's machine and red on the runner.
/// </para>
/// </remarks>
public class TimestampDisplayContractTests
{
    /// <summary>A trailing numeric UTC offset, whatever the runner's zone happens to be.</summary>
    private static readonly Regex TrailingOffset = new(@" [+-]\d{2}:\d{2}$", RegexOptions.Compiled);

    /// <summary>A human-facing timestamp format written out at a call site instead of asked for.</summary>
    private static readonly Regex InlineTimestampFormat =
        new(@"[:""]yyyy-MM-dd HH:mm", RegexOptions.Compiled);

    [Fact]
    public void ARenderedTimestampNamesItsZone()
    {
        Assert.Matches(TrailingOffset, TimestampDisplay.Local(new DateTime(2026, 9, 9, 5, 30, 0, DateTimeKind.Utc)));
        Assert.Matches(TrailingOffset, TimestampDisplay.Local(DateTimeOffset.UtcNow));
        Assert.Matches(TrailingOffset, TimestampDisplay.LocalToMinutes(new DateTime(2026, 9, 9, 5, 30, 0, DateTimeKind.Utc)));
    }

    [Fact]
    public void TheSameInstantRendersTheSameWhicheverTypeCarriesIt()
    {
        var utc = new DateTime(2026, 9, 9, 5, 30, 0, DateTimeKind.Utc);

        Assert.Equal(TimestampDisplay.Local(utc), TimestampDisplay.Local(new DateTimeOffset(utc)));
    }

    [Fact]
    public void AnOffsetIsNeverAnAbbreviation()
    {
        // .NET has no dependable cross-platform source of zone abbreviations, so a rendered "KST"
        // could only have come from a literal — which would report the author's continent to every
        // reader on another one.
        var rendered = TimestampDisplay.Local(DateTimeOffset.UtcNow);

        Assert.DoesNotMatch(new Regex(@"[A-Z]{2,5}$"), rendered);
    }

    [Fact]
    public void NoCommandWritesItsOwnHumanFacingTimestampFormat()
    {
        // Scoped to the CLI, where the authority lives and is internal. The API's Serilog console
        // template writes its own timestamp and is not an offender: it already ends in `zzz`, so it
        // honours the same convention by the only means available to it.
        var scanned = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
            .Where(f => !Path.GetFileName(f).Equals(nameof(TimestampDisplay) + ".cs", StringComparison.Ordinal))
            .ToList();

        // The scan reaches the files it is about. Without this, narrowing the directory to one that
        // no longer resolves would empty the set and read as "no call site writes its own format".
        Assert.Contains(scanned, f => Path.GetFileName(f) == "PromoteCommand.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "LogsCommand.cs");

        var offenders = scanned
            .Where(f => InlineTimestampFormat.IsMatch(File.ReadAllText(f)))
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "A timestamp a person reads is rendered by " + nameof(TimestampDisplay) + ", so that the "
            + "zone it is in is decided once rather than per call site:\n" + string.Join('\n', offenders));
    }

    /// <summary>A file that drops the zone from its rows has to put it back in a heading.</summary>
    private static bool DropsTheZoneWithoutNamingIt(string source) =>
        source.Contains(nameof(TimestampDisplay.LocalWithoutZone), StringComparison.Ordinal)
        && !source.Contains(nameof(TimestampDisplay.ZoneHeading), StringComparison.Ordinal);

    [Fact]
    public void ATimestampWithoutItsZoneAlwaysSitsUnderAHeadingThatNamesOne()
    {
        var scanned = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
            .Where(f => !Path.GetFileName(f).Equals(nameof(TimestampDisplay) + ".cs", StringComparison.Ordinal))
            .ToList();

        // The two call sites this pairing exists for, so a rename cannot empty the scan silently.
        Assert.Contains(scanned, f => Path.GetFileName(f) == "LogsCommand.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "FeedbackCommand.cs");

        var unlabelled = scanned
            .Where(f => DropsTheZoneWithoutNamingIt(File.ReadAllText(f)))
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(unlabelled.Count == 0,
            "Dropping the offset from a listing's rows is only allowed because the column heading "
            + "carries it. Without the heading the rows are back to being a number with no zone:\n"
            + string.Join('\n', unlabelled));
    }

    [Fact]
    public void ThePairingCheckSeesBothHalves()
    {
        // Companion: a pairing check that has stopped matching one of its two halves reports every
        // file compliant, which is the same output as compliance.
        Assert.True(DropsTheZoneWithoutNamingIt("var t = TimestampDisplay.LocalWithoutZone(x);"));
        Assert.False(DropsTheZoneWithoutNamingIt(
            "table.AddColumn(TimestampDisplay.ZoneHeading(\"At\")); var t = TimestampDisplay.LocalWithoutZone(x);"));
        Assert.False(DropsTheZoneWithoutNamingIt("var t = TimestampDisplay.Local(x);"));
    }

    [Fact]
    public void TheScanSeesTheShapeItForbidsAndAcceptsTheOneItWants()
    {
        // Companion to the scan above. Its pattern is the only thing standing between "no call site
        // writes its own format" and "the pattern no longer matches anything", and those two report
        // themselves identically.
        Assert.Matches(InlineTimestampFormat, """table.AddRow("At", t.ToString("yyyy-MM-dd HH:mm:ss"));""");
        Assert.Matches(InlineTimestampFormat, """AnsiConsole.MarkupLine($"{t:yyyy-MM-dd HH:mm:ss}");""");
        Assert.DoesNotMatch(InlineTimestampFormat, """table.AddRow("At", TimestampDisplay.Local(t));""");

        // And a file name is not a screen: those stay local and compact, and must not be dragged in.
        Assert.DoesNotMatch(InlineTimestampFormat, """var name = $"forecast-{DateTime.Now:yyyyMMdd-HHmmss}.csv";""");
    }
}
