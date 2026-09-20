using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// The note under `mloop list`'s experiment table that says which way its Metric column reads.
/// Before it, the heading named the metric and nothing said whether a bigger number was better —
/// while the trial leaderboard, rendered by the same file, had been saying it all along.
/// </summary>
public class ListCommandMetricNoteTests
{
    [Fact]
    public void NamesTheDirection_ForALowerIsBetterMetric()
    {
        var note = ListCommand.MetricDirectionNote(["rmse"]);

        Assert.NotNull(note);
        Assert.Contains("rmse", note);
        Assert.Contains("lower is better", note);
    }

    [Fact]
    public void NamesTheDirection_ForAHigherIsBetterMetric()
    {
        var note = ListCommand.MetricDirectionNote(["r_squared"]);

        Assert.NotNull(note);
        Assert.Contains("higher is better", note);
    }

    /// <summary>
    /// The case the note exists for. Rows optimizing different metrics sit next to each other, and
    /// the larger number is the worse model — a single-metric-only note would have skipped exactly
    /// the reader who needs it.
    /// </summary>
    [Fact]
    public void DescribesEveryMetric_WhenRowsOptimizedDifferentOnes()
    {
        var note = ListCommand.MetricDirectionNote(["rmse", "r_squared"]);

        Assert.NotNull(note);
        Assert.Contains("rmse", note);
        Assert.Contains("r_squared", note);
        Assert.Contains("lower is better", note);
        Assert.Contains("higher is better", note);
    }

    /// <summary>
    /// The list is ordered by time. Saying the direction without saying that invites reading the
    /// top row as the best one.
    /// </summary>
    [Fact]
    public void SaysTheListIsNotRanked()
        => Assert.Contains("not ranked", ListCommand.MetricDirectionNote(["rmse"])!);

    /// <summary>
    /// <c>MetricDirection.IsLowerBetter</c> answers <c>false</c> both for "higher is better" and
    /// for "never heard of it". Claiming a direction for an unrecognized metric would be inventing
    /// one, and for a custom lower-is-better metric it would be inverted.
    /// </summary>
    [Fact]
    public void SaysNothing_AboutAnUnrecognizedMetric()
        => Assert.Null(ListCommand.MetricDirectionNote(["business_score_v3"]));

    [Fact]
    public void SaysNothing_WhenThereIsNoMetricAtAll()
    {
        Assert.Null(ListCommand.MetricDirectionNote([]));
        Assert.Null(ListCommand.MetricDirectionNote([null]));
        Assert.Null(ListCommand.MetricDirectionNote(["   "]));
    }

    /// <summary>A note longer than the table it annotates has stopped being a note.</summary>
    [Fact]
    public void SaysNothing_WhenTooManyMetricsAreMixed()
        => Assert.Null(ListCommand.MetricDirectionNote(["rmse", "r_squared", "mae", "accuracy"]));

    /// <summary>
    /// Only recognized metrics are described, and an unrecognized one does not suppress the rest —
    /// it simply is not claimed.
    /// </summary>
    [Fact]
    public void SkipsTheUnrecognizedAndKeepsTheRest()
    {
        var note = ListCommand.MetricDirectionNote(["rmse", "business_score_v3"]);

        Assert.NotNull(note);
        Assert.Contains("rmse", note);
        Assert.DoesNotContain("business_score_v3", note);
    }
}
