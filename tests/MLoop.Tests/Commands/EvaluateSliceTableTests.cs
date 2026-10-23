using MLoop.CLI.Commands;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop evaluate --slice-by</c> prints one row per slice and one column per metric. A binary model
/// reports nine metrics, and in one table at 80 columns Spectre wrapped the numbers themselves —
/// <c>0.862</c> on one line, <c>1</c> under it — found by the release walkthrough on the churn example.
/// The metrics are now split into tables that each fit; these tests hold the width model to what
/// Spectre actually draws.
/// </summary>
public class EvaluateSliceTableTests
{
    private static readonly string[] BinaryMetrics =
    [
        "accuracy", "auc", "auprc", "f1_score", "precision", "recall",
        "negative_precision", "negative_recall", "ece",
    ];

    private static List<(string Name, int Width)> Widths(IEnumerable<string> names, int valueWidth = 6) =>
        names.Select(n => (n, Math.Max(n.Length, valueWidth))).ToList();

    [Fact]
    public void Groups_KeepEveryMetricInOrder()
    {
        var groups = EvaluateCommand.SliceMetricGroups("Month-to-month".Length, 4, Widths(BinaryMetrics), 80);

        Assert.True(groups.Count > 1);
        Assert.Equal(BinaryMetrics, groups.SelectMany(g => g));
    }

    [Fact]
    public void Groups_FitOneTableWhenTheConsoleIsWide()
    {
        var groups = EvaluateCommand.SliceMetricGroups(14, 4, Widths(BinaryMetrics), 400);

        Assert.Equal(BinaryMetrics, Assert.Single(groups));
    }

    [Fact]
    public void Groups_GiveAMetricWiderThanTheConsoleATableOfItsOwn()
    {
        var groups = EvaluateCommand.SliceMetricGroups(10, 4, [("a_very_long_metric_name", 23), ("auc", 6)], 30);

        Assert.Equal([["a_very_long_metric_name"], ["auc"]], groups.Select(g => g.ToArray()).ToArray());
    }

    [Theory]
    [InlineData(80, 6)]
    [InlineData(60, 6)]
    [InlineData(80, 11)] // regression errors such as "123456.7890"
    public void EachGroup_RendersWithoutWrappingAValue(int consoleWidth, int valueWidth)
    {
        var value = new string('7', valueWidth - 5) + ".1234";
        var labels = new[] { "Month-to-month", "One year" };
        var groups = EvaluateCommand.SliceMetricGroups(
            labels.Max(l => l.Length), "29*".Length + 1, Widths(BinaryMetrics, valueWidth), consoleWidth);

        foreach (var group in groups)
            Assert.True(RendersUnwrapped(group, labels, value, consoleWidth), string.Join(",", group));
    }

    [Fact]
    public void OneTableOfEveryMetric_WrapsAt80_TheCheckCanSeeIt()
    {
        // The shape that failed: the check above must report it, or its passes say nothing.
        Assert.False(RendersUnwrapped(BinaryMetrics, ["Month-to-month", "One year"], "0.8621", 80));
    }

    private static bool RendersUnwrapped(IReadOnlyList<string> group, string[] labels, string value, int consoleWidth)
    {
        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Ansi = AnsiSupport.No,
            ColorSystem = ColorSystemSupport.NoColors,
            Out = new AnsiConsoleOutput(writer),
        });
        console.Profile.Width = consoleWidth;

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Contract");
        table.AddColumn(new TableColumn("Rows").RightAligned());
        foreach (var name in group)
            table.AddColumn(new TableColumn(name).RightAligned());
        foreach (var label in labels)
            table.AddRow([label, "29*", .. group.Select(_ => value)]);
        console.Write(table);

        var lines = writer.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        // Two borders, the header, its rule and one line per row: anything more is a wrapped cell.
        return lines.Length == 4 + labels.Length
            && lines.Where(l => l.Contains("29*")).All(l => CountOf(l, value) == group.Count);
    }

    private static int CountOf(string line, string value)
    {
        var count = 0;
        for (var i = line.IndexOf(value, StringComparison.Ordinal); i >= 0; i = line.IndexOf(value, i + value.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
