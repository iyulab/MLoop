using DataLens.Models;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Handles display/presentation logic for info command output.
/// Extracted from InfoCommand following the TrainPresenter pattern.
/// </summary>
internal static class InfoPresenter
{
    public static void DisplayFileInfo(string fileName, long fileSize, int lineCount, DateTime lastModified)
    {
        AnsiConsole.Write(new Rule("[yellow]File Information[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Property");
        table.AddColumn("Value");

        table.AddRow("File Size", FormatFileSize(fileSize));
        table.AddRow("Rows (excluding header)", lineCount.ToString("N0"));
        table.AddRow("Last Modified", lastModified.ToString("yyyy-MM-dd HH:mm:ss"));

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    public static void DisplayColumnInfo(
        string[] columns,
        Func<string, int, string> inferDisplayType,
        Func<string, string, string> getColumnPurpose)
    {
        AnsiConsole.Write(new Rule("[yellow]Column Information[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("#");
        table.AddColumn("Column Name");
        table.AddColumn("Data Type");
        table.AddColumn("Purpose");

        for (int i = 0; i < columns.Length; i++)
        {
            var dataType = inferDisplayType(columns[i], i);
            var purpose = getColumnPurpose(columns[i], dataType);
            table.AddRow((i + 1).ToString(), columns[i], dataType, purpose);
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays the enhanced statistics table with DataLens profile data merged in.
    /// Falls back to basic 4-column table when profile is null.
    /// </summary>
    public static void DisplayDataStatistics(
        string[] columns,
        Dictionary<string, (long MissingCount, int UniqueCount)> columnStats,
        int lineCount,
        ProfileReport? profile)
    {
        AnsiConsole.Write(new Rule("[yellow]Data Statistics[/]").LeftJustified());
        AnsiConsole.WriteLine();

        // Build a lookup from profile column name to ColumnProfile
        var profileLookup = new Dictionary<string, ColumnProfile>();
        if (profile?.Columns != null)
        {
            foreach (var col in profile.Columns)
            {
                profileLookup[col.Name] = col;
            }
        }

        bool hasEnhanced = profileLookup.Count > 0;

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Column");
        table.AddColumn(new TableColumn("Missing").RightAligned());
        table.AddColumn(new TableColumn("Missing %").RightAligned());
        table.AddColumn(new TableColumn("Unique (sample)").RightAligned());

        if (hasEnhanced)
        {
            table.AddColumn(new TableColumn("Type").Centered());
            table.AddColumn(new TableColumn("Mean").RightAligned());
            table.AddColumn(new TableColumn("StdDev").RightAligned());
            table.AddColumn(new TableColumn("Min").RightAligned());
            table.AddColumn(new TableColumn("Max").RightAligned());
        }

        for (int i = 0; i < columns.Length; i++)
        {
            var colName = columns[i];
            if (!columnStats.TryGetValue(colName, out var stats)) continue;

            var missingPct = lineCount > 0 ? (stats.MissingCount / (double)lineCount) * 100 : 0;
            var uniqueStr = stats.UniqueCount > 0 ? $"~{stats.UniqueCount}" : "N/A";

            if (hasEnhanced && profileLookup.TryGetValue(colName, out var summary))
            {
                var typeStr = summary.DataType.ToLowerInvariant() switch
                {
                    "numeric" or "float" or "integer" => "[cyan]Num[/]",
                    "boolean" => "[green]Bool[/]",
                    "categorical" => "[yellow]Cat[/]",
                    "text" or "string" => "[blue]Text[/]",
                    _ => $"[grey]{summary.DataType}[/]"
                };

                bool isNumeric = summary.Mean.HasValue;
                table.AddRow(
                    colName,
                    stats.MissingCount.ToString("N0"),
                    $"{missingPct:F2}%",
                    uniqueStr,
                    typeStr,
                    isNumeric ? summary.Mean!.Value.ToString("F4") : "[grey]-[/]",
                    isNumeric ? (summary.StdDev?.ToString("F4") ?? "[grey]-[/]") : "[grey]-[/]",
                    isNumeric ? (summary.Min?.ToString("F4") ?? "[grey]-[/]") : "[grey]-[/]",
                    isNumeric ? (summary.Max?.ToString("F4") ?? "[grey]-[/]") : "[grey]-[/]");
            }
            else if (hasEnhanced)
            {
                table.AddRow(
                    colName,
                    stats.MissingCount.ToString("N0"),
                    $"{missingPct:F2}%",
                    uniqueStr,
                    "[grey]-[/]", "[grey]-[/]", "[grey]-[/]", "[grey]-[/]", "[grey]-[/]");
            }
            else
            {
                table.AddRow(
                    colName,
                    stats.MissingCount.ToString("N0"),
                    $"{missingPct:F2}%",
                    uniqueStr);
            }
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    public static void DisplayLabelDistribution(
        Dictionary<string, int> labelDistribution,
        int lineCount)
    {
        if (labelDistribution.Count == 0) return;

        // Detect regression target (many unique numeric values)
        var nonEmptyClasses = labelDistribution
            .Where(p => p.Key != "(empty)").ToList();
        bool isLikelyRegression = nonEmptyClasses.Count > 20
            && nonEmptyClasses.All(p => double.TryParse(p.Key,
                System.Globalization.NumberStyles.Any,
                System.Globalization.CultureInfo.InvariantCulture, out _));

        if (isLikelyRegression)
        {
            DisplayRegressionLabelStats(labelDistribution, nonEmptyClasses);
        }
        else
        {
            DisplayClassificationLabelDist(labelDistribution, lineCount);
        }
    }

    public static void DisplayCorrelation(CorrelationReport report)
    {
        AnsiConsole.Write(new Rule("[yellow]Correlation Analysis[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var colNames = report.ColumnNames;
        int n = colNames.Count;

        if (n <= 6 && report.Matrix != null)
        {
            // Full matrix display
            var table = new Table().Border(TableBorder.Rounded);
            table.AddColumn("");
            for (int i = 0; i < n; i++)
                table.AddColumn(new TableColumn(Truncate(colNames[i], 12)).RightAligned());

            for (int i = 0; i < n; i++)
            {
                var cells = new string[n + 1];
                cells[0] = Truncate(colNames[i], 12);
                for (int j = 0; j < n; j++)
                {
                    var r = report.Matrix[i, j];
                    cells[j + 1] = ColorCorrelation(r);
                }
                table.AddRow(cells);
            }

            AnsiConsole.Write(table);
        }
        else if (report.HighCorrelationPairs.Count > 0)
        {
            // Show high-correlation pairs only
            var table = new Table().Border(TableBorder.Rounded);
            table.AddColumn("Column 1");
            table.AddColumn("Column 2");
            table.AddColumn(new TableColumn("Pearson r").RightAligned());

            foreach (var pair in report.HighCorrelationPairs.OrderByDescending(p => p.AbsValue))
            {
                table.AddRow(pair.Column1, pair.Column2, ColorCorrelation(pair.Value));
            }

            AnsiConsole.Write(table);
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]No high correlations found among numeric columns.[/]");
        }

        AnsiConsole.WriteLine();
    }

    public static void DisplayFeatureImportance(FeatureImportanceSummary importance)
    {
        AnsiConsole.Write(new Rule("[yellow]Feature Importance[/]").LeftJustified());
        AnsiConsole.WriteLine();

        if (importance.Scores.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No feature importance scores available.[/]");
            AnsiConsole.WriteLine();
            return;
        }

        // Sort descending by score
        var ranked = importance.Scores
            .OrderByDescending(x => x.Score)
            .ToList();

        double maxScore = ranked.Max(x => x.Score);
        if (maxScore <= 0) maxScore = 1;

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Feature");
        table.AddColumn(new TableColumn("Score").RightAligned());
        table.AddColumn("Bar");

        foreach (var item in ranked)
        {
            int barLen = (int)Math.Round((item.Score / maxScore) * 25);
            var bar = new string('#', Math.Max(0, barLen));
            var color = item.Score >= maxScore * 0.7 ? "green" :
                       item.Score >= maxScore * 0.4 ? "yellow" : "grey";
            table.AddRow(item.Name, item.Score.ToString("F4"), $"[{color}]{bar}[/]");
        }

        AnsiConsole.Write(table);

        // Diagnostic notes
        if (importance.LowVarianceCount > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Note:[/] {importance.LowVarianceCount} feature(s) have low variance.");
        }
        if (importance.HighCorrPairsCount > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Note:[/] {importance.HighCorrPairsCount} highly correlated feature pair(s) detected.");
        }
        if (importance.ConditionNumber > 1000)
        {
            AnsiConsole.MarkupLine($"[yellow]Note:[/] High condition number ({importance.ConditionNumber:F0}) -- possible multicollinearity.");
        }

        AnsiConsole.WriteLine();
    }

    public static void DisplayDistributions(DistributionReport report)
    {
        AnsiConsole.Write(new Rule("[yellow]Distribution Analysis[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var distColumns = Math.Min(report.Columns.Count, 10);
        for (int c = 0; c < distColumns; c++)
        {
            var dist = report.Columns[c];
            DisplayDistribution(dist);
        }

        if (report.Columns.Count > 10)
        {
            AnsiConsole.MarkupLine($"[grey]  ... and {report.Columns.Count - 10} more numeric columns (showing top 10)[/]");
        }
        AnsiConsole.WriteLine();
    }

    private static void DisplayDistribution(ColumnDistribution dist)
    {
        var normalStr = dist.IsNormal
            ? "[green]Normal[/]"
            : "[yellow]Non-normal[/]";

        var table = new Table().Border(TableBorder.Simple).HideHeaders();
        table.AddColumn("Test");
        table.AddColumn(new TableColumn("Statistic").RightAligned());
        table.AddColumn(new TableColumn("p-value").RightAligned());

        table.AddRow("Shapiro-Wilk", dist.SwStatistic.ToString("F4"), FormatPValue(dist.SwPValue));
        table.AddRow("Jarque-Bera", dist.JbStatistic.ToString("F4"), FormatPValue(dist.JbPValue));
        table.AddRow("Anderson-Darling", dist.AdStatistic.ToString("F4"), FormatPValue(dist.AdPValue));

        AnsiConsole.MarkupLine($"  [cyan]{dist.Name}[/]: {normalStr} (n={dist.SampleSize})");
        AnsiConsole.Write(table);
    }

    public static void DisplayOutlierSummary(OutlierReport report)
    {
        AnsiConsole.Write(new Rule("[yellow]Outlier Detection[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var pct = report.OutlierPercentage;
        var color = pct > 10 ? "red" : pct > 5 ? "yellow" : "green";

        AnsiConsole.MarkupLine($"  Outliers detected: [{color}]{report.OutlierCount}[/] / {report.TotalRows} ([{color}]{pct:F2}%[/])");

        if (report.IsolationForest != null)
        {
            AnsiConsole.MarkupLine($"  Isolation Forest threshold: {report.IsolationForest.Threshold:F4}");
        }

        if (pct > 10)
        {
            AnsiConsole.MarkupLine("[yellow]  Note:[/] High outlier rate. Consider investigating data quality.");
        }

        AnsiConsole.WriteLine();
    }

    #region Private Helpers

    private static void DisplayRegressionLabelStats(
        Dictionary<string, int> labelDistribution,
        List<KeyValuePair<string, int>> nonEmptyClasses)
    {
        AnsiConsole.Write(new Rule("[yellow]Label Statistics (Regression)[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var values = nonEmptyClasses
            .SelectMany(p => Enumerable.Repeat(
                double.Parse(p.Key, System.Globalization.CultureInfo.InvariantCulture), p.Value))
            .OrderBy(v => v)
            .ToList();

        var mean = values.Average();
        var stdDev = Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
        var median = values.Count % 2 == 0
            ? (values[values.Count / 2 - 1] + values[values.Count / 2]) / 2.0
            : values[values.Count / 2];

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Statistic");
        table.AddColumn("Value");

        table.AddRow("Count", values.Count.ToString("N0"));
        table.AddRow("Unique Values", nonEmptyClasses.Count.ToString("N0"));
        table.AddRow("Min", values.First().ToString("F4"));
        table.AddRow("Max", values.Last().ToString("F4"));
        table.AddRow("Mean", mean.ToString("F4"));
        table.AddRow("Median", median.ToString("F4"));
        table.AddRow("Std Dev", stdDev.ToString("F4"));

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        if (labelDistribution.ContainsKey("(empty)"))
        {
            var emptyCount = labelDistribution["(empty)"];
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] {emptyCount} rows have empty label values. These will be dropped during training.");
            AnsiConsole.WriteLine();
        }

        AnsiConsole.MarkupLine("[green]Info:[/] Continuous label detected -- suitable for [blue]regression[/] task.");
        AnsiConsole.MarkupLine("[grey]  Tip: [blue]mloop init . --task regression[/][/]");
        AnsiConsole.WriteLine();
    }

    private static void DisplayClassificationLabelDist(
        Dictionary<string, int> labelDistribution,
        int lineCount)
    {
        AnsiConsole.Write(new Rule("[yellow]Label Distribution[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Class");
        table.AddColumn("Count");
        table.AddColumn("Percentage");

        foreach (var pair in labelDistribution.OrderByDescending(p => p.Value))
        {
            var percent = lineCount > 0 ? (pair.Value / (double)lineCount) * 100 : 0;
            table.AddRow(pair.Key, pair.Value.ToString("N0"), $"{percent:F2}%");
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        // Imbalance warning for binary classification
        var realClasses = labelDistribution.Where(p => p.Key != "(empty)")
            .OrderByDescending(p => p.Value).ToList();
        if (realClasses.Count == 2)
        {
            var ratio = (double)realClasses[0].Value / realClasses[1].Value;
            if (ratio > 10)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] Severe class imbalance detected (ratio {ratio:F1}:1). Consider using [blue]--balance auto[/] during training.");
            }
            else if (ratio > 3)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] Class imbalance detected (ratio {ratio:F1}:1). Consider using [blue]--balance auto[/] during training.");
            }
            AnsiConsole.WriteLine();
        }

        if (labelDistribution.ContainsKey("(empty)"))
        {
            var emptyCount = labelDistribution["(empty)"];
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] {emptyCount} rows have empty label values. These will be dropped during training.");
            AnsiConsole.MarkupLine("[grey]  Tip: Clean data with [blue]mloop train --drop-missing-labels[/] (default for classification)[/]");
            AnsiConsole.WriteLine();
        }
    }

    private static string ColorCorrelation(double r)
    {
        var abs = Math.Abs(r);
        var color = abs >= 0.8 ? "red" :
                   abs >= 0.5 ? "yellow" : "grey";
        return $"[{color}]{r:F2}[/]";
    }

    private static string FormatPValue(double p)
    {
        if (p < 0.001) return "[red]<0.001[/]";
        if (p < 0.05) return $"[yellow]{p:F4}[/]";
        return $"[green]{p:F4}[/]";
    }

    private static string Truncate(string s, int maxLen)
    {
        return s.Length <= maxLen ? s : s[..(maxLen - 2)] + "..";
    }

    private static string FormatFileSize(long bytes)
    {
        string[] sizes = ["B", "KB", "MB", "GB", "TB"];
        double len = bytes;
        int order = 0;

        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }

        return $"{len:0.##} {sizes[order]}";
    }

    #endregion
}
