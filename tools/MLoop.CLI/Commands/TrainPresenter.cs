using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.DataQuality;
using MLoop.Core.Diagnostics;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Handles display/presentation logic for train command output.
/// Extracted from TrainCommand to reduce file size and improve separation of concerns.
/// </summary>
internal static class TrainPresenter
{
    /// <summary>
    /// Displays a summary of the training data (row count, column count, file size).
    /// </summary>
    public static void DisplayDataSummary(string dataFile, string labelColumn)
    {
        try
        {
            using var reader = new StreamReader(dataFile);
            var header = reader.ReadLine();
            if (string.IsNullOrEmpty(header)) return;

            var columns = CsvFieldParser.ParseFields(header).Length;
            var rowCount = 0;
            while (reader.ReadLine() != null) rowCount++;

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[blue]Data Summary[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var table = new Table()
                .BorderColor(Color.Grey)
                .AddColumn("Property")
                .AddColumn("Value");

            table.AddRow("Rows", $"[cyan]{rowCount:N0}[/]");
            table.AddRow("Columns", $"[cyan]{columns}[/]");
            table.AddRow("Features", $"[cyan]{columns - 1}[/]");
            table.AddRow("Label", $"[cyan]{labelColumn}[/]");

            // Check file size
            var fileSize = new FileInfo(dataFile).Length;
            var sizeStr = fileSize switch
            {
                < 1024 => $"{fileSize} B",
                < 1024 * 1024 => $"{fileSize / 1024.0:F1} KB",
                _ => $"{fileSize / (1024.0 * 1024.0):F1} MB"
            };
            table.AddRow("File Size", $"[grey]{sizeStr}[/]");

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
        }
        catch
        {
            // Non-critical — skip data summary on error
        }
    }

    /// <summary>
    /// Displays the training configuration before training starts.
    /// </summary>
    public static void DisplayTrainingConfig(string dataFile, string modelName, ModelDefinition definition)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[blue]Training Configuration[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Setting")
            .AddColumn("Value");

        table.AddRow("Model", $"[cyan]{modelName}[/]");
        table.AddRow("Task", definition.Task);
        table.AddRow("Data File", Path.GetFileName(dataFile));
        table.AddRow("Label Column", definition.Label);
        table.AddRow("Time Limit", $"{definition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds}s");
        table.AddRow("Metric", definition.Training?.Metric ?? ConfigDefaults.DefaultMetric);
        table.AddRow("Test Split", $"{(definition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit) * 100:F0}%");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays training results including metrics table and next steps.
    /// </summary>
    public static void DisplayResults(TrainingResult result, string modelName)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Training Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment ID: [blue]{result.ExperimentId}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Best Trainer: [yellow]{result.BestTrainer}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Training Time: [cyan]{result.TrainingTimeSeconds:F2}s[/]");
        AnsiConsole.WriteLine();

        // Metrics table
        var metricsTable = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Metric")
            .AddColumn("Value");

        foreach (var (metricName, metricValue) in result.Metrics.OrderByDescending(m => m.Value))
        {
            var formattedValue = metricValue.ToString("F4");
            var color = metricValue >= 0.9 ? "green" :
                       metricValue >= 0.8 ? "yellow" :
                       metricValue >= 0.7 ? "orange1" : "red";

            metricsTable.AddRow(
                metricName.Replace("_", " ").ToUpperInvariant(),
                $"[{color}]{formattedValue}[/]");
        }

        AnsiConsole.Write(metricsTable);
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[grey]Model saved to:[/] {result.ModelPath}");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
        AnsiConsole.MarkupLine($"  mloop list --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop predict data.csv --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop promote {result.ExperimentId} --name {modelName}");
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays performance diagnostics warnings and suggestions.
    /// </summary>
    public static void DisplayDiagnostics(PerformanceDiagnosticResult diagnosticResult)
    {
        if (!diagnosticResult.NeedsAttention) return;

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[yellow]Performance Diagnostics[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var levelColor = diagnosticResult.OverallAssessment switch
        {
            PerformanceLevel.Poor => "red",
            PerformanceLevel.Low => "yellow",
            _ => "orange1"
        };

        AnsiConsole.MarkupLine($"[{levelColor}]{Emoji.Known.Warning} {diagnosticResult.Summary}[/]");
        AnsiConsole.WriteLine();

        if (diagnosticResult.Warnings.Count > 0)
        {
            AnsiConsole.MarkupLine("[yellow]Warnings:[/]");
            foreach (var warning in diagnosticResult.Warnings)
            {
                AnsiConsole.MarkupLine($"  [yellow]{Emoji.Known.SmallBlueDiamond}[/] {warning}");
            }
            AnsiConsole.WriteLine();
        }

        if (diagnosticResult.Suggestions.Count > 0)
        {
            AnsiConsole.MarkupLine("[blue]Suggestions to improve performance:[/]");
            foreach (var suggestion in diagnosticResult.Suggestions)
            {
                AnsiConsole.MarkupLine($"  [grey]{Emoji.Known.SmallBlueDiamond}[/] {suggestion}");
            }
            AnsiConsole.WriteLine();
        }
    }

    /// <summary>
    /// Displays comparison table between new and production model metrics.
    /// </summary>
    public static void DisplayProductionComparison(
        TrainingResult result,
        ModelInfo production)
    {
        AnsiConsole.Write(new Rule("[blue]Production Comparison[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var compTable = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Metric")
            .AddColumn(new TableColumn($"[grey]Production ({production.ExperimentId})[/]").RightAligned())
            .AddColumn(new TableColumn($"[cyan]New ({result.ExperimentId})[/]").RightAligned())
            .AddColumn(new TableColumn("Delta").RightAligned());

        foreach (var (metricName, newValue) in result.Metrics.OrderByDescending(m => m.Value))
        {
            var prodValue = production.Metrics!.TryGetValue(metricName, out var pv) ? pv : (double?)null;
            var prodStr = prodValue.HasValue ? $"{prodValue.Value:F4}" : "[grey]-[/]";
            var newStr = $"{newValue:F4}";

            string deltaStr;
            if (prodValue.HasValue)
            {
                var delta = newValue - prodValue.Value;
                var sign = delta >= 0 ? "+" : "";
                var color = delta > 0 ? "green" : delta < 0 ? "red" : "grey";
                deltaStr = $"[{color}]{sign}{delta:F4}[/]";
            }
            else
            {
                deltaStr = "[grey]-[/]";
            }

            compTable.AddRow(
                metricName.Replace("_", " ").ToUpperInvariant(),
                prodStr, newStr, deltaStr);
        }

        AnsiConsole.Write(compTable);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays promotion result messages.
    /// </summary>
    public static void DisplayPromotionResult(
        bool promoted,
        string primaryMetric,
        TrainingResult result,
        ModelInfo? production,
        double? minThreshold)
    {
        if (promoted)
        {
            AnsiConsole.MarkupLine("[green]Model promoted to production![/]");
            AnsiConsole.WriteLine($"   Better {primaryMetric} than current production model");
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]Model saved to staging[/]");

            if (minThreshold.HasValue && result.Metrics != null &&
                result.Metrics.TryGetValue(primaryMetric, out var metricValue) &&
                metricValue < minThreshold.Value)
            {
                AnsiConsole.MarkupLine($"[red]   Model {primaryMetric} ({metricValue:F4}) is below minimum threshold ({minThreshold.Value:F4})[/]");
                AnsiConsole.MarkupLine("[grey]   Tip: Model performance is near-random. Check data quality and feature relevance.[/]");
            }
            else if (production?.Metrics != null && result.Metrics != null && production.Metrics.ContainsKey(primaryMetric))
            {
                var prodValue = production.Metrics[primaryMetric];
                var newValue = result.Metrics.TryGetValue(primaryMetric, out var v) ? v : double.NaN;
                if (Math.Abs(prodValue - newValue) < 1e-10)
                {
                    AnsiConsole.WriteLine($"   Performance converged — same {primaryMetric} as production model");
                    AnsiConsole.MarkupLine("[grey]   Tip: Additional training time may not improve this dataset further.[/]");
                }
                else
                {
                    AnsiConsole.WriteLine($"   Current production model has better {primaryMetric}");
                }
            }
        }

        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays unused data file warnings.
    /// </summary>
    public static void DisplayUnusedDataWarning(UnusedDataScanResult scanResult)
    {
        if (!scanResult.HasUnusedFiles) return;

        AnsiConsole.Write(new Rule("[grey]Data Directory Summary[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]{scanResult.Summary}[/]");

        foreach (var warning in scanResult.Warnings.Take(3))
        {
            AnsiConsole.MarkupLine($"[yellow]  {Emoji.Known.Warning} {warning}[/]");
        }

        foreach (var suggestion in scanResult.Suggestions.Take(2))
        {
            AnsiConsole.MarkupLine($"[grey]  {Emoji.Known.LightBulb} {suggestion}[/]");
        }

        if (scanResult.UnusedFiles.Count <= 5)
        {
            AnsiConsole.MarkupLine("[grey]  Unused files:[/]");
            foreach (var file in scanResult.UnusedFiles)
            {
                AnsiConsole.MarkupLine($"[grey]    {Emoji.Known.SmallBlueDiamond} {file.FileName} ({file.SizeFormatted})[/]");
            }
        }
        else
        {
            var remaining = scanResult.UnusedFiles.Count - 3;
            AnsiConsole.MarkupLine("[grey]  Unused files:[/]");
            foreach (var file in scanResult.UnusedFiles.Take(3))
            {
                AnsiConsole.MarkupLine($"[grey]    {Emoji.Known.SmallBlueDiamond} {file.FileName} ({file.SizeFormatted})[/]");
            }
            AnsiConsole.MarkupLine($"[grey]    ... and {remaining} more[/]");
        }
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays class distribution analysis results for classification tasks.
    /// </summary>
    public static void DisplayClassDistribution(ClassDistributionResult distributionResult)
    {
        if (distributionResult.Error != null) return;

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[blue]Class Distribution[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.WriteLine(distributionResult.DistributionVisualization);
        AnsiConsole.WriteLine();

        if (distributionResult.NeedsAttention)
        {
            AnsiConsole.MarkupLine($"[yellow]{Emoji.Known.Warning} {distributionResult.Summary}[/]");

            foreach (var warning in distributionResult.Warnings)
            {
                AnsiConsole.MarkupLine($"[yellow]  {Emoji.Known.SmallBlueDiamond} {warning}[/]");
            }

            if (distributionResult.Suggestions.Count > 0)
            {
                AnsiConsole.MarkupLine("[grey]Suggestions:[/]");
                foreach (var suggestion in distributionResult.Suggestions)
                {
                    AnsiConsole.MarkupLine($"[grey]  {Emoji.Known.SmallBlueDiamond} {suggestion}[/]");
                }
            }
            AnsiConsole.WriteLine();
        }
        else
        {
            AnsiConsole.MarkupLine($"[green]{Emoji.Known.CheckMark}[/] {distributionResult.Summary}");
            AnsiConsole.WriteLine();
        }
    }

    /// <summary>
    /// Displays data quality issues grouped by severity.
    /// </summary>
    public static void DisplayDataQualityIssues(List<DataQualityIssue> issues)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[blue]Data Quality Analysis[/]").LeftJustified());
        AnsiConsole.WriteLine();

        if (issues.Count == 0)
        {
            AnsiConsole.MarkupLine($"[green]{Emoji.Known.CheckMark}[/] No data quality issues detected!");
            AnsiConsole.WriteLine();
            return;
        }

        var criticalIssues = issues.Where(i => i.Severity == IssueSeverity.Critical).ToList();
        var highIssues = issues.Where(i => i.Severity == IssueSeverity.High).ToList();
        var mediumIssues = issues.Where(i => i.Severity == IssueSeverity.Medium).ToList();
        var lowIssues = issues.Where(i => i.Severity == IssueSeverity.Low).ToList();

        if (criticalIssues.Count > 0)
        {
            AnsiConsole.MarkupLine("[red]CRITICAL Issues:[/]");
            foreach (var issue in criticalIssues)
            {
                AnsiConsole.MarkupLine($"  [red]{Emoji.Known.SmallBlueDiamond}[/] {issue.Description}");
                if (!string.IsNullOrEmpty(issue.SuggestedFix))
                {
                    AnsiConsole.MarkupLine($"    [grey]Fix: {issue.SuggestedFix.Replace("\n", "\n    ")}[/]");
                }
            }
            AnsiConsole.WriteLine();
        }

        if (highIssues.Count > 0)
        {
            AnsiConsole.MarkupLine("[yellow]HIGH Priority Issues:[/]");
            foreach (var issue in highIssues)
            {
                AnsiConsole.MarkupLine($"  [yellow]{Emoji.Known.SmallBlueDiamond}[/] {issue.Description}");
                if (!string.IsNullOrEmpty(issue.SuggestedFix))
                {
                    AnsiConsole.MarkupLine($"    [grey]Fix: {issue.SuggestedFix.Replace("\n", "\n    ")}[/]");
                }
            }
            AnsiConsole.WriteLine();
        }

        if (mediumIssues.Count > 0)
        {
            AnsiConsole.MarkupLine("[blue]MEDIUM Priority Issues:[/]");
            foreach (var issue in mediumIssues)
            {
                AnsiConsole.MarkupLine($"  [blue]{Emoji.Known.SmallBlueDiamond}[/] {issue.Description}");
            }
            AnsiConsole.WriteLine();
        }

        if (lowIssues.Count > 0)
        {
            AnsiConsole.MarkupLine("[grey]LOW Priority Issues: {0}[/]", lowIssues.Count);
            AnsiConsole.WriteLine();
        }
    }
}
