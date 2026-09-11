using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.DataQuality;
using MLoop.Core.Prediction;
using MLoop.Core.Diagnostics;
using MLoop.Core.Models;
using MLoop.Core.Storage;
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
        catch (Exception)
        {
            // Non-critical — skip data summary on error
        }
    }

    /// <summary>
    /// Displays the training configuration before training starts.
    /// </summary>
    public static void DisplayTrainingConfig(
        string dataFile, string modelName, ModelDefinition definition, string? testDataFile = null, bool useAutoTime = false)
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
        table.AddRow("Time Limit", useAutoTime
            ? "[cyan]auto[/] (estimated from data size)"
            : $"{definition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds}s");
        table.AddRow("Metric", definition.Training?.Metric ?? ConfigDefaults.DefaultMetric);

        if (testDataFile != null)
        {
            table.AddRow("Test Split", $"Pre-split ([cyan]{Path.GetFileName(testDataFile)}[/])");
        }
        else
        {
            table.AddRow("Test Split", $"{(definition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit) * 100:F0}%");
        }

        if (definition.Columns is { Count: > 0 })
        {
            var overrideSummary = string.Join(", ",
                definition.Columns.Select(c => $"{c.Key}→{c.Value.Type}"));
            table.AddRow("Column Overrides", $"[yellow]{Markup.Escape(overrideSummary)}[/]");
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// The trainer as a one-line summary: the trainer itself, not the whole pipeline AutoML
    /// assembled, plus whatever qualifies it (hyperparameters, a fallback reason).
    /// </summary>
    /// <remarks>
    /// A full pipeline name is one unbreakable 68-character token
    /// (<c>ReplaceMissingValues=&gt;FeaturizeText=&gt;Concatenate=&gt;LightGbmRegression</c>), and
    /// <c>"&gt; Best Trainer: "</c> plus that exceeds the 80 columns Spectre assumes when stdout is
    /// redirected. It wrapped, leaving the label alone on its line — which reads as a missing value,
    /// and was reported as one by a downstream consumer. Shortening is not a workaround for the
    /// wrap: a summary line should say which trainer won, and the assembled pipeline is still
    /// carried in full by <c>--json</c>, <c>metadata.json</c> and <c>mloop list</c>.
    /// </remarks>
    private static string SummarizeTrainer(TrainerDescriptor trainer) =>
        (trainer with { Name = TrainingProgressTracker.ShortTrainerName(trainer.Name) }).Display;

    /// <summary>
    /// Displays training results including metrics table and next steps.
    /// </summary>
    /// <summary>
    /// What a user is told to do once training has finished.
    /// </summary>
    /// <remarks>
    /// These lines are printed before auto-promotion runs, which is why the promotion step has to be
    /// asked about rather than always offered: with promotion enabled — the default — the list told
    /// the reader to run <c>mloop promote exp-001</c> and the very next lines reported the model as
    /// already promoted. Read top to bottom, the command asked for something it had just done.
    /// </remarks>
    internal static string[] NextStepsAfterTraining(string modelName, string experimentId, bool promotionFollows)
    {
        string[] always = [$"mloop list --name {modelName}", $"mloop predict data.csv --name {modelName}"];
        return promotionFollows ? always : [.. always, $"mloop promote {experimentId} --name {modelName}"];
    }

    /// <param name="promotionFollows">
    /// Whether this command is about to promote the experiment itself. The next steps are printed
    /// before the promotion runs, so with this false the reader is told to run a command that the
    /// very next lines report as already done.
    /// </param>
    public static void DisplayResults(TrainingResult result, string modelName, bool promotionFollows = false)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Training Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment ID: [blue]{result.ExperimentId}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Best Trainer: [yellow]{Markup.Escape(SummarizeTrainer(result.Trainer))}[/]");
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

        ValueLine.Write("[grey]Model saved to:[/] ", WhereItLandedInTheProject(result.ModelPath));
        if (result.ReportPath is not null)
            ValueLine.Write("[grey]Report:[/]         ", WhereItLandedInTheProject(result.ReportPath));
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
        foreach (var step in NextStepsAfterTraining(modelName, result.ExperimentId, promotionFollows))
        {
            AnsiConsole.MarkupLine($"  {step}");
        }
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Displays performance diagnostics warnings and suggestions.
    /// </summary>
    /// <summary>
    /// The saved model's path as it reads from inside the project.
    /// </summary>
    /// <remarks>
    /// The absolute path is long enough that a redirected stdout (80 columns) folds it mid-token,
    /// splitting <c>model.zip</c> across two lines — a path a user copies out and finds broken, the
    /// same way an over-long summary value once read as missing. From the <c>models</c> directory
    /// down is the part that identifies the artifact, it is what every command that takes an
    /// experiment already speaks in, and it fits. A path that does not run through this project's
    /// layout is left whole rather than guessed at.
    /// </remarks>
    internal static string WhereItLandedInTheProject(string modelPath)
    {
        var marker = Path.DirectorySeparatorChar + ExperimentLayout.ModelsDirectory + Path.DirectorySeparatorChar;
        var at = modelPath.LastIndexOf(marker, StringComparison.OrdinalIgnoreCase);
        return at < 0 ? modelPath : modelPath[(at + 1)..];
    }

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
        double? minThreshold,
        double? majorityClassRatio = null)
    {
        if (promoted)
        {
            AnsiConsole.MarkupLine("[green]Model promoted to production![/]");

            // The reason has to be true of the run that just happened. On a project's first
            // training there is no production model, so "better than current production" states a
            // comparison that was never made — and that is the very first thing a new user is told.
            // The comparison is available whenever it did happen, so it is shown rather than
            // asserted: both numbers travel, the way every other threshold message here reports.
            var previous = production?.Metrics is { } m && m.TryGetValue(primaryMetric, out var p)
                ? p
                : (double?)null;
            var current = result.Metrics != null
                && result.Metrics.TryGetValue(primaryMetric, out var c)
                ? c
                : (double?)null;

            if (production is null)
                AnsiConsole.WriteLine("   First model for this project — nothing to compare against yet");
            else if (previous.HasValue && current.HasValue)
                AnsiConsole.WriteLine(
                    $"   Better {primaryMetric} than {production.ExperimentId}: "
                    + $"{previous.Value:F4} -> {current.Value:F4}");
            else
                AnsiConsole.WriteLine($"   Better {primaryMetric} than {production.ExperimentId}");
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]Model saved to staging[/]");

            if (minThreshold.HasValue && result.Metrics != null &&
                result.Metrics.TryGetValue(primaryMetric, out var metricValue) &&
                metricValue < minThreshold.Value)
            {
                // The quality gate blocking promotion is a warning-class fact: it goes through the
                // warning seam so a --json consumer receives it as an event instead of losing it
                // with the silenced narration.
                // Name the opponent the model lost to. "Near-random" was the only explanation on
                // offer, and on imbalanced data it is the wrong one — a model can sit far above
                // chance and still be beaten by answering with the most common class every time.
                var beatenBy = majorityClassRatio.HasValue && minThreshold.Value >= majorityClassRatio.Value - 1e-9
                    ? $"always predicting the most common class scores {majorityClassRatio.Value:P1} here"
                    : "that floor is the accuracy of random guessing";
                WarningConsole.Warn(
                    $"Model {primaryMetric} ({metricValue:F4}) is below minimum threshold ({minThreshold.Value:F4}) " +
                    $"— saved to staging, not promoted. The threshold is what a model that learned nothing would score: " +
                    $"{beatenBy}. Check data quality, feature relevance, and class balance.");
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
                AnsiConsole.MarkupLine($"[grey]    {Emoji.Known.SmallBlueDiamond} {file.DisplayName} ({file.SizeFormatted})[/]");
            }
        }
        else
        {
            var remaining = scanResult.UnusedFiles.Count - 3;
            AnsiConsole.MarkupLine("[grey]  Unused files:[/]");
            foreach (var file in scanResult.UnusedFiles.Take(3))
            {
                AnsiConsole.MarkupLine($"[grey]    {Emoji.Known.SmallBlueDiamond} {file.DisplayName} ({file.SizeFormatted})[/]");
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

    /// <summary>
    /// Displays the auto-time static estimate summary.
    /// </summary>
    public static void DisplayAutoTimeEstimate(
        int staticEstimate, int rows, int columns, string task, int classCount)
    {
        AnsiConsole.MarkupLine(
            $"[cyan]⏱ Auto-time:[/] {staticEstimate}s estimated " +
            $"({rows:N0} rows × {columns} cols, {task}" +
            (classCount > 0 ? $", {classCount} classes" : "") + ")");
        AnsiConsole.MarkupLine("  [dim]Tip: Override with --time <seconds>[/]");
    }

    /// <summary>
    /// Displays probe and main training phase results.
    /// </summary>
    public static void DisplayProbeResult(int probeTime, double bestMetric, int trials, int finalTime)
    {
        AnsiConsole.MarkupLine(
            $"  Phase 1: Quick probe ({probeTime}s) → {bestMetric:F4}, {trials} trials");
        AnsiConsole.MarkupLine(
            $"  Phase 2: Main training ({finalTime}s)");
    }

    /// <summary>
    /// Displays message when probe phase converged early.
    /// </summary>
    public static void DisplayProbeConverged(int probeTime, double bestMetric)
    {
        AnsiConsole.MarkupLine(
            $"  [green]✓[/] Converged in probe phase ({probeTime}s, {bestMetric:F4}) — skipping main training");
    }

    /// <summary>
    /// Says why main training did not run when the probe fell back to a direct pipeline. Without
    /// this the run announces "Phase 1" and then simply ends, which reads as the phase having been
    /// dropped rather than deliberately skipped.
    /// </summary>
    public static void DisplayProbeFellBack(int probeTime, double bestMetric)
    {
        AnsiConsole.MarkupLine(
            $"  [yellow]![/] AutoML could not run on this data ({probeTime}s probe, {bestMetric:F4}) — "
            + "trained a direct pipeline instead, and skipped main training since a longer search "
            + "would fail the same way");
    }
}
