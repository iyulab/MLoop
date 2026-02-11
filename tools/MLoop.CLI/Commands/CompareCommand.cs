using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop compare - Compare experiments side by side
/// </summary>
public static class CompareCommand
{
    public static Command Create()
    {
        var experimentsArgument = new Argument<string[]>("experiments")
        {
            Description = "Experiment IDs to compare (e.g., exp-001 exp-002)",
            Arity = ArgumentArity.ZeroOrMore
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = "Model name (required when comparing all experiments)"
        };

        var bestOption = new Option<int?>("--best", "-b")
        {
            Description = "Compare top N experiments by metric (default: all completed)"
        };

        var metricOption = new Option<string?>("--metric", "-m")
        {
            Description = "Sort by specific metric (default: best metric from training)"
        };

        var command = new Command("compare", "Compare experiments side by side");
        command.Arguments.Add(experimentsArgument);
        command.Options.Add(nameOption);
        command.Options.Add(bestOption);
        command.Options.Add(metricOption);

        command.SetAction((parseResult) =>
        {
            var experiments = parseResult.GetValue(experimentsArgument) ?? [];
            var name = parseResult.GetValue(nameOption);
            var best = parseResult.GetValue(bestOption);
            var metric = parseResult.GetValue(metricOption);
            return ExecuteAsync(experiments, name, best, metric);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string[] experimentIds,
        string? modelName,
        int? bestCount,
        string? sortMetric)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            // Determine which experiments to compare
            List<ExperimentData> experimentsToCompare;

            // Support comma-separated experiment IDs (e.g., "exp-001,exp-002")
            experimentIds = experimentIds
                .SelectMany(id => id.Split(',', StringSplitOptions.RemoveEmptyEntries))
                .Select(id => id.Trim())
                .ToArray();

            if (experimentIds.Length > 0)
            {
                // Specific experiments provided
                var resolvedModelName = CommandContext.ResolveModelName(modelName);

                experimentsToCompare = [];
                foreach (var expId in experimentIds)
                {
                    try
                    {
                        var exp = await ctx.ExperimentStore.LoadAsync(resolvedModelName, expId, CancellationToken.None);
                        experimentsToCompare.Add(exp);
                    }
                    catch (FileNotFoundException)
                    {
                        AnsiConsole.MarkupLine($"[yellow]Warning:[/] Experiment '{expId}' not found for model '{resolvedModelName}'");
                    }
                }
            }
            else
            {
                // Compare all experiments for the model
                if (string.IsNullOrWhiteSpace(modelName))
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] Model name is required when not specifying experiment IDs.");
                    AnsiConsole.MarkupLine("Usage: [blue]mloop compare exp-001 exp-002[/] or [blue]mloop compare --name <model>[/]");
                    return 1;
                }

                var resolvedModelName = CommandContext.ResolveModelName(modelName);
                var summaries = await ctx.ExperimentStore.ListAsync(resolvedModelName, CancellationToken.None);
                var completedSummaries = summaries
                    .Where(s => s.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(s => s.BestMetric ?? 0)
                    .ToList();

                if (bestCount.HasValue && bestCount.Value > 0)
                {
                    completedSummaries = completedSummaries.Take(bestCount.Value).ToList();
                }

                experimentsToCompare = [];
                foreach (var summary in completedSummaries)
                {
                    var exp = await ctx.ExperimentStore.LoadAsync(resolvedModelName, summary.ExperimentId, CancellationToken.None);
                    experimentsToCompare.Add(exp);
                }
            }

            if (experimentsToCompare.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No experiments to compare.[/]");
                return 0;
            }

            if (experimentsToCompare.Count == 1)
            {
                AnsiConsole.MarkupLine("[yellow]Only one experiment found. Need at least 2 experiments to compare.[/]");
                return 0;
            }

            // Get production model for highlighting
            var productionModels = await ctx.ModelRegistry.ListAsync(null, CancellationToken.None);
            var productionDict = productionModels.ToDictionary(m => m.ModelName, m => m.ExperimentId);

            // Collect all metrics across experiments
            var allMetricNames = experimentsToCompare
                .Where(e => e.Metrics != null)
                .SelectMany(e => e.Metrics!.Keys)
                .Distinct()
                .OrderBy(m => m)
                .ToList();

            // Sort experiments by metric if specified
            if (!string.IsNullOrEmpty(sortMetric) && allMetricNames.Contains(sortMetric))
            {
                experimentsToCompare = experimentsToCompare
                    .OrderByDescending(e => e.Metrics?.GetValueOrDefault(sortMetric, 0) ?? 0)
                    .ToList();
            }

            // Display comparison
            AnsiConsole.WriteLine();
            var title = $"[bold]Experiment Comparison[/]";
            AnsiConsole.Write(new Rule(title).LeftJustified());
            AnsiConsole.WriteLine();

            // Create comparison table
            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            // Add metric column
            table.AddColumn(new TableColumn("[bold]Metric[/]").LeftAligned());

            // Add experiment columns
            foreach (var exp in experimentsToCompare)
            {
                var isProduction = productionDict.TryGetValue(exp.ModelName, out var prodExpId)
                    && exp.ExperimentId == prodExpId;

                var header = isProduction
                    ? $"[green bold]{exp.ExperimentId}[/]\n[green](Production)[/]"
                    : $"[cyan]{exp.ExperimentId}[/]";

                table.AddColumn(new TableColumn(header).Centered());
            }

            // Add basic info rows
            AddComparisonRow(table, "Model", experimentsToCompare, e => $"[blue]{e.ModelName}[/]");
            AddComparisonRow(table, "Task", experimentsToCompare, e => e.Task);
            AddComparisonRow(table, "Status", experimentsToCompare, e =>
                e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)
                    ? "[green]Completed[/]"
                    : $"[yellow]{e.Status}[/]");
            AddComparisonRow(table, "Timestamp", experimentsToCompare,
                e => e.Timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm"));
            AddComparisonRow(table, "Time Limit", experimentsToCompare,
                e => $"{e.Config.TimeLimitSeconds}s");
            AddComparisonRow(table, "Label Column", experimentsToCompare,
                e => e.Config.LabelColumn ?? "-");

            // Add separator
            table.AddEmptyRow();

            // Add metrics rows with highlighting for best values
            if (allMetricNames.Count > 0)
            {
                foreach (var metricName in allMetricNames)
                {
                    var values = experimentsToCompare
                        .Select(e => e.Metrics?.GetValueOrDefault(metricName))
                        .ToList();

                    // Find best value (handle both higher-is-better and lower-is-better)
                    var isLowerBetter = metricName.Contains("Loss", StringComparison.OrdinalIgnoreCase) ||
                                        metricName.Contains("Error", StringComparison.OrdinalIgnoreCase) ||
                                        metricName.Contains("MAE", StringComparison.OrdinalIgnoreCase) ||
                                        metricName.Contains("MSE", StringComparison.OrdinalIgnoreCase) ||
                                        metricName.Contains("RMSE", StringComparison.OrdinalIgnoreCase);

                    double? bestValue = null;
                    if (values.Any(v => v.HasValue))
                    {
                        var nonNullValues = values.Where(v => v.HasValue).Select(v => v!.Value);
                        bestValue = isLowerBetter ? nonNullValues.Min() : nonNullValues.Max();
                    }

                    var row = new List<string> { $"[bold]{metricName}[/]" };
                    foreach (var value in values)
                    {
                        if (value.HasValue)
                        {
                            var isBest = Math.Abs(value.Value - (bestValue ?? 0)) < 0.0001;
                            var formatted = value.Value.ToString("F4");
                            row.Add(isBest ? $"[green bold]{formatted}[/]" : $"[yellow]{formatted}[/]");
                        }
                        else
                        {
                            row.Add("[grey]-[/]");
                        }
                    }

                    table.AddRow(row.ToArray());
                }
            }
            else
            {
                table.AddRow("[grey]No metrics available[/]");
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Summary
            AnsiConsole.MarkupLine($"[grey]Compared {experimentsToCompare.Count} experiments[/]");

            // Recommendation: find the experiment with the best primary metric
            if (allMetricNames.Count > 0)
            {
                var primaryMetric = sortMetric ?? allMetricNames.First();
                var isLowerBetter = primaryMetric.Contains("Loss", StringComparison.OrdinalIgnoreCase) ||
                                    primaryMetric.Contains("Error", StringComparison.OrdinalIgnoreCase) ||
                                    primaryMetric.Contains("MAE", StringComparison.OrdinalIgnoreCase) ||
                                    primaryMetric.Contains("MSE", StringComparison.OrdinalIgnoreCase) ||
                                    primaryMetric.Contains("RMSE", StringComparison.OrdinalIgnoreCase);

                var rankedExperiments = experimentsToCompare
                    .Where(e => e.Metrics?.ContainsKey(primaryMetric) == true)
                    .OrderBy(e => isLowerBetter ? e.Metrics![primaryMetric] : -e.Metrics![primaryMetric])
                    .ToList();

                if (rankedExperiments.Count > 0)
                {
                    var bestExp = rankedExperiments.First();
                    var isProduction = productionDict.TryGetValue(bestExp.ModelName, out var prodExpId)
                        && bestExp.ExperimentId == prodExpId;

                    if (!isProduction)
                    {
                        AnsiConsole.MarkupLine($"[grey]Recommendation: Consider promoting [cyan]{bestExp.ExperimentId}[/] to production (best {primaryMetric}: {bestExp.Metrics![primaryMetric]:F4})[/]");
                        AnsiConsole.MarkupLine($"[grey]  Run: [blue]mloop promote {bestExp.ExperimentId} --name {bestExp.ModelName}[/][/]");
                    }
                    else
                    {
                        AnsiConsole.MarkupLine($"[grey]Best experiment [cyan]{bestExp.ExperimentId}[/] is already in production ({primaryMetric}: {bestExp.Metrics![primaryMetric]:F4})[/]");
                    }
                }
            }

            AnsiConsole.WriteLine();
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "compare");
            return 1;
        }
    }

    private static void AddComparisonRow(
        Table table,
        string label,
        List<ExperimentData> experiments,
        Func<ExperimentData, string> valueSelector)
    {
        var row = new List<string> { $"[bold]{label}[/]" };
        row.AddRange(experiments.Select(valueSelector));
        table.AddRow(row.ToArray());
    }
}
