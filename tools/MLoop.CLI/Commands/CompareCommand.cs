using System.CommandLine;
using System.Text.Json;
using MLoop.Core.Evaluation;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
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

        var metricsFileOption = new Option<string?>("--metrics-file")
        {
            Description = "Provided-state mode: rank candidates and select the best from a JSON file of " +
                          "candidate metrics, without a local .mloop/ project (for distributed/external-SoR " +
                          "consumers). Use '-' to read from stdin. Implies --json. " +
                          "Input: [{\"id\":\"exp-a\",\"metrics\":{\"r_squared\":0.87}}, ...]"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the result as JSON (machine-readable)"
        };

        var command = new Command("compare", "Compare experiments side by side");
        command.Arguments.Add(experimentsArgument);
        command.Options.Add(nameOption);
        command.Options.Add(bestOption);
        command.Options.Add(metricOption);
        command.Options.Add(metricsFileOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var experiments = parseResult.GetValue(experimentsArgument) ?? [];
            var name = parseResult.GetValue(nameOption);
            var best = parseResult.GetValue(bestOption);
            var metric = parseResult.GetValue(metricOption);
            var metricsFile = parseResult.GetValue(metricsFileOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(experiments, name, best, metric, metricsFile, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string[] experimentIds,
        string? modelName,
        int? bestCount,
        string? sortMetric,
        string? metricsFile,
        bool jsonOutput)
    {
        // Provided-state mode: rank/select from supplied metrics with no local .mloop/ project.
        // Read/decide only — never touches project state — so distributed consumers (external SoR)
        // can delegate mloop's direction-aware best-selection instead of re-implementing it.
        if (metricsFile != null)
            return await ExecuteProvidedStateAsync(metricsFile, sortMetric);

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
                // F-28: rank best-first by metric direction (not unconditionally descending), so
                // lower-is-better tasks aren't ordered worst-first.
                var completedSummaries = ExperimentRanking.OrderByQuality(
                        summaries.Where(s => s.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)))
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

            // Sort experiments by metric: --metric flag, or optimizing metric from config
            var effectiveSortMetric = sortMetric;
            if (string.IsNullOrEmpty(effectiveSortMetric))
            {
                effectiveSortMetric = experimentsToCompare
                    .Select(e => e.Config.Metric)
                    .Where(m => !string.IsNullOrEmpty(m))
                    .FirstOrDefault();
            }

            // Resolve user-facing alias (e.g. "f1") to the canonical stored key ("f1_score")
            // so --sort / config metric isn't silently ignored when the alias differs.
            var sortKey = string.IsNullOrEmpty(effectiveSortMetric)
                ? null
                : MetricPolicy.ResolveMetricKey(effectiveSortMetric, allMetricNames);
            if (sortKey != null)
            {
                var sortLowerBetter = MLoop.Core.Evaluation.MetricDirection.IsLowerBetter(sortKey);

                experimentsToCompare = experimentsToCompare
                    .OrderBy(e => sortLowerBetter
                        ? (e.Metrics?.GetValueOrDefault(sortKey, 0) ?? 0)
                        : -(e.Metrics?.GetValueOrDefault(sortKey, 0) ?? 0))
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
                    var isLowerBetter = MLoop.Core.Evaluation.MetricDirection.IsLowerBetter(metricName);

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
                // Use --metric flag if specified, otherwise read from experiment metadata.
                // Resolve aliases ("f1" → "f1_score") to the actual stored key; fall back to
                // the first available metric when the requested one isn't present.
                var requestedMetric = !string.IsNullOrEmpty(sortMetric)
                    ? sortMetric
                    : experimentsToCompare
                        .Select(e => e.Config.Metric)
                        .FirstOrDefault(m => !string.IsNullOrEmpty(m));

                var primaryMetric = (requestedMetric != null
                        ? MetricPolicy.ResolveMetricKey(requestedMetric, allMetricNames)
                        : null)
                    ?? allMetricNames.First();

                var isLowerBetter = MLoop.Core.Evaluation.MetricDirection.IsLowerBetter(primaryMetric);

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

    // ---- Provided-state comparison (no local .mloop/) ----

    /// <summary>A candidate to rank, supplied by an external caller (its own SoR).</summary>
    public sealed record ProvidedCandidate(string Id, Dictionary<string, double> Metrics);

    public sealed record ProvidedRankEntry(string Id, double Value);

    public sealed record ProvidedCompareResult(
        string Metric, string Direction, string Best, IReadOnlyList<ProvidedRankEntry> Ranking);

    private static readonly JsonSerializerOptions ProvidedJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// Direction-aware ranking/best-selection over caller-supplied metrics — the same authority
    /// (<see cref="MetricDirection"/>) the local-state path uses, so a distributed consumer gets
    /// identical semantics without re-implementing them. Pure; no filesystem/project state.
    /// </summary>
    internal static ProvidedCompareResult CompareProvidedMetrics(
        IReadOnlyList<ProvidedCandidate> candidates, string? metric)
    {
        if (candidates.Count == 0)
            throw new ArgumentException("No candidates provided.");

        var allKeys = candidates.SelectMany(c => c.Metrics.Keys).Distinct().ToList();

        string metricKey;
        if (!string.IsNullOrWhiteSpace(metric))
        {
            metricKey = MetricPolicy.ResolveMetricKey(metric, allKeys)
                ?? throw new ArgumentException(
                    $"Metric '{metric}' not found in provided candidates. Available: {string.Join(", ", allKeys)}");
        }
        else
        {
            // Comparing across different metrics is meaningless; require a single shared metric.
            var common = allKeys.Where(k => candidates.All(c => c.Metrics.ContainsKey(k))).ToList();
            if (common.Count != 1)
                throw new ArgumentException(common.Count == 0
                    ? "No metric is present on all candidates; specify --metric."
                    : $"Multiple metrics present ({string.Join(", ", common)}); specify --metric.");
            metricKey = common[0];
        }

        var scored = candidates.Where(c => c.Metrics.ContainsKey(metricKey)).ToList();
        if (scored.Count == 0)
            throw new ArgumentException($"No candidate has metric '{metricKey}'.");

        var lowerBetter = MetricDirection.IsLowerBetter(metricKey);
        var ranked = (lowerBetter
                ? scored.OrderBy(c => c.Metrics[metricKey])
                : scored.OrderByDescending(c => c.Metrics[metricKey]))
            .ToList();

        return new ProvidedCompareResult(
            metricKey,
            lowerBetter ? "minimize" : "maximize",
            ranked[0].Id,
            ranked.Select(c => new ProvidedRankEntry(c.Id, c.Metrics[metricKey])).ToList());
    }

    private sealed class CandidateDto
    {
        public string? Id { get; set; }
        public Dictionary<string, double>? Metrics { get; set; }
    }

    internal static IReadOnlyList<ProvidedCandidate> ParseCandidates(string json)
    {
        var dtos = JsonSerializer.Deserialize<List<CandidateDto>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new ArgumentException("Metrics input did not parse to a candidate array.");

        var result = new List<ProvidedCandidate>();
        for (var i = 0; i < dtos.Count; i++)
        {
            var d = dtos[i];
            if (string.IsNullOrWhiteSpace(d.Id))
                throw new ArgumentException($"Candidate at index {i} is missing 'id'.");
            if (d.Metrics == null || d.Metrics.Count == 0)
                throw new ArgumentException($"Candidate '{d.Id}' is missing 'metrics'.");
            result.Add(new ProvidedCandidate(d.Id, d.Metrics));
        }
        return result;
    }

    private static async Task<int> ExecuteProvidedStateAsync(string metricsFile, string? metric)
    {
        try
        {
            string content;
            if (metricsFile == "-")
            {
                content = await Console.In.ReadToEndAsync();
            }
            else if (File.Exists(metricsFile))
            {
                content = await File.ReadAllTextAsync(metricsFile);
            }
            else
            {
                Console.WriteLine(JsonSerializer.Serialize(new { error = $"Metrics file not found: {metricsFile}" }));
                return 1;
            }

            var result = CompareProvidedMetrics(ParseCandidates(content), metric);
            Console.WriteLine(JsonSerializer.Serialize(result, ProvidedJsonOptions));
            return 0;
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or IOException)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { error = ex.Message }));
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
