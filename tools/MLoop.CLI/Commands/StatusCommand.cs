using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop status - Show project status at a glance
/// </summary>
public static class StatusCommand
{
    public static Command Create()
    {
        var verboseOption = new Option<bool>("--verbose", "-v")
        {
            Description = "Show detailed status information",
            DefaultValueFactory = _ => false
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the project status as JSON to stdout instead of the human report. " +
                          "Progress and narration go to stderr."
        };

        var command = new Command("status", "Show project status at a glance");
        command.Options.Add(verboseOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var verbose = parseResult.GetValue(verboseOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(verbose, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(bool verbose, bool jsonOutput = false)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            AnsiConsole.WriteLine();

            // Project Header
            var projectName = Path.GetFileName(ctx.ProjectRoot);
            AnsiConsole.Write(new Rule($"[bold]MLoop Project: {projectName}[/]").LeftJustified());
            AnsiConsole.WriteLine();

            // Project Info Panel
            var projectPanel = new Panel(
                new Markup($"[grey]Path:[/] {ctx.ProjectRoot}"))
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Grey)
                .Header("[blue]Project[/]");

            AnsiConsole.Write(projectPanel);
            AnsiConsole.WriteLine();

            // Config Summary
            await ShowConfigSummaryAsync(ctx);

            // Get all data
            var models = await ctx.ModelRegistry.ListAsync(null, CancellationToken.None);
            var productionDict = models.ToDictionary(m => m.ModelName, m => m.ExperimentId);

            var allExperiments = await ctx.ExperimentStore.ListAsync(null, CancellationToken.None);
            var experimentsList = allExperiments.ToList();

            // Models Overview Table
            var modelNames = experimentsList
                .Select(e => e.ModelName ?? ConfigDefaults.DefaultModelName)
                .Distinct()
                .OrderBy(n => n)
                .ToList();

            if (modelNames.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No models found.[/]");
                AnsiConsole.MarkupLine("[grey]Get started with: [blue]mloop train <data.csv> <label-column>[/][/]");
                AnsiConsole.WriteLine();
                EmitJson(ctx.ProjectRoot, [], null, 0, 0, 0, 0, jsonOutput);
                return 0;
            }

            // Create models status table
            var modelsTable = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey)
                .Title("[bold]Models Overview[/]");

            modelsTable.AddColumn(new TableColumn("[bold]Model[/]"));
            modelsTable.AddColumn(new TableColumn("[bold]Experiments[/]").Centered());
            modelsTable.AddColumn(new TableColumn("[bold]Completed[/]").Centered());
            modelsTable.AddColumn(new TableColumn("[bold]Production[/]").Centered());
            modelsTable.AddColumn(new TableColumn("[bold]Best Metric[/]").RightAligned());
            modelsTable.AddColumn(new TableColumn("[bold]Last Prediction[/]").RightAligned());

            var predictionsDir = ctx.FileSystem.CombinePath(ctx.ProjectRoot, "predictions");

            // Collect model rows first so the same data feeds both the human table and --json —
            // one computation path, no separate JSON re-derivation.
            var modelRows = new List<ModelStatusRow>();
            foreach (var modelName in modelNames)
            {
                var modelExperiments = experimentsList
                    .Where(e => (e.ModelName ?? ConfigDefaults.DefaultModelName) == modelName)
                    .ToList();

                var total = modelExperiments.Count;
                var completed = modelExperiments.Count(e =>
                    e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));

                var hasProduction = productionDict.TryGetValue(modelName, out var prodExpId);

                // F-28: honor metric direction so lower-is-better metrics (clustering/forecasting/
                // recommendation) don't report the worst experiment as best.
                var bestMetric = ExperimentRanking.SelectBest(modelExperiments)?.BestMetric;

                var lastPredictionDisplay = GetLatestPrediction(predictionsDir, modelName);
                var lastPredictionAtUtc = GetLatestPredictionTimestampUtc(predictionsDir, modelName);

                modelRows.Add(new ModelStatusRow(
                    modelName, total, completed, hasProduction,
                    hasProduction ? prodExpId : null, bestMetric,
                    lastPredictionDisplay, lastPredictionAtUtc));
            }

            foreach (var row in modelRows)
            {
                var productionStatus = row.HasProduction
                    ? $"[green]✓[/] {row.ProductionExperimentId}"
                    : "[grey]-[/]";

                var bestMetricStr = row.BestMetric.HasValue
                    ? $"[yellow]{row.BestMetric.Value:F4}[/]"
                    : "[grey]-[/]";

                modelsTable.AddRow(
                    $"[cyan]{row.ModelName}[/]",
                    row.TotalExperiments.ToString(),
                    row.CompletedExperiments > 0 ? $"[green]{row.CompletedExperiments}[/]" : "[grey]0[/]",
                    productionStatus,
                    bestMetricStr,
                    row.LastPredictionDisplay);
            }

            AnsiConsole.Write(modelsTable);
            AnsiConsole.WriteLine();

            // Data Files Status
            List<DataFileRow>? dataFileRows = null;
            if (verbose)
            {
                var datasetsDir = ctx.FileSystem.CombinePath(ctx.ProjectRoot, "datasets");

                dataFileRows =
                [
                    CheckDataFile(ctx, "Train", datasetsDir, "train.csv"),
                    CheckDataFile(ctx, "Test", datasetsDir, "test.csv"),
                    CheckDataFile(ctx, "Predict", datasetsDir, "predict.csv"),
                    GetPredictionsDataFile(ctx, predictionsDir)
                ];

                var dataTable = new Table()
                    .Border(TableBorder.Rounded)
                    .BorderColor(Color.Grey)
                    .Title("[bold]Data Files[/]");

                dataTable.AddColumn(new TableColumn("[bold]Type[/]"));
                dataTable.AddColumn(new TableColumn("[bold]Path[/]"));
                dataTable.AddColumn(new TableColumn("[bold]Status[/]").Centered());

                foreach (var row in dataFileRows)
                {
                    dataTable.AddRow(row.Type, row.Path, row.Exists ? "[green]✓[/]" : "[grey]-[/]");
                }

                AnsiConsole.Write(dataTable);
                AnsiConsole.WriteLine();
            }

            // Summary Statistics
            var totalExperiments = experimentsList.Count;
            var completedExperiments = experimentsList.Count(e =>
                e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));
            var failedExperiments = experimentsList.Count(e =>
                e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase));
            var productionCount = productionDict.Count;

            AnsiConsole.Write(new Rule("[bold]Summary[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var grid = new Grid();
            grid.AddColumn();
            grid.AddColumn();
            grid.AddColumn();
            grid.AddColumn();

            grid.AddRow(
                $"[grey]Models:[/] [cyan]{modelNames.Count}[/]",
                $"[grey]Experiments:[/] {totalExperiments}",
                $"[grey]Completed:[/] [green]{completedExperiments}[/]",
                $"[grey]Production:[/] [green]{productionCount}[/]");

            AnsiConsole.Write(grid);
            AnsiConsole.WriteLine();

            // Quick Actions
            if (productionCount < modelNames.Count)
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[grey]💡 Tip: Some models don't have production versions.[/]");
                AnsiConsole.MarkupLine("[grey]   Use [blue]mloop promote <exp-id> --name <model>[/] to promote.[/]");
            }

            if (totalExperiments == 0)
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[grey]💡 Get started: [blue]mloop train datasets/train.csv <label-column>[/][/]");
            }

            AnsiConsole.WriteLine();
            EmitJson(ctx.ProjectRoot, modelRows, dataFileRows, totalExperiments, completedExperiments,
                failedExperiments, productionCount, jsonOutput);
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "status");
            return 1;
        }
    }

    /// <summary>
    /// The <c>--json</c> counterpart to the human tables above — same data (<see cref="ModelStatusRow"/>/
    /// <see cref="DataFileRow"/> collected once in <see cref="ExecuteAsync"/>), machine shape. No-op when
    /// <paramref name="jsonOutput"/> is false so both exit points can call it unconditionally.
    /// </summary>
    private static void EmitJson(
        string projectRoot,
        List<ModelStatusRow> modelRows,
        List<DataFileRow>? dataFileRows,
        int totalExperiments,
        int completedExperiments,
        int failedExperiments,
        int productionCount,
        bool jsonOutput)
    {
        if (!jsonOutput)
            return;

        var payload = new
        {
            ProjectRoot = projectRoot,
            Models = modelRows.Select(r => new
            {
                r.ModelName,
                r.TotalExperiments,
                r.CompletedExperiments,
                r.HasProduction,
                r.ProductionExperimentId,
                r.BestMetric,
                LastPredictionAt = r.LastPredictionAtUtc
            }),
            DataFiles = dataFileRows?.Select(d => new { d.Type, d.Path, d.Exists }),
            Summary = new
            {
                ModelsCount = modelRows.Count,
                TotalExperiments = totalExperiments,
                CompletedExperiments = completedExperiments,
                FailedExperiments = failedExperiments,
                ProductionCount = productionCount
            }
        };
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(payload, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
        }));
    }

    internal record ModelStatusRow(
        string ModelName,
        int TotalExperiments,
        int CompletedExperiments,
        bool HasProduction,
        string? ProductionExperimentId,
        double? BestMetric,
        string LastPredictionDisplay,
        DateTime? LastPredictionAtUtc);

    internal record DataFileRow(string Type, string Path, bool Exists);

    private static async Task ShowConfigSummaryAsync(CommandContext ctx)
    {
        try
        {
            var config = await ctx.ConfigLoader.LoadUserConfigAsync();

            if (config.Models.Count == 0)
                return;

            var configTable = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey)
                .Title("[bold]Configuration[/] [grey](mloop.yaml)[/]");

            configTable.AddColumn(new TableColumn("[bold]Model[/]"));
            configTable.AddColumn(new TableColumn("[bold]Task[/]"));
            configTable.AddColumn(new TableColumn("[bold]Label[/]"));
            configTable.AddColumn(new TableColumn("[bold]Time Limit[/]").RightAligned());
            configTable.AddColumn(new TableColumn("[bold]Metric[/]").Centered());

            foreach (var (name, model) in config.Models)
            {
                var timeLimit = model.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds;
                var metric = model.Training?.Metric ?? ConfigDefaults.DefaultMetric;

                configTable.AddRow(
                    $"[cyan]{name}[/]",
                    model.Task,
                    model.Label,
                    $"{timeLimit}s",
                    metric);
            }

            AnsiConsole.Write(configTable);
            AnsiConsole.WriteLine();
        }
        catch (Exception)
        {
            // Config not available — skip silently
        }
    }

    internal static string GetLatestPrediction(string predictionsDir, string modelName)
    {
        var lastWrite = GetLatestPredictionTimestampUtc(predictionsDir, modelName);
        if (lastWrite == null)
            return "[grey]-[/]";

        var age = DateTime.UtcNow - lastWrite.Value;

        if (age.TotalMinutes < 60)
            return $"[green]{(int)age.TotalMinutes}m ago[/]";
        if (age.TotalHours < 24)
            return $"[green]{(int)age.TotalHours}h ago[/]";
        if (age.TotalDays < 30)
            return $"[yellow]{(int)age.TotalDays}d ago[/]";

        return $"[grey]{lastWrite.Value:yyyy-MM-dd}[/]";
    }

    /// <summary>Raw fact behind <see cref="GetLatestPrediction"/>'s markup — the <c>--json</c> shape
    /// wants the timestamp itself, not a "15m ago" display string.</summary>
    internal static DateTime? GetLatestPredictionTimestampUtc(string predictionsDir, string modelName)
    {
        if (!Directory.Exists(predictionsDir))
            return null;

        var pattern = $"{modelName}-predictions-*.csv";
        var files = Directory.GetFiles(predictionsDir, pattern);

        if (files.Length == 0)
            return null;

        var latestFile = files
            .OrderByDescending(f => File.GetLastWriteTimeUtc(f))
            .First();

        return File.GetLastWriteTimeUtc(latestFile);
    }

    private static DataFileRow CheckDataFile(
        CommandContext ctx,
        string type,
        string directory,
        string filename)
    {
        var filePath = ctx.FileSystem.CombinePath(directory, filename);
        var relativePath = $"datasets/{filename}";

        return new DataFileRow(type, relativePath, ctx.FileSystem.FileExists(filePath));
    }

    private static DataFileRow GetPredictionsDataFile(CommandContext ctx, string predictionsDir)
    {
        if (!ctx.FileSystem.DirectoryExists(predictionsDir))
            return new DataFileRow("Predictions", "predictions/", false);

        var predFiles = Directory.GetFiles(predictionsDir, "*.csv");
        if (predFiles.Length == 0)
            return new DataFileRow("Predictions", "predictions/", false);

        var latestPred = predFiles
            .OrderByDescending(f => File.GetLastWriteTime(f))
            .First();
        var relativePath = Path.GetRelativePath(ctx.ProjectRoot, latestPred);
        return new DataFileRow("Predictions", relativePath, true);
    }
}
