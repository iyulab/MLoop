using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
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

        var command = new Command("status", "Show project status at a glance");
        command.Options.Add(verboseOption);

        command.SetAction((parseResult) =>
        {
            var verbose = parseResult.GetValue(verboseOption);
            return ExecuteAsync(verbose);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(bool verbose)
    {
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

            foreach (var modelName in modelNames)
            {
                var modelExperiments = experimentsList
                    .Where(e => (e.ModelName ?? ConfigDefaults.DefaultModelName) == modelName)
                    .ToList();

                var total = modelExperiments.Count;
                var completed = modelExperiments.Count(e =>
                    e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));

                var hasProduction = productionDict.TryGetValue(modelName, out var prodExpId);
                var productionStatus = hasProduction
                    ? $"[green]✓[/] {prodExpId}"
                    : "[grey]-[/]";

                var bestMetric = modelExperiments
                    .Where(e => e.BestMetric.HasValue)
                    .OrderByDescending(e => e.BestMetric!.Value)
                    .FirstOrDefault()?.BestMetric;

                var bestMetricStr = bestMetric.HasValue
                    ? $"[yellow]{bestMetric.Value:F4}[/]"
                    : "[grey]-[/]";

                var lastPrediction = GetLatestPrediction(predictionsDir, modelName);

                modelsTable.AddRow(
                    $"[cyan]{modelName}[/]",
                    total.ToString(),
                    completed > 0 ? $"[green]{completed}[/]" : "[grey]0[/]",
                    productionStatus,
                    bestMetricStr,
                    lastPrediction);
            }

            AnsiConsole.Write(modelsTable);
            AnsiConsole.WriteLine();

            // Data Files Status
            if (verbose)
            {
                var dataTable = new Table()
                    .Border(TableBorder.Rounded)
                    .BorderColor(Color.Grey)
                    .Title("[bold]Data Files[/]");

                dataTable.AddColumn(new TableColumn("[bold]Type[/]"));
                dataTable.AddColumn(new TableColumn("[bold]Path[/]"));
                dataTable.AddColumn(new TableColumn("[bold]Status[/]").Centered());

                var datasetsDir = ctx.FileSystem.CombinePath(ctx.ProjectRoot, "datasets");

                // Check for common data files
                CheckDataFile(dataTable, ctx, "Train", datasetsDir, "train.csv");
                CheckDataFile(dataTable, ctx, "Test", datasetsDir, "test.csv");
                CheckDataFile(dataTable, ctx, "Predict", datasetsDir, "predict.csv");

                // Check predictions directory
                if (ctx.FileSystem.DirectoryExists(predictionsDir))
                {
                    var predFiles = Directory.GetFiles(predictionsDir, "*.csv");
                    if (predFiles.Length > 0)
                    {
                        var latestPred = predFiles
                            .OrderByDescending(f => File.GetLastWriteTime(f))
                            .First();
                        var relativePath = Path.GetRelativePath(ctx.ProjectRoot, latestPred);
                        dataTable.AddRow("Predictions", relativePath, "[green]✓[/]");
                    }
                    else
                    {
                        dataTable.AddRow("Predictions", "predictions/", "[grey]-[/]");
                    }
                }
                else
                {
                    dataTable.AddRow("Predictions", "predictions/", "[grey]-[/]");
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
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "status");
            return 1;
        }
    }

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
        catch
        {
            // Config not available — skip silently
        }
    }

    private static string GetLatestPrediction(string predictionsDir, string modelName)
    {
        if (!Directory.Exists(predictionsDir))
            return "[grey]-[/]";

        var pattern = $"{modelName}-predictions-*.csv";
        var files = Directory.GetFiles(predictionsDir, pattern);

        if (files.Length == 0)
            return "[grey]-[/]";

        var latestFile = files
            .OrderByDescending(f => File.GetLastWriteTimeUtc(f))
            .First();

        var lastWrite = File.GetLastWriteTimeUtc(latestFile);
        var age = DateTime.UtcNow - lastWrite;

        if (age.TotalMinutes < 60)
            return $"[green]{(int)age.TotalMinutes}m ago[/]";
        if (age.TotalHours < 24)
            return $"[green]{(int)age.TotalHours}h ago[/]";
        if (age.TotalDays < 30)
            return $"[yellow]{(int)age.TotalDays}d ago[/]";

        return $"[grey]{lastWrite:yyyy-MM-dd}[/]";
    }

    private static void CheckDataFile(
        Table table,
        CommandContext ctx,
        string type,
        string directory,
        string filename)
    {
        var filePath = ctx.FileSystem.CombinePath(directory, filename);
        var relativePath = $"datasets/{filename}";

        if (ctx.FileSystem.FileExists(filePath))
        {
            table.AddRow(type, relativePath, "[green]✓[/]");
        }
        else
        {
            table.AddRow(type, relativePath, "[grey]-[/]");
        }
    }
}
