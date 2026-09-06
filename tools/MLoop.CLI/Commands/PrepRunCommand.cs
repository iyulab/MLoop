using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Preprocessing;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop prep run - Execute YAML-defined preprocessing pipeline independently.
/// Allows previewing and debugging prep steps without running full training.
/// </summary>
public static class PrepRunCommand
{
    public static Command Create()
    {
        var inputOption = new Option<string?>("--input", "-i")
        {
            Description = "Input CSV file path (defaults to datasets/train.csv)"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output CSV file path (defaults to <input>_prep.csv)"
        };

        var modelOption = new Option<string?>("--name", "-n")
        {
            Description = "Model name to use prep steps from (defaults to 'default')"
        };

        var dryRunOption = new Option<bool>("--dry-run")
        {
            Description = "Show pipeline steps without executing",
            DefaultValueFactory = _ => false
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the pipeline result as JSON to stdout instead of the human report. " +
                          "Progress and narration go to stderr."
        };

        var command = new Command("run", "Execute YAML preprocessing pipeline");
        command.Options.Add(inputOption);
        command.Options.Add(outputOption);
        command.Options.Add(modelOption);
        command.Options.Add(dryRunOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var input = parseResult.GetValue(inputOption);
            var output = parseResult.GetValue(outputOption);
            var modelName = parseResult.GetValue(modelOption);
            var dryRun = parseResult.GetValue(dryRunOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(input, output, modelName, dryRun, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? inputPath,
        string? outputPath,
        string? modelName,
        bool dryRun,
        bool jsonOutput = false)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        var resolvedModelName = modelName ?? ConfigDefaults.DefaultModelName;

        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            // Load config
            var config = await ctx.ConfigLoader.LoadUserConfigAsync();

            // Resolve model name
            if (!config.Models.TryGetValue(resolvedModelName, out var modelDef))
            {
                AnsiConsole.MarkupLine($"[red]Model '{resolvedModelName}' not found in mloop.yaml.[/]");
                AnsiConsole.MarkupLine($"[grey]Available models: {string.Join(", ", config.Models.Keys)}[/]");
                EmitJson(resolvedModelName, dryRun, null, null, null,
                    [$"Model '{resolvedModelName}' not found in mloop.yaml"], null, null, jsonOutput);
                return 1;
            }

            // Check prep steps
            if (modelDef.Prep == null || modelDef.Prep.Count == 0)
            {
                AnsiConsole.MarkupLine($"[yellow]No prep steps defined for model '{resolvedModelName}'.[/]");
                AnsiConsole.MarkupLine("[grey]Add a 'prep' section to your mloop.yaml model definition.[/]");
                AnsiConsole.MarkupLine("[grey]Example:[/]");
                AnsiConsole.MarkupLine("[grey]  prep:[/]");
                AnsiConsole.MarkupLine("[grey]    - type: fill-missing[/]");
                // Literal brackets in Spectre markup must be doubled ("[[" / "]]") or they're
                // parsed as a style tag (same class of defect as the table-cell fix above).
                AnsiConsole.MarkupLine("[grey]      columns: [[pH, Temp]][/]");
                AnsiConsole.MarkupLine("[grey]      method: mean[/]");
                EmitJson(resolvedModelName, dryRun, null, null, [], null, null, null, jsonOutput);
                return 0;
            }

            // Validate prep steps
            var validationErrors = ConfigValidator.ValidatePrepSteps(modelDef.Prep);
            if (validationErrors.Count > 0)
            {
                AnsiConsole.MarkupLine("[red]Prep step validation failed:[/]");
                foreach (var error in validationErrors)
                {
                    // Same display-boundary rule as the validate report: a validation message carries
                    // user data (step index, column names) and must not be parsed as markup.
                    AnsiConsole.MarkupLine($"  [red]✗[/] {Markup.Escape(error)}");
                }
                EmitJson(resolvedModelName, dryRun, null, null, modelDef.Prep, validationErrors, null, null, jsonOutput);
                return 1;
            }

            // Resolve input path
            var resolvedInput = inputPath;
            if (string.IsNullOrEmpty(resolvedInput))
            {
                // Try data.train from config
                if (!string.IsNullOrEmpty(config.Data?.Train))
                {
                    resolvedInput = ctx.FileSystem.CombinePath(ctx.ProjectRoot, config.Data.Train);
                }
                else
                {
                    resolvedInput = ctx.FileSystem.CombinePath(ctx.ProjectRoot, "datasets", "train.csv");
                }
            }

            if (!File.Exists(resolvedInput))
            {
                AnsiConsole.MarkupLine($"[red]Input file not found: {resolvedInput}[/]");
                AnsiConsole.MarkupLine("[grey]Use --input to specify the CSV file path.[/]");
                EmitJson(resolvedModelName, dryRun, resolvedInput, null, modelDef.Prep,
                    [$"Input file not found: {resolvedInput}"], null, null, jsonOutput);
                return 1;
            }

            // Display pipeline info
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule($"[bold]Prep Pipeline: {resolvedModelName}[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var stepsTable = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            stepsTable.AddColumn(new TableColumn("[bold]#[/]").Centered());
            stepsTable.AddColumn(new TableColumn("[bold]Type[/]"));
            stepsTable.AddColumn(new TableColumn("[bold]Details[/]"));

            for (int i = 0; i < modelDef.Prep.Count; i++)
            {
                var step = modelDef.Prep[i];
                var details = GetStepDetails(step);
                stepsTable.AddRow($"{i + 1}", $"[cyan]{step.Type}[/]", details);
            }

            AnsiConsole.Write(stepsTable);
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine($"[grey]Input:  {resolvedInput}[/]");

            if (dryRun)
            {
                AnsiConsole.MarkupLine("[yellow]Dry run mode — no changes made.[/]");
                AnsiConsole.WriteLine();
                EmitJson(resolvedModelName, dryRun, resolvedInput, null, modelDef.Prep, null, null, null, jsonOutput);
                return 0;
            }

            // Resolve output path
            var resolvedOutput = outputPath;
            if (string.IsNullOrEmpty(resolvedOutput))
            {
                var dir = Path.GetDirectoryName(resolvedInput) ?? ".";
                var name = Path.GetFileNameWithoutExtension(resolvedInput);
                var ext = Path.GetExtension(resolvedInput);
                resolvedOutput = Path.Combine(dir, $"{name}_prep{ext}");
            }

            AnsiConsole.MarkupLine($"[grey]Output: {resolvedOutput}[/]");
            AnsiConsole.WriteLine();

            // Execute pipeline
            var logger = new PrepRunLogger();
            var executor = new DataPipelineExecutor(logger);

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync("Running preprocessing pipeline...", async ctx =>
                {
                    await executor.ExecuteAsync(resolvedInput, modelDef.Prep, resolvedOutput);
                });

            // Show result summary
            var inputLines = File.ReadLines(resolvedInput).Count();
            var outputLines = File.ReadLines(resolvedOutput).Count();

            AnsiConsole.MarkupLine("[green]Pipeline executed successfully![/]");
            AnsiConsole.MarkupLine($"[grey]  Rows: {inputLines - 1} → {outputLines - 1}[/]");
            AnsiConsole.MarkupLine($"[grey]  Output: {resolvedOutput}[/]");
            AnsiConsole.WriteLine();

            EmitJson(resolvedModelName, dryRun, resolvedInput, resolvedOutput, modelDef.Prep, null,
                inputLines - 1, outputLines - 1, jsonOutput);
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "preprocessing");
            return 1;
        }
    }

    /// <summary>
    /// The <c>--json</c> counterpart to the human report above — same data (the resolved paths and
    /// the model's own <see cref="PrepStep"/> list, already fully structured — no separate view model
    /// needed), machine shape. No-op when <paramref name="jsonOutput"/> is false so every exit point
    /// can call it unconditionally rather than repeating the branch (the <c>validate --json</c>
    /// pattern). <paramref name="errors"/> covers every failure exit (model/steps/input not found,
    /// validation failures) with one shape rather than a separate one per case.
    /// </summary>
    private static void EmitJson(
        string modelName,
        bool dryRun,
        string? input,
        string? output,
        List<PrepStep>? steps,
        List<string>? errors,
        int? rowsBefore,
        int? rowsAfter,
        bool jsonOutput)
    {
        if (!jsonOutput)
            return;

        var payload = new
        {
            Model = modelName,
            DryRun = dryRun,
            Input = input,
            Output = output,
            Steps = steps,
            Errors = errors,
            RowsBefore = rowsBefore,
            RowsAfter = rowsAfter
        };
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(payload, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        }));
    }

    internal static string GetStepDetails(PrepStep step)
    {
        var parts = new List<string>();

        if (step.Columns is { Count: > 0 })
            parts.Add($"columns: [{string.Join(", ", step.Columns)}]");
        if (!string.IsNullOrEmpty(step.Column))
            parts.Add($"column: {step.Column}");
        if (!string.IsNullOrEmpty(step.Method))
            parts.Add($"method: {step.Method}");
        if (!string.IsNullOrEmpty(step.Format))
            parts.Add($"format: {step.Format}");
        if (step.Mapping is { Count: > 0 })
            parts.Add($"mapping: {step.Mapping.Count} entries");
        if (!string.IsNullOrEmpty(step.Operator))
            parts.Add($"op: {step.Operator} {step.Value}");
        if (step.WindowSize > 0)
            parts.Add($"window: {step.WindowSize}");
        if (!string.IsNullOrEmpty(step.Window))
            parts.Add($"window: {step.Window}");
        if (!string.IsNullOrEmpty(step.TimeColumn))
            parts.Add($"time: {step.TimeColumn}");
        if (step.RemoveOriginal)
            parts.Add("remove_original");
        if (step.Count > 0)
            parts.Add($"count: {step.Count}");
        if (step.Seed != 42 && step.Count > 0) // Only show non-default seed for sample steps
            parts.Add($"seed: {step.Seed}");

        // The joined parts are plain user data (column names etc.) and may contain raw "[...]"
        // (e.g. "columns: [pH, Temp]") that Spectre would otherwise parse as a style tag when
        // rendered — escape only this branch. The empty-case sentinel below is deliberately
        // literal Spectre markup, not user data, so it stays un-escaped.
        return parts.Count > 0 ? Markup.Escape(string.Join(", ", parts)) : "[grey]-[/]";
    }

    private class PrepRunLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) => AnsiConsole.MarkupLine($"[grey]  {Markup.Escape(message)}[/]");
        public void Warning(string message) => AnsiConsole.MarkupLine($"[yellow]  {Markup.Escape(message)}[/]");
        public void Error(string message) => AnsiConsole.MarkupLine($"[red]  {Markup.Escape(message)}[/]");
        public void Error(string message, Exception ex) => AnsiConsole.MarkupLine($"[red]  {Markup.Escape(message)}: {Markup.Escape(ex.Message)}[/]");
    }
}
