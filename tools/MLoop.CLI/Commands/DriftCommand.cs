using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Monitoring;
using MLoop.DataStore.Services;
using MLoop.Ops.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop drift — how far the production model's recent inputs have moved from the data it was trained
/// on, column by column (population stability index), overall and per value of a slice column.
/// </summary>
public static class DriftCommand
{
    public static Command Create()
    {
        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };
        var rowsOption = new Option<int>("--rows")
        {
            Description = "How many of the most recent logged predictions to compare",
            DefaultValueFactory = _ => DriftEvaluator.DefaultWindowRows
        };
        var sliceOption = new Option<string?>("--slice-by")
        {
            Description = $"Also report drift per value of this input column (values with at least {DriftEvaluator.MinimumRows} rows)"
        };
        var jsonOption = new Option<bool>("--json") { Description = "Output in JSON format" };

        var command = new Command("drift", "Compare recent inputs with the training data (PSI per column)");
        command.Options.Add(nameOption);
        command.Options.Add(rowsOption);
        command.Options.Add(sliceOption);
        command.Options.Add(jsonOption);
        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(nameOption), parseResult.GetValue(rowsOption),
            parseResult.GetValue(sliceOption), parseResult.GetValue(jsonOption)));
        return command;
    }

    private static async Task<int> ExecuteAsync(string? modelName, int rows, string? sliceBy, bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        var fileSystem = new FileSystemManager();
        string projectRoot;
        try
        {
            projectRoot = new ProjectDiscovery(fileSystem).FindRoot();
        }
        catch (InvalidOperationException)
        {
            ErrorConsole.Error(ProjectDiscovery.NotInsideProjectCause, ProjectDiscovery.NotInsideProjectGuidance);
            if (jsonOutput)
                JsonError.EmitMarkup(ProjectDiscovery.NotInsideProjectCause, ProjectDiscovery.NotInsideProjectGuidance);
            return 1;
        }
        if (rows < DriftEvaluator.MinimumRows)
        {
            var cause = $"--rows must be at least {DriftEvaluator.MinimumRows}.";
            ErrorConsole.Error(cause);
            if (jsonOutput)
                JsonError.Emit(cause);
            return 1;
        }

        var name = CommandContext.ResolveModelName(modelName);
        var check = await DriftEvaluator.CheckAsync(projectRoot, new FilePredictionLogger(projectRoot), name, rows, sliceBy);

        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                modelName = name,
                status = check.Status.ToString(),
                experimentId = check.ExperimentId,
                message = check.Message,
                drift = check.Report is { } r ? DriftEvaluator.Signal(r) : null,
                sliceBy,
                slices = check.Slices?.Select(s => new { value = s.Value, drift = DriftEvaluator.Signal(s.Report) }),
            }, new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }

        if (check.Report is not { } report)
        {
            AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(check.Message)}[/]");
            return 0;
        }

        var table = new Table().Border(TableBorder.Rounded)
            .Title($"[bold]Input drift — {Markup.Escape(name)} ({Markup.Escape(check.ExperimentId!)}, {report.Rows:N0} predictions)[/]")
            .AddColumn("[grey]Column[/]")
            .AddColumn(new TableColumn("[grey]PSI[/]").RightAligned())
            .AddColumn(new TableColumn("[grey]Level[/]").Centered())
            .AddColumn("[grey]Most moved[/]");
        foreach (var f in report.Features)
            table.AddRow(Markup.Escape(f.Column), $"{f.Psi:0.000}", LevelMarkup(f.Level),
                $"{Markup.Escape(f.MostChangedBin)}: {f.ReferenceShare:P0} → {f.CurrentShare:P0}");
        AnsiConsole.Write(table);

        if (check.Slices is { Count: > 0 } slices)
        {
            var bySlice = new Table().Border(TableBorder.Rounded)
                .Title($"[bold]By {Markup.Escape(sliceBy!)}[/]")
                .AddColumn("[grey]Value[/]")
                .AddColumn(new TableColumn("[grey]Rows[/]").RightAligned())
                .AddColumn(new TableColumn("[grey]Max PSI[/]").RightAligned())
                .AddColumn(new TableColumn("[grey]Level[/]").Centered())
                .AddColumn("[grey]Most moved column[/]");
            foreach (var s in slices)
                bySlice.AddRow(Markup.Escape(s.Value), $"{s.Report.Rows:N0}", $"{s.Report.MaxPsi:0.000}",
                    LevelMarkup(s.Report.Level), Markup.Escape(s.Report.Features.FirstOrDefault()?.Column ?? "-"));
            AnsiConsole.Write(bySlice);
        }
        else if (sliceBy is not null)
        {
            AnsiConsole.MarkupLine($"[grey]No value of '{Markup.Escape(sliceBy)}' has {DriftEvaluator.MinimumRows}+ predictions in the window.[/]");
        }

        AnsiConsole.MarkupLine($"[grey]PSI under {DriftMeasure.ModerateThreshold} is stable, {DriftMeasure.ModerateThreshold}–{DriftMeasure.SignificantThreshold} moderate, {DriftMeasure.SignificantThreshold}+ significant.[/]");
        return 0;
    }

    private static string LevelMarkup(DriftLevel level) => level switch
    {
        DriftLevel.Significant => "[red]significant[/]",
        DriftLevel.Moderate => "[yellow]moderate[/]",
        _ => "[green]stable[/]",
    };
}
