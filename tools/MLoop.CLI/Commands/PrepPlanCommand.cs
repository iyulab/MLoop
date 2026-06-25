using System.CommandLine;
using MLoop.CLI.Commands.Policy;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.AutoML;
using MLoop.Core.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop prep plan — declaratively edit a model's prep steps in mloop.yaml.
/// Records policy only; statistical fit happens fold-internally at train time (P0).
/// Never touches the data file.
/// </summary>
public static class PrepPlanCommand
{
    public static Command Create()
    {
        var setOption = new Option<string?>("--set") { Description = "Add/replace a prep step: type[:method] (e.g. normalize:z-score)" };
        var removeOption = new Option<string?>("--remove") { Description = "Remove prep step(s) of this type" };
        var columnsOption = new Option<string?>("--columns", "-c") { Description = "Comma-separated target columns" };
        var listOption = new Option<bool>("--list") { Description = "List current prep steps" };
        var nameOption = new Option<string>("--name", "-n") { Description = "Model name", DefaultValueFactory = _ => ConfigDefaults.DefaultModelName };

        var cmd = new Command("plan", "Edit prep step declarations in mloop.yaml (policy only, no data change)");
        cmd.Options.Add(setOption);
        cmd.Options.Add(removeOption);
        cmd.Options.Add(columnsOption);
        cmd.Options.Add(listOption);
        cmd.Options.Add(nameOption);

        cmd.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(setOption),
            parseResult.GetValue(removeOption),
            parseResult.GetValue(columnsOption),
            parseResult.GetValue(listOption),
            parseResult.GetValue(nameOption)!));

        return cmd;
    }

    private static async Task<int> ExecuteAsync(string? set, string? remove, string? columnsCsv, bool list, string modelName)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var config = await ctx.ConfigLoader.LoadUserConfigAsync();
            if (!config.Models.TryGetValue(modelName, out var model))
            {
                AnsiConsole.MarkupLine($"[red]Model '{modelName}' not found in mloop.yaml.[/]");
                return 1;
            }

            model.Prep ??= new List<PrepStep>();
            var columns = ParseColumns(columnsCsv);

            if (!string.IsNullOrEmpty(set))
            {
                var (type, method) = PrepPlanEditor.ParseSet(set);
                var step = new PrepStep { Type = type, Method = method, Columns = columns };
                PrepPlanEditor.SetStep(model.Prep, step);
                await ctx.ConfigLoader.SaveUserConfigAsync(config);
                AnsiConsole.MarkupLine($"[green]Set prep step:[/] {Markup.Escape(type)}{(method != null ? ":" + Markup.Escape(method) : "")}");
                WarnIfLeaky(step, model.Task);
            }
            else if (!string.IsNullOrEmpty(remove))
            {
                var n = PrepPlanEditor.RemoveStep(model.Prep, remove, columns);
                await ctx.ConfigLoader.SaveUserConfigAsync(config);
                AnsiConsole.MarkupLine($"[green]Removed {n} prep step(s)[/] of type '{Markup.Escape(remove)}'.");
            }

            // Always print the resulting plan (also the --list path).
            PrintPlan(model.Prep, model.Task);
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "prep plan");
            return 1;
        }
    }

    private static List<string>? ParseColumns(string? csv) =>
        string.IsNullOrWhiteSpace(csv)
            ? null
            : csv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();

    private static void WarnIfLeaky(PrepStep step, string task)
    {
        switch (PrepStepClassifier.Classify(step))
        {
            case PrepCategory.UnsupportedLeakageWarn:
                AnsiConsole.MarkupLine($"[yellow]⚠ {Markup.Escape(PrepStepClassifier.LeakageWarning(step))}[/]");
                break;
            case PrepCategory.PreFeaturizer when !AutoMLRunner.SupportsPreFeaturizer(task):
                AnsiConsole.MarkupLine($"[yellow]⚠ {Markup.Escape(PrepStepClassifier.UnsupportedTaskLeakageWarning(step))}[/]");
                break;
        }
    }

    private static void PrintPlan(List<PrepStep> prep, string task)
    {
        if (prep.Count == 0) { AnsiConsole.MarkupLine("[grey]No prep steps defined.[/]"); return; }
        var supports = AutoMLRunner.SupportsPreFeaturizer(task);
        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("#");
        table.AddColumn("Type");
        table.AddColumn("Details");
        table.AddColumn("Leakage");
        for (int i = 0; i < prep.Count; i++)
        {
            var s = prep[i];
            table.AddRow($"{i + 1}", Markup.Escape(s.Type),
                Markup.Escape(PrepRunCommand.GetStepDetails(s)), CategoryLabel(s, supports));
        }
        AnsiConsole.Write(table);
    }

    private static string CategoryLabel(PrepStep step, bool supportsPreFeaturizer) =>
        PrepStepClassifier.Classify(step) switch
        {
            PrepCategory.PreFeaturizer => supportsPreFeaturizer ? "[green]✓ fold-safe[/]" : "[yellow]⚠ leak (task)[/]",
            PrepCategory.UnsupportedLeakageWarn => "[yellow]⚠ leak[/]",
            _ => "[grey]ℹ csv-stage[/]"
        };
}
