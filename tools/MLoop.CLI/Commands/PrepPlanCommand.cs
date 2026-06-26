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
        var jsonOption = new Option<bool>("--json") { Description = "Output the resulting plan as structured JSON instead of a console table" };

        var cmd = new Command("plan", "Edit prep step declarations in mloop.yaml (policy only, no data change)");
        cmd.Options.Add(setOption);
        cmd.Options.Add(removeOption);
        cmd.Options.Add(columnsOption);
        cmd.Options.Add(listOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);

        cmd.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(setOption),
            parseResult.GetValue(removeOption),
            parseResult.GetValue(columnsOption),
            parseResult.GetValue(listOption),
            parseResult.GetValue(nameOption)!,
            parseResult.GetValue(jsonOption)));

        return cmd;
    }

    private static async Task<int> ExecuteAsync(string? set, string? remove, string? columnsCsv, bool list, string modelName, bool json)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var config = await ctx.ConfigLoader.LoadUserConfigAsync();
            if (!config.Models.TryGetValue(modelName, out var model))
            {
                AnsiConsole.MarkupLine($"[red]Model '{modelName}' not found in mloop.yaml.[/]");
                AnsiConsole.MarkupLine($"[grey]Available models: {string.Join(", ", config.Models.Keys)}[/]");
                return 1;
            }

            model.Prep ??= new List<PrepStep>();
            var columns = ParseColumns(columnsCsv);
            PrepPlanApplied? applied = null;
            PrepStep? appliedStep = null;

            if (!string.IsNullOrEmpty(set))
            {
                var (type, method) = PrepPlanEditor.ParseSet(set);
                var step = new PrepStep { Type = type, Method = method, Columns = columns };
                PrepPlanEditor.SetStep(model.Prep, step);
                if (model.Prep.Count == 0) model.Prep = null;
                await ctx.ConfigLoader.SaveUserConfigAsync(config);
                model.Prep ??= new List<PrepStep>();
                applied = new PrepPlanApplied("set", type, method, columns, RemovedCount: null);
                appliedStep = step;
            }
            else if (!string.IsNullOrEmpty(remove))
            {
                var n = PrepPlanEditor.RemoveStep(model.Prep, remove, columns);
                var prepForDisplay = model.Prep; // capture before nulling
                if (model.Prep.Count == 0) model.Prep = null;
                await ctx.ConfigLoader.SaveUserConfigAsync(config);
                model.Prep = prepForDisplay; // restore for display
                applied = new PrepPlanApplied("remove", remove, Method: null, columns, RemovedCount: n);
            }

            var prep = model.Prep ?? new List<PrepStep>();

            if (json)
            {
                var views = PolicyJson.BuildStepViews(prep, model.Task);
                var env = new PrepPlanEnvelope(
                    "prep plan", modelName, model.Task, applied, views, PolicyJson.CollectWarnings(views));
                Console.WriteLine(PolicyJson.Serialize(env));
                return 0;
            }

            if (applied is { Action: "set" })
            {
                AnsiConsole.MarkupLine($"[green]Set prep step:[/] {Markup.Escape(applied.Type!)}{(applied.Method != null ? ":" + Markup.Escape(applied.Method) : "")}");
                if (appliedStep != null) WarnIfLeaky(appliedStep, model.Task);
            }
            else if (applied is { Action: "remove" })
            {
                AnsiConsole.MarkupLine($"[green]Removed {applied.RemovedCount} prep step(s)[/] of type '{Markup.Escape(applied.Type!)}'.");
            }

            // Always print the resulting plan (also the --list path).
            PrintPlan(prep, model.Task);
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
