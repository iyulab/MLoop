using System.CommandLine;
using MLoop.CLI.Commands.Policy;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop features select — declare feature include/exclude in mloop.yaml via
/// ColumnOverride (Type:"ignore"). Records policy only; never changes data.
/// </summary>
public sealed class FeaturesCommand : Command
{
    public FeaturesCommand() : base("features", "Feature selection policy (mloop.yaml only)")
    {
        this.Add(CreateSelectCommand());
    }

    private static Command CreateSelectCommand()
    {
        var dropOption = new Option<string?>("--drop") { Description = "Comma-separated columns to exclude from features" };
        var keepOption = new Option<string?>("--keep") { Description = "Comma-separated columns to keep (all others excluded)" };
        var resetOption = new Option<bool>("--reset") { Description = "Remove all 'ignore' overrides" };
        var nameOption = new Option<string>("--name", "-n") { Description = "Model name", DefaultValueFactory = _ => ConfigDefaults.DefaultModelName };

        var cmd = new Command("select", "Include/exclude features in mloop.yaml (policy only, no data change)");
        cmd.Options.Add(dropOption);
        cmd.Options.Add(keepOption);
        cmd.Options.Add(resetOption);
        cmd.Options.Add(nameOption);

        cmd.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(dropOption),
            parseResult.GetValue(keepOption),
            parseResult.GetValue(resetOption),
            parseResult.GetValue(nameOption)!));

        return cmd;
    }

    private static async Task<int> ExecuteAsync(string? dropCsv, string? keepCsv, bool reset, string modelName)
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
            model.Columns ??= new Dictionary<string, ColumnOverride>();

            if (reset)
            {
                var n = FeatureSelector.Reset(model.Columns);
                AnsiConsole.MarkupLine($"[green]Reset {n} ignore override(s).[/]");
            }
            else if (!string.IsNullOrWhiteSpace(dropCsv))
            {
                FeatureSelector.Drop(model.Columns, SplitCsv(dropCsv));
                AnsiConsole.MarkupLine($"[green]Excluded:[/] {dropCsv}");
            }
            else if (!string.IsNullOrWhiteSpace(keepCsv))
            {
                var trainPath = ResolveTrainPath(ctx, config);
                if (trainPath == null || !File.Exists(trainPath))
                {
                    AnsiConsole.MarkupLine("[red]--keep requires train data to compute the complement.[/]");
                    AnsiConsole.MarkupLine("[grey]Expected datasets/train.csv or data.train in mloop.yaml.[/]");
                    return 1;
                }
                var allColumns = ReadHeaderColumns(trainPath);
                FeatureSelector.ApplyKeep(model.Columns, allColumns, SplitCsv(keepCsv), model.Label);
                AnsiConsole.MarkupLine($"[green]Kept:[/] {keepCsv} [grey](+ label {model.Label}); others excluded[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[yellow]Specify --drop, --keep, or --reset.[/]");
                return 1;
            }

            PrintIgnored(model.Columns);
            // Drop an empty overrides map so OmitNull keeps `columns:` out of mloop.yaml.
            if (model.Columns is { Count: 0 }) model.Columns = null;
            await ctx.ConfigLoader.SaveUserConfigAsync(config);
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "features select");
            return 1;
        }
    }

    /// <summary>Reads the CSV header column names without mutating the file (encoding-aware).</summary>
    internal static IReadOnlyList<string> ReadHeaderColumns(string csvPath)
    {
        var (converted, _) = EncodingDetector.ConvertToUtf8WithBom(csvPath);
        var flattened = CsvDataLoader.FlattenMultiLineHeaders(converted);
        try
        {
            using var reader = new StreamReader(flattened, System.Text.Encoding.UTF8, true);
            var header = reader.ReadLine();
            return string.IsNullOrEmpty(header)
                ? Array.Empty<string>()
                : CsvFieldParser.ParseFields(header);
        }
        finally
        {
            // Best-effort cleanup of temp files (delete flattened first; it may derive from converted).
            TryDeleteTemp(flattened, csvPath);
            TryDeleteTemp(converted, csvPath);
        }
    }

    private static void TryDeleteTemp(string path, string original)
    {
        if (!string.Equals(path, original, StringComparison.OrdinalIgnoreCase))
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { /* best-effort temp cleanup */ }
        }
    }

    private static string? ResolveTrainPath(CommandContext ctx, MLoopConfig config) =>
        !string.IsNullOrEmpty(config.Data?.Train)
            ? ctx.FileSystem.CombinePath(ctx.ProjectRoot, config.Data.Train)
            : ctx.FileSystem.CombinePath(ctx.ProjectRoot, "datasets", "train.csv");

    private static List<string> SplitCsv(string csv) =>
        csv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();

    private static void PrintIgnored(Dictionary<string, ColumnOverride> columns)
    {
        var ignored = columns.Where(kv => kv.Value.Type == "ignore").Select(kv => kv.Key).ToList();
        AnsiConsole.MarkupLine(ignored.Count == 0
            ? "[grey]No columns excluded.[/]"
            : $"[grey]Excluded columns ({ignored.Count}): {string.Join(", ", ignored)}[/]");
    }
}
