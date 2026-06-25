using System.CommandLine;
using DataLens.Models;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop analyze - granular, read-only EDA aspects for feature-engineering decisions.
/// Decomposes `mloop info --analyze` so an agent (or user) can request one aspect at a time.
/// Read-only: never mutates the data file or mloop.yaml.
/// </summary>
public static class AnalyzeCommand
{
    public static Command Create()
    {
        var command = new Command("analyze",
            "Granular, read-only EDA aspects (profile, correlation, importance, outliers, distribution)");
        command.Subcommands.Add(CreateProfileCommand());
        // Tasks 2-5 add: correlation, importance, outliers, distribution
        return command;
    }

    internal readonly record struct AnalyzeContext(string DataFile, string? Label);

    // Shared option/argument factories (one instance per subcommand call).
    internal static Argument<string> DataFileArg() =>
        new("data-file") { Description = "Path to the CSV dataset to analyze" };

    internal static Option<string?> LabelOption() =>
        new("--label", "-l") { Description = "Label/target column name (overrides mloop.yaml)" };

    internal static Option<string> NameOption() =>
        new("--name", "-n")
        {
            Description = "Model name to read label configuration from mloop.yaml",
            DefaultValueFactory = _ => "default"
        };

    internal static Option<bool> JsonOption() =>
        new("--json") { Description = "Output structured JSON instead of a console table" };

    /// <summary>
    /// Resolves the data file path (project-relative ok) and the label column
    /// (--label → mloop.yaml model → null). Prints an error and returns null if the file is missing.
    /// </summary>
    internal static async Task<AnalyzeContext?> ResolveAsync(
        string dataFile, string? labelOption, string modelName)
    {
        var fileSystem = new FileSystemManager();
        var projectDiscovery = new ProjectDiscovery(fileSystem);

        string? projectRoot = null;
        try { projectRoot = projectDiscovery.FindRoot(); } catch { /* not in a project: ok */ }

        var resolved = (projectRoot != null && !Path.IsPathRooted(dataFile))
            ? Path.Combine(projectRoot, dataFile)
            : dataFile;

        if (!File.Exists(resolved))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] File not found: {resolved}");
            return null;
        }

        string? label = labelOption;
        if (label == null && projectRoot != null)
        {
            var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
            var config = await configLoader.LoadUserConfigAsync();
            if (config?.Models != null && config.Models.TryGetValue(modelName, out var def)
                && !string.IsNullOrEmpty(def.Label))
            {
                label = def.Label;
            }
        }

        return new AnalyzeContext(resolved, label);
    }

    /// <summary>Emits an envelope as JSON (machine) or via the console renderer (human).</summary>
    internal static void Emit(AnalyzeEnvelope env, bool json, Action<AnalyzeEnvelope> renderConsole)
    {
        if (json) Console.WriteLine(AnalyzeJson.Serialize(env));
        else if (!env.Available)
            AnsiConsole.MarkupLine("[yellow]Warning:[/] DataLens is not available. Install the DataLens NuGet package to enable analysis.");
        else renderConsole(env);
    }

    private static Command CreateProfileCommand()
    {
        var dataFileArg = DataFileArg();
        var labelOption = LabelOption();
        var nameOption = NameOption();
        var jsonOption = JsonOption();

        var cmd = new Command("profile", "Column types, null %, cardinality, constant columns");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            return ExecuteProfileAsync(dataFile, label, modelName, json);
        });

        return cmd;
    }

    private static async Task<int> ExecuteProfileAsync(
        string dataFile, string? labelOption, string modelName, bool json)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName);
            if (ctx == null) return 1;

            var dataLens = new Infrastructure.ML.DataLensAnalyzer();
            if (!dataLens.IsAvailable)
            {
                Emit(AnalyzeJson.Unavailable("profile"), json, _ => { });
                return 0;
            }

            // DataLens needs the original .csv path (its own CsvBridge loading).
            var profile = await dataLens.ProfileAsync(ctx.Value.DataFile);

            // Column stats (missing/unique) come from MLoop's own scan, matching `info`.
            var (rowCount, stats) = ComputeColumnStats(ctx.Value.DataFile);

            var env = AnalyzeJson.MapProfile(profile, stats, rowCount);
            Emit(env, json, e => RenderProfileConsole(e, profile, stats, rowCount));
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze profile");
            return 1;
        }
    }

    /// <summary>Counts rows and per-column (missing, unique-sample) from the CSV header + body.</summary>
    private static (int RowCount, Dictionary<string, (long MissingCount, int UniqueCount)> Stats)
        ComputeColumnStats(string csvPath)
    {
        var (converted, _) = EncodingDetector.ConvertToUtf8WithBom(csvPath);
        var flattened = CsvDataLoader.FlattenMultiLineQuotedFields(converted);
        flattened = CsvDataLoader.FlattenMultiLineHeaders(flattened);
        flattened = CsvDataLoader.RemoveIndexColumns(flattened);

        using var reader = new StreamReader(flattened, System.Text.Encoding.UTF8, true);
        var header = reader.ReadLine();
        if (string.IsNullOrEmpty(header))
            return (0, new Dictionary<string, (long, int)>());

        var columns = CsvFieldParser.ParseFields(header);
        var missing = new long[columns.Length];
        var seen = new HashSet<string>[columns.Length];
        for (int i = 0; i < columns.Length; i++) seen[i] = new HashSet<string>();

        int rows = 0;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            rows++;
            var fields = CsvFieldParser.ParseFields(line);
            for (int i = 0; i < columns.Length; i++)
            {
                var v = i < fields.Length ? fields[i] : "";
                if (string.IsNullOrWhiteSpace(v)) missing[i]++;
                else if (seen[i].Count <= 10_000) seen[i].Add(v);
            }
        }

        var stats = new Dictionary<string, (long, int)>();
        for (int i = 0; i < columns.Length; i++)
            stats[columns[i]] = (missing[i], seen[i].Count);
        return (rows, stats);
    }

    private static void RenderProfileConsole(
        AnalyzeEnvelope env,
        ProfileReport? profile,
        Dictionary<string, (long MissingCount, int UniqueCount)> stats,
        int rowCount)
    {
        AnsiConsole.MarkupLine($"[blue]Profile:[/] {Spectre.Console.Markup.Escape(env.Summary)}");
        AnsiConsole.WriteLine();
        InfoPresenter.DisplayDataStatistics(stats.Keys.ToArray(), stats, rowCount, profile);
        foreach (var flag in env.Flags)
            AnsiConsole.MarkupLine($"[yellow]Flag:[/] {Spectre.Console.Markup.Escape(flag)}");
    }
}
