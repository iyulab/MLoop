using System.CommandLine;
using DataLens;
using DataLens.Analyzers;
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
            "Granular, read-only EDA aspects (profile, correlation, importance, outliers, distribution, series)");
        command.Subcommands.Add(CreateProfileCommand());
        command.Subcommands.Add(CreateCorrelationCommand());
        command.Subcommands.Add(CreateImportanceCommand());
        command.Subcommands.Add(CreateOutliersCommand());
        command.Subcommands.Add(CreateDistributionCommand());
        command.Subcommands.Add(CreateSeriesCommand());
        return command;
    }

    internal readonly record struct AnalyzeContext(string DataFile, string? Label);

    // Shared option/argument factories (one instance per subcommand call).
    internal static Argument<string?> DataFileArg() =>
        new("data-file")
        {
            Description = "Path to the CSV dataset to analyze (defaults to the project's data.train if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

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
    /// Resolves the data file path and the label column for an analyze invocation.
    /// <para>
    /// Data file: an explicit <paramref name="dataFile"/> (project-relative ok) is used as-is;
    /// when omitted, it defaults to the project's configured train data via the same resolver
    /// as <c>mloop train</c> (<see cref="TrainDataValidator.ResolveDataFileAsync"/>:
    /// <c>data.train</c> → <c>datasets/train.csv</c> discovery). This keeps CLI and the MCP
    /// bridge consistent so an agent need not guess the data path.
    /// </para>
    /// Label: <c>--label</c> → mloop.yaml model → null. Prints an error and returns null if no
    /// data file can be resolved. Read-only: never mutates the data file or mloop.yaml.
    /// </summary>
    internal static async Task<AnalyzeContext?> ResolveAsync(
        string? dataFile, string? labelOption, string modelName, string? records = null)
    {
        var fileSystem = new FileSystemManager();
        var projectDiscovery = new ProjectDiscovery(fileSystem);

        string? projectRoot = null;
        try { projectRoot = projectDiscovery.FindRoot(); } catch { /* not in a project: ok */ }

        var config = projectRoot != null
            ? await new ConfigLoader(fileSystem, projectDiscovery).LoadUserConfigAsync()
            : null;

        string? resolved;
        if (projectRoot != null)
        {
            // Reuse train's resolver: explicit path → data.train → datasets/train.csv discovery.
            resolved = await TrainDataValidator.ResolveDataFileAsync(
                dataFile, config!, projectRoot, new DatasetDiscovery(fileSystem), fileSystem);
        }
        else
        {
            // Outside a project only an explicit, existing path works (no config/discovery).
            resolved = !string.IsNullOrEmpty(dataFile) && TabularDataFile.Exists(dataFile) ? dataFile : null;
        }

        if (resolved == null || !TabularDataFile.Exists(resolved))
        {
            if (string.IsNullOrEmpty(dataFile))
                AnsiConsole.MarkupLine(
                    "[red]Error:[/] No data file specified and none found. Pass [blue]<data-file>[/] or set [blue]data.train[/] in mloop.yaml.");
            else
                ErrorConsole.Error(
                    $"File not found: {resolved ?? dataFile}",
                    projectRoot != null ? ErrorConsole.PathNotFoundTip(projectRoot) : ErrorConsole.PathNotFoundTipCwd());
            return null;
        }

        string? label = labelOption;
        if (label == null && config?.Models != null
            && config.Models.TryGetValue(modelName, out var def)
            && !string.IsNullOrEmpty(def.Label))
        {
            label = def.Label;
        }

        // A data file in another format (Parquet, Excel, JSON, TSV) is read as a table from here on.
        return new AnalyzeContext(await TabularDataFile.AsCsvAsync(resolved, records), label);
    }

    /// <summary>
    /// Runs one <c>analyze</c> sub-command with the stdout guarantee its <c>--json</c> flag promises.
    /// </summary>
    /// <remarks>
    /// Every sub-command here declares <c>--json</c> and none of them opened a scope, so narration —
    /// the loader's <c>[Info] Removed index column(s): …</c> among it — went to the same stdout as
    /// the document, and an error exit left that stdout empty. Five handlers would need five copies
    /// of the fix and a sixth sub-command would ship without one; routing them all through here makes
    /// the scope a property of running an analysis rather than something each handler remembers.
    /// </remarks>
    private static async Task<int> WithJsonContract(bool json, Func<Task<int>> run)
    {
        using var scope = json ? new JsonOutputScope() : null;
        return await run().ConfigureAwait(false);
    }

    /// <summary>Emits an envelope as JSON (machine) or via the console renderer (human).</summary>
    internal static void Emit(AnalyzeEnvelope env, bool json, Action<AnalyzeEnvelope> renderConsole)
    {
        if (json) Console.WriteLine(AnalyzeJson.Serialize(env with { Warnings = JsonOutputScope.Warnings }));
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
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("profile", "Column types, null %, cardinality, constant columns");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteProfileAsync(dataFile, label, modelName, json, records));
        });

        return cmd;
    }

    private static async Task<int> ExecuteProfileAsync(
        string? dataFile, string? labelOption, string modelName, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName, records);
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
            // The columns the profile reads as text or categories are asked whether they are numbers a reader would call numbers.
            var textColumns = profile?.Columns?.Where(c => c.DataType.Equals("text", StringComparison.OrdinalIgnoreCase)
                                                        || c.DataType.Equals("categorical", StringComparison.OrdinalIgnoreCase))
                .Select(c => c.Name).ToHashSet(StringComparer.Ordinal);
            var (rowCount, stats, numberLikeText) = ComputeColumnStats(ctx.Value.DataFile, textColumns);

            // surface strictly-increasing ID/index columns (same detector train uses) as a profile
            // flag, so the agent's FE loop can drop them. analyze was silent while train warned (inconsistency).
            var (convertedForScan, _) = EncodingDetector.ConvertToUtf8WithBom(ctx.Value.DataFile);
            var monotonic = CsvDataLoader.DetectMonotonicColumns(convertedForScan, ctx.Value.Label);

            var env = AnalyzeJson.MapProfile(profile, stats, rowCount, monotonic, numberLikeText);
            Emit(env, json, e => RenderProfileConsole(e, profile, stats, rowCount));
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze profile");
            return 1;
        }
    }

    /// <summary>How many data rows are read to judge whether a text column is numbers written for people (as <c>train</c> and <c>info</c> do).</summary>
    private const int FormattedNumberSampleRows = 1000;

    /// <summary>
    /// Counts rows and per-column (missing, unique-sample) from the CSV header + body. For each column of
    /// <paramref name="textColumns"/> (the ones the profile reads as text or as categories) it also says when the values are numbers a reader would call
    /// numbers: a few cells that are not (<c>N/A</c>) keep the profile from reading the column as one and give it no statistics,
    /// and a thousands separator, a percent sign or a currency sign does the same to a column of all numbers.
    /// </summary>
    private static (int RowCount, Dictionary<string, (long MissingCount, int UniqueCount)> Stats, IReadOnlyList<string> NumberLikeText)
        ComputeColumnStats(string csvPath, IReadOnlySet<string>? textColumns = null)
    {
        var (converted, _) = EncodingDetector.ConvertToUtf8WithBom(csvPath);
        var flattened = CsvDataLoader.FlattenMultiLineQuotedFields(converted, CoreNarration.Sink);
        flattened = CsvDataLoader.FlattenMultiLineHeaders(flattened, CoreNarration.Sink);
        flattened = CsvDataLoader.RemoveIndexColumns(flattened, CoreNarration.Sink);

        using var reader = new StreamReader(flattened, System.Text.Encoding.UTF8, true);
        var header = reader.ReadLine();
        if (string.IsNullOrEmpty(header))
            return (0, new Dictionary<string, (long, int)>(), []);

        var columns = CsvFieldParser.ParseFields(header);
        var missing = new long[columns.Length];
        var seen = new HashSet<string>[columns.Length];
        for (int i = 0; i < columns.Length; i++) seen[i] = new HashSet<string>();

        var asked = columns.Select(c => textColumns?.Contains(c) == true).ToArray();
        var census = new NonNumericCells(asked);
        var sample = asked.Select(a => a ? new List<string>() : null).ToArray();

        int rows = 0;
        long lineNumber = 1;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            lineNumber++;
            if (!TableRows.IsRow(line, columns.Length))
                continue;
            rows++;
            var fields = CsvFieldParser.ParseFields(line);
            census.Add(fields, lineNumber);
            for (int i = 0; i < columns.Length; i++)
            {
                var v = i < fields.Length ? fields[i] : "";
                if (string.IsNullOrWhiteSpace(v)) missing[i]++;
                else if (seen[i].Count <= 10_000) seen[i].Add(v);
                if (rows <= FormattedNumberSampleRows)
                    sample[i]?.Add(v);
            }
        }

        var stats = new Dictionary<string, (long, int)>();
        var numberLike = new List<string>();
        for (int i = 0; i < columns.Length; i++)
        {
            stats[columns[i]] = (missing[i], seen[i].Count);

            if (!asked[i])
                continue;
            var (stray, firstLine, example) = census.Of(i);
            if (census.MostlyNumbers(i))
                numberLike.Add($"numbers-read-as-text: {columns[i]} ({stray:N0} of {rows:N0} cells are not numbers, first '{example}' on line {firstLine:N0}; " +
                               "the profile does not read the column as numbers and gives it no statistics)");
            else if (FormattedNumbers.Looks(sample[i]!, out var formatted) && formatted is not null)
                numberLike.Add($"formatted-numbers: {columns[i]} (e.g. {formatted} - a thousands separator, percent sign or currency sign makes numbers text; " +
                               "write the plain number)");
        }
        return (rows, stats, numberLike);
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

    private static Command CreateCorrelationCommand()
    {
        var dataFileArg = DataFileArg();
        var labelOption = LabelOption();
        var nameOption = NameOption();
        var jsonOption = JsonOption();
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("correlation", "High-correlation pairs and multicollinearity");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteCorrelationAsync(dataFile, label, modelName, json, records));
        });
        return cmd;
    }

    private static async Task<int> ExecuteCorrelationAsync(
        string? dataFile, string? labelOption, string modelName, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName, records);
            if (ctx == null) return 1;

            var dataLens = new Infrastructure.ML.DataLensAnalyzer();
            if (!dataLens.IsAvailable) { Emit(AnalyzeJson.Unavailable("correlation"), json, _ => { }); return 0; }

            var result = await dataLens.AnalyzeAsync(ctx.Value.DataFile, Infrastructure.ML.DataLensAnalyzer.Only(o =>
            {
                o.IncludeCorrelation = true;
            }));

            var env = AnalyzeJson.MapCorrelation(result?.Correlation);
            Emit(env, json, _ =>
            {
                if (result?.Correlation != null) InfoPresenter.DisplayCorrelation(result.Correlation);
                else AnsiConsole.MarkupLine("[grey]No numeric columns to correlate.[/]");
            });
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze correlation");
            return 1;
        }
    }

    private static Command CreateImportanceCommand()
    {
        var dataFileArg = DataFileArg();
        var labelOption = LabelOption();
        var nameOption = NameOption();
        var jsonOption = JsonOption();
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("importance", "Feature importance ranking (requires a label)");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteImportanceAsync(dataFile, label, modelName, json, records));
        });
        return cmd;
    }

    private static async Task<int> ExecuteImportanceAsync(
        string? dataFile, string? labelOption, string modelName, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName, records);
            if (ctx == null) return 1;

            if (string.IsNullOrEmpty(ctx.Value.Label))
            {
                ErrorConsole.Error("analyze importance requires a label. Pass [blue]--label <col>[/] or set it in mloop.yaml.");
                return 1;
            }

            var dataLens = new Infrastructure.ML.DataLensAnalyzer();
            if (!dataLens.IsAvailable) { Emit(AnalyzeJson.Unavailable("importance"), json, _ => { }); return 0; }

            var result = await dataLens.AnalyzeAsync(ctx.Value.DataFile, Infrastructure.ML.DataLensAnalyzer.Only(o =>
            {
                o.IncludeFeatures = true;
                o.TargetColumn = ctx.Value.Label;
            }));

            var env = AnalyzeJson.MapImportance(result?.Features, ctx.Value.Label);
            Emit(env, json, _ =>
            {
                var (method, ranking) = AnalyzeJson.RankImportance(result?.Features, ctx.Value.Label);
                if (method == null || ranking.Count == 0)
                {
                    AnsiConsole.MarkupLine("[grey]No feature importance scores available.[/]");
                    return;
                }
                var table = new Table().Border(TableBorder.Rounded);
                table.AddColumn("Feature");
                table.AddColumn(new TableColumn($"Importance ({method})").RightAligned());
                foreach (var r in ranking)
                    table.AddRow(Markup.Escape(r.Feature), r.Score.ToString("F4"));
                AnsiConsole.Write(table);
            });
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze importance");
            return 1;
        }
    }

    private static Command CreateOutliersCommand()
    {
        var dataFileArg = DataFileArg();
        var labelOption = LabelOption();
        var nameOption = NameOption();
        var jsonOption = JsonOption();
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("outliers", "Outlier count, rate, and isolation-forest threshold");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteOutliersAsync(dataFile, label, modelName, json, records));
        });
        return cmd;
    }

    private static async Task<int> ExecuteOutliersAsync(
        string? dataFile, string? labelOption, string modelName, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName, records);
            if (ctx == null) return 1;

            var dataLens = new Infrastructure.ML.DataLensAnalyzer();
            if (!dataLens.IsAvailable) { Emit(AnalyzeJson.Unavailable("outliers"), json, _ => { }); return 0; }

            var result = await dataLens.AnalyzeAsync(ctx.Value.DataFile, Infrastructure.ML.DataLensAnalyzer.Only(o =>
            {
                o.IncludeOutliers = true;
            }));

            var env = AnalyzeJson.MapOutliers(result?.Outliers);
            Emit(env, json, _ =>
            {
                if (result?.Outliers != null) InfoPresenter.DisplayOutlierSummary(result.Outliers);
                else AnsiConsole.MarkupLine("[grey]No outlier analysis available.[/]");
            });
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze outliers");
            return 1;
        }
    }

    private static Command CreateDistributionCommand()
    {
        var dataFileArg = DataFileArg();
        var labelOption = LabelOption();
        var nameOption = NameOption();
        var jsonOption = JsonOption();
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("distribution", "Skewness, kurtosis, and normality tests per numeric column");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(labelOption);
        cmd.Options.Add(nameOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteDistributionAsync(dataFile, label, modelName, json, records));
        });
        return cmd;
    }

    private static async Task<int> ExecuteDistributionAsync(
        string? dataFile, string? labelOption, string modelName, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption, modelName, records);
            if (ctx == null) return 1;

            var dataLens = new Infrastructure.ML.DataLensAnalyzer();
            if (!dataLens.IsAvailable) { Emit(AnalyzeJson.Unavailable("distribution"), json, _ => { }); return 0; }

            var result = await dataLens.AnalyzeAsync(ctx.Value.DataFile, Infrastructure.ML.DataLensAnalyzer.Only(o =>
            {
                o.IncludeDescriptive = true;
                o.IncludeDistribution = true;
            }));

            // Per-column cardinality (MLoop's own scan, same as `profile`) lets MapDistribution
            // suppress non-actionable skew flags on low-cardinality/categorical columns.
            var (_, stats, _) = ComputeColumnStats(ctx.Value.DataFile);

            var env = AnalyzeJson.MapDistribution(result?.Descriptive, result?.Distribution, stats);
            Emit(env, json, _ =>
            {
                if (result?.Descriptive?.Columns is { Count: > 0 }) InfoPresenter.DisplayDescriptiveStatistics(result.Descriptive);
                if (result?.Distribution?.Columns is { Count: > 0 }) InfoPresenter.DisplayDistributions(result.Distribution);
            });
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze distribution");
            return 1;
        }
    }

    private static Command CreateSeriesCommand()
    {
        var dataFileArg = DataFileArg();
        var columnOption = new Option<string?>("--column", "-c")
        {
            Description = "Series column to analyze (auto-selected when the CSV has a single column)"
        };
        var maxLagOption = new Option<int?>("--max-lag")
        {
            Description = "Largest lag of the autocorrelation functions (default: min(10*log10 n, n - 1))"
        };
        var aroundTrendOption = new Option<bool>("--around-trend")
        {
            Description = "Test stationarity around a linear trend instead of a constant level"
        };
        var jsonOption = JsonOption();
        var recordsOption = DataFileOptions.Records();

        var cmd = new Command("series",
            "Time-series structure: ACF/PACF with bands, ADF and KPSS stationarity tests, dominant period");
        cmd.Arguments.Add(dataFileArg);
        cmd.Options.Add(columnOption);
        cmd.Options.Add(maxLagOption);
        cmd.Options.Add(aroundTrendOption);
        cmd.Options.Add(jsonOption);
        cmd.Options.Add(recordsOption);

        cmd.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var column = parseResult.GetValue(columnOption);
            var maxLag = parseResult.GetValue(maxLagOption);
            var aroundTrend = parseResult.GetValue(aroundTrendOption);
            var json = parseResult.GetValue(jsonOption);
            var records = parseResult.GetValue(recordsOption);
            return WithJsonContract(json, () => ExecuteSeriesAsync(dataFile, column, maxLag, aroundTrend, json, records));
        });
        return cmd;
    }

    private static async Task<int> ExecuteSeriesAsync(
        string? dataFile, string? column, int? maxLag, bool aroundTrend, bool json, string? records)
    {
        try
        {
            var ctx = await ResolveAsync(dataFile, labelOption: null, modelName: "default", records);
            if (ctx == null) return 1;

            var series = await DetectCommand.LoadSeriesAsync(ctx.Value.DataFile, column, json);
            if (series == null) return 1;
            var (values, resolvedColumn) = series.Value;

            var autocorrelation = TimeSeriesAnalyzer.Autocorrelation(values, maxLag);
            var stationarity = TimeSeriesAnalyzer.Stationarity(values, aroundTrend);
            var period = TimeSeriesAnalyzer.EstimatePeriod(values);

            var env = AnalyzeJson.MapSeries(resolvedColumn, autocorrelation, stationarity, period);
            Emit(env, json, e => RenderSeriesConsole(e, autocorrelation, stationarity));
            return 0;
        }
        catch (ArgumentException ex)
        {
            // The parameter-name suffix .NET appends ("(Parameter 'series')") names a variable in
            // library code, not anything the user typed.
            var message = ex.ParamName is { } name ? ex.Message.Replace($" (Parameter '{name}')", "") : ex.Message;
            ErrorConsole.Error(Markup.Escape(message));
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "analyze series");
            return 1;
        }
    }

    private static void RenderSeriesConsole(
        AnalyzeEnvelope env, SeriesAutocorrelation autocorrelation, SeriesStationarity stationarity)
    {
        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(env.Summary)}[/]");

        var table = new Table().AddColumn("Test").AddColumn("Statistic").AddColumn("Reading");
        table.AddRow("ADF (H0: unit root)", AnalyzeJson.Invariant($"{stationarity.Adf.Statistic:F3}"),
            AnalyzeJson.AdfReading(stationarity.Adf));
        table.AddRow("KPSS (H0: stationary)", AnalyzeJson.Invariant($"{stationarity.Kpss.Statistic:F3}"),
            AnalyzeJson.KpssReading(stationarity.Kpss));
        AnsiConsole.Write(table);

        var acf = AnalyzeJson.SignificantLags(autocorrelation.Acf, autocorrelation.AcfThreshold, firstLag: 0);
        var pacf = AnalyzeJson.SignificantLags(autocorrelation.Pacf, autocorrelation.PacfThreshold, firstLag: 1);
        AnsiConsole.MarkupLine($"ACF beyond the white-noise band (lags): {Markup.Escape(acf.Count == 0 ? "none" : string.Join(", ", acf))}");
        AnsiConsole.MarkupLine($"PACF beyond the white-noise band (lags): {Markup.Escape(pacf.Count == 0 ? "none" : string.Join(", ", pacf))}");

        foreach (var flag in env.Flags)
            AnsiConsole.MarkupLine($"[yellow]•[/] {Markup.Escape(flag)}");
    }
}
