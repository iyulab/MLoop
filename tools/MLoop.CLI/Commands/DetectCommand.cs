using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using DataLens.Analyzers;
using DataLens.Models;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Data;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop detect - One-shot time-series anomaly detection over an entire series (no train/predict
/// split, no model artifact), by spectral residual saliency. Every point gets an anomaly verdict, a
/// score, an expected value and control limits (chart these); the series' dominant period is
/// reported. Works on any CSV — does not require an MLoop project.
/// </summary>
/// <remarks>
/// The statistics come from DataLens' <see cref="TimeSeriesAnalyzer"/>, which runs on UInsight's own
/// transform and so works on every platform UInsight ships for. The earlier implementation ran on
/// ML.NET's SR-CNN, whose FFT native is missing on Apple silicon and assumes a system OpenMP on
/// Linux, and which on a seasonal series flagged a whole cycle around one spike while missing
/// spikes on a sine and level shifts entirely.
/// </remarks>
public static class DetectCommand
{
    /// <summary>The paper's threshold (Ren et al. 2019).</summary>
    internal const double DefaultThreshold = 3.0;

    /// <summary>Shewhart 3-sigma: the control limits a control chart draws by default.</summary>
    internal const double DefaultSensitivity = 99.73;

    /// <summary>A detection pass: the per-point scores and the series' period.</summary>
    internal sealed record Detection(SeriesAnomalyReport Report, SeriesPeriod Period)
    {
        public int AnomalyCount => Report.Anomalies.Count;
    }

    /// <summary>
    /// Scores <paramref name="values"/> and estimates its period. The control limits are the
    /// library's band at <paramref name="sensitivity"/> percent coverage — the same robust residual
    /// scale this command used to compute itself (residual MAD × 1.4826), so there is one
    /// implementation of it.
    /// </summary>
    /// <exception cref="ArgumentException">Series too short, a non-finite value, or an option out of
    /// range — the message names which.</exception>
    internal static Detection Detect(IReadOnlyList<double> values, double threshold, double sensitivity)
    {
        var report = TimeSeriesAnalyzer.SpectralResidual(values, new SpectralResidualOptions
        {
            Threshold = threshold,
            Sensitivity = sensitivity
        });
        return new Detection(report, TimeSeriesAnalyzer.EstimatePeriod(values));
    }

    public static Command Create()
    {
        var dataFileArg = new Argument<string>("data-file")
        {
            Description = "Path to the CSV file containing the time series"
        };

        var columnOption = new Option<string?>("--column", "-c")
        {
            Description = "Value column to monitor (auto-selected when the CSV has a single column)"
        };

        var thresholdOption = new Option<double>("--threshold")
        {
            Description = "Score above which a point is an anomaly (> 0; the score is a point's saliency "
                          + "relative to the points before it)",
            DefaultValueFactory = _ => DefaultThreshold
        };

        var sensitivityOption = new Option<double>("--sensitivity")
        {
            Description = "Coverage of the control limits in percent (0-100, exclusive; 99.73 = 3 sigma)",
            DefaultValueFactory = _ => DefaultSensitivity
        };

        // Removed: the score assumes no period, so a supplied one could not change the result. Kept
        // hidden so a script that still passes it is told why instead of getting a parser error
        // about a stray argument.
        var removedPeriodOption = new Option<string?>("--period") { Hidden = true };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Write the full per-point result to this CSV file"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the full result as JSON (machine-readable)"
        };

        var command = new Command("detect", "One-shot time-series anomaly detection (spectral residual, no training required)");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(columnOption);
        command.Options.Add(thresholdOption);
        command.Options.Add(sensitivityOption);
        command.Options.Add(removedPeriodOption);
        command.Options.Add(outputOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var column = parseResult.GetValue(columnOption);
            var threshold = parseResult.GetValue(thresholdOption);
            var sensitivity = parseResult.GetValue(sensitivityOption);
            var output = parseResult.GetValue(outputOption);
            var json = parseResult.GetValue(jsonOption);
            if (parseResult.GetResult(removedPeriodOption) is not null)
            {
                using var _ = json ? new JsonOutputScope() : null;
                WriteError("--period was removed: detection no longer takes a period, so one could not change "
                           + "the result. The series' own period is reported instead (\"period\" in --json).", json);
                return Task.FromResult(1);
            }
            return ExecuteAsync(dataFile, column, threshold, sensitivity, output, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string dataFile,
        string? column,
        double threshold,
        double sensitivity,
        string? outputPath,
        bool jsonOutput)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            // Checked here, in the command's own terms, so the message names the flag rather than
            // the library property it becomes.
            if (!(threshold > 0) || !double.IsFinite(threshold))
            {
                WriteError(string.Create(CultureInfo.InvariantCulture,
                    $"--threshold must be greater than 0 (got {threshold})."), jsonOutput);
                return 1;
            }
            if (!(sensitivity > 0 && sensitivity < 100))
            {
                WriteError(string.Create(CultureInfo.InvariantCulture,
                    $"--sensitivity must be between 0 and 100, exclusive (got {sensitivity})."), jsonOutput);
                return 1;
            }

            if (!File.Exists(dataFile))
            {
                WriteError($"Data file not found: {dataFile}", jsonOutput);
                return 1;
            }

            var series = await LoadSeriesAsync(dataFile, column, jsonOutput);
            if (series == null)
                return 1;

            var (values, resolvedColumn) = series.Value;

            var result = Detect(values, threshold, sensitivity);

            if (outputPath != null)
                await WriteCsvAsync(outputPath, result);

            if (jsonOutput)
                OutputAsJson(resolvedColumn, result, outputPath);
            else
                OutputAsTable(dataFile, resolvedColumn, result, outputPath);

            return 0;
        }
        catch (ArgumentException ex)
        {
            // The parameter-name suffix .NET appends ("(Parameter 'series')") names a variable in
            // library code, not anything the user typed.
            var message = ex.ParamName is { } name ? ex.Message.Replace($" (Parameter '{name}')", "") : ex.Message;
            WriteError(message, jsonOutput);
            return 1;
        }
        catch (Exception ex)
        {
            if (jsonOutput)
            {
                WriteError(ex.Message, jsonOutput: true);
                return 1;
            }
            ErrorSuggestions.DisplayError(ex, "detect");
            return 1;
        }
    }

    /// <summary>Loads the target column as doubles; null (after printing an error) when unusable.</summary>
    internal static async Task<(IReadOnlyList<double> Values, string Column)?> LoadSeriesAsync(
        string dataFile, string? column, bool jsonOutput)
    {
        // CsvHelperImpl handles encoding auto-detection (CP949/EUC-KR → UTF-8).
        var rows = await new CsvHelperImpl().ReadAsync(dataFile);
        if (rows.Count == 0)
        {
            WriteError("The CSV file contains no data rows.", jsonOutput);
            return null;
        }

        var headers = rows[0].Keys.ToList();

        string? resolved;
        if (column != null)
        {
            resolved = headers.FirstOrDefault(h => string.Equals(h, column, StringComparison.OrdinalIgnoreCase));
            if (resolved == null)
            {
                WriteError($"Column '{column}' not found. Available columns: {string.Join(", ", headers)}", jsonOutput);
                return null;
            }
        }
        else if (headers.Count == 1)
        {
            resolved = headers[0];
        }
        else
        {
            WriteError($"The CSV has {headers.Count} columns — specify the series with --column. " +
                       $"Available columns: {string.Join(", ", headers)}", jsonOutput);
            return null;
        }

        var values = new List<double>(rows.Count);
        for (int i = 0; i < rows.Count; i++)
        {
            var raw = rows[i].GetValueOrDefault(resolved);
            if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            {
                // Row numbering: +2 = 1-based + header line. The transform needs a contiguous numeric
                // series, so a gap is an input error, not something to silently skip.
                WriteError($"Column '{resolved}' has a non-numeric value '{raw}' at data row {i + 2} — " +
                           "the series must be fully numeric with no gaps.", jsonOutput);
                return null;
            }
            values.Add(v);
        }

        return (values, resolved);
    }

    /// <summary>Per-point output columns. Control = the band to chart.</summary>
    /// <summary>Marks a listed anomaly that sits near an end of the series.</summary>
    internal const string EdgeMark = "*";

    internal const string CsvHeader = "Index,Value,IsAnomaly,Score,ExpectedValue,ControlLower,ControlUpper,NearEdge";

    internal static async Task WriteCsvAsync(string outputPath, Detection result)
    {
        var lines = new List<string>(result.Report.Points.Count + 1) { CsvHeader };
        foreach (var p in result.Report.Points)
        {
            lines.Add(string.Create(CultureInfo.InvariantCulture,
                $"{p.Index},{p.Value},{(p.IsAnomaly ? 1 : 0)},{p.Score},{p.Expected},{p.Lower},{p.Upper},{(p.NearEdge ? 1 : 0)}"));
        }
        await File.WriteAllLinesAsync(outputPath, lines);
    }

    private static void OutputAsJson(string column, Detection result, string? outputPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var payload = new
        {
            Column = column,
            TotalPoints = result.Report.Points.Count,
            result.AnomalyCount,
            // null = the series has no dominant period — a finding, not a failure
            result.Period.Period,
            OutputFile = outputPath,
            Points = result.Report.Points.Select(p => new
            {
                p.Index,
                p.Value,
                p.IsAnomaly,
                p.Score,
                ExpectedValue = p.Expected,
                ControlLower = p.Lower,
                ControlUpper = p.Upper,
                p.NearEdge
            })
        };

        Console.WriteLine(JsonSerializer.Serialize(payload, options));
    }

    private static void OutputAsTable(string dataFile, string column, Detection result, string? outputPath)
    {
        AnsiConsole.Write(new Rule($"[cyan]One-Shot Anomaly Detection - {Markup.Escape(Path.GetFileName(dataFile))}[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"Column: [cyan]{Markup.Escape(column)}[/]  Points: [cyan]{result.Report.Points.Count}[/]  " +
                               $"Anomalies: [{(result.AnomalyCount > 0 ? "red" : "green")}]{result.AnomalyCount}[/]  " +
                               $"Period: [cyan]{result.Period.Period?.ToString(CultureInfo.InvariantCulture) ?? "none"}[/]");
        AnsiConsole.WriteLine();

        if (result.AnomalyCount == 0)
        {
            AnsiConsole.MarkupLine("[green]No anomalies detected.[/]");
        }
        else
        {
            var table = new Table();
            table.Border(TableBorder.Rounded);
            table.AddColumn("Index");
            table.AddColumn("Value");
            table.AddColumn("Score");
            table.AddColumn("Expected");
            table.AddColumn("Control Lower");
            table.AddColumn("Control Upper");

            const int maxRows = 50;
            var anomalies = result.Report.Points.Where(p => p.IsAnomaly).Take(maxRows).ToList();
            foreach (var p in anomalies)
            {
                table.AddRow(
                    p.Index.ToString(CultureInfo.InvariantCulture) + (p.NearEdge ? EdgeMark : ""),
                    p.Value.ToString("G6", CultureInfo.InvariantCulture),
                    p.Score.ToString("F3", CultureInfo.InvariantCulture),
                    p.Expected.ToString("G6", CultureInfo.InvariantCulture),
                    p.Lower.ToString("G6", CultureInfo.InvariantCulture),
                    p.Upper.ToString("G6", CultureInfo.InvariantCulture));
            }

            AnsiConsole.Write(table);
            if (anomalies.Any(p => p.NearEdge))
                AnsiConsole.MarkupLine($"[grey]{Markup.Escape(EdgeMark)} near an end of the series, where the score is least " +
                                       "reliable — a flag seen only there is worth a second look.[/]");
            if (result.AnomalyCount > maxRows)
                AnsiConsole.MarkupLine($"[grey]Showing first {maxRows} of {result.AnomalyCount} anomalies " +
                                       "(use --output or --json for the full list)[/]");
        }

        AnsiConsole.WriteLine();
        if (outputPath != null)
            ValueLine.Write("[green]Full per-point result written to:[/] ", outputPath);
        else
            AnsiConsole.MarkupLine("[grey]Use [blue]--output result.csv[/] for per-point control limits or [blue]--json[/] for machine-readable output.[/]");
    }

    private static void WriteError(string message, bool jsonOutput)
    {
        // The JSON envelope stays on stdout for --json consumers; the cause goes to stderr through
        // the shared sink either way, so "exit != 0 ⇒ stderr has a cause" holds in every mode and the
        // "Error:" prefix reads the same as every other command. The message is escaped rather than
        // interpolated as markup: it carries user data (file paths, column names) that may contain
        // brackets Spectre would otherwise try to parse.
        if (jsonOutput)
            JsonError.Emit(message);

        ErrorConsole.Error(Markup.Escape(message));
    }
}
