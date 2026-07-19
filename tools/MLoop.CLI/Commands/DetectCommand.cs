using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Data;
using MLoop.Core.Detection;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop detect - One-shot time-series anomaly detection over an entire series (no train/predict
/// split, no model artifact). SR-CNN AnomalyAndMargin mode: every point gets an anomaly verdict,
/// score, and SPC-chart bounds (ExpectedValue/UpperBound/LowerBound). Works on any CSV — does not
/// require an MLoop project.
/// </summary>
public static class DetectCommand
{
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
            Description = "Anomaly decision threshold in [0, 1]",
            DefaultValueFactory = _ => 0.3
        };

        var sensitivityOption = new Option<double>("--sensitivity")
        {
            Description = "Boundary sensitivity in [0, 100] — larger = tighter bounds",
            DefaultValueFactory = _ => 99.0
        };

        var periodOption = new Option<int?>("--period")
        {
            Description = "Seasonality period in points (default: auto-detect; 0 = non-seasonal)"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Write the full per-point result to this CSV file"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the full result as JSON (machine-readable)"
        };

        var command = new Command("detect", "One-shot time-series anomaly detection (SR-CNN, no training required)");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(columnOption);
        command.Options.Add(thresholdOption);
        command.Options.Add(sensitivityOption);
        command.Options.Add(periodOption);
        command.Options.Add(outputOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var column = parseResult.GetValue(columnOption);
            var threshold = parseResult.GetValue(thresholdOption);
            var sensitivity = parseResult.GetValue(sensitivityOption);
            var period = parseResult.GetValue(periodOption);
            var output = parseResult.GetValue(outputOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(dataFile, column, threshold, sensitivity, period, output, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string dataFile,
        string? column,
        double threshold,
        double sensitivity,
        int? period,
        string? outputPath,
        bool jsonOutput)
    {
        try
        {
            if (!File.Exists(dataFile))
            {
                WriteError($"Data file not found: {dataFile}", jsonOutput);
                return 1;
            }

            var series = await LoadSeriesAsync(dataFile, column, jsonOutput);
            if (series == null)
                return 1;

            var (values, resolvedColumn) = series.Value;

            var result = SrCnnOneShotDetector.Detect(values, new OneShotAnomalyOptions
            {
                Threshold = threshold,
                Sensitivity = sensitivity,
                Period = period,
            });

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
            WriteError(ex.Message, jsonOutput);
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
                // Row numbering: +2 = 1-based + header line. SR-CNN needs a contiguous numeric series,
                // so a gap is an input error, not something to silently skip.
                WriteError($"Column '{resolved}' has a non-numeric value '{raw}' at data row {i + 2} — " +
                           "the series must be fully numeric with no gaps.", jsonOutput);
                return null;
            }
            values.Add(v);
        }

        return (values, resolved);
    }

    private static async Task WriteCsvAsync(string outputPath, OneShotAnomalyResult result)
    {
        var lines = new List<string>(result.Points.Count + 1)
        {
            "Index,Value,IsAnomaly,Score,ExpectedValue,UpperBound,LowerBound"
        };
        foreach (var p in result.Points)
        {
            lines.Add(string.Create(CultureInfo.InvariantCulture,
                $"{p.Index},{p.Value},{(p.IsAnomaly ? 1 : 0)},{p.Score},{p.ExpectedValue},{p.UpperBound},{p.LowerBound}"));
        }
        await File.WriteAllLinesAsync(outputPath, lines);
    }

    private static void OutputAsJson(string column, OneShotAnomalyResult result, string? outputPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var payload = new
        {
            Column = column,
            TotalPoints = result.Points.Count,
            result.AnomalyCount,
            result.Period,
            OutputFile = outputPath,
            Points = result.Points.Select(p => new
            {
                p.Index,
                p.Value,
                p.IsAnomaly,
                p.Score,
                p.ExpectedValue,
                p.UpperBound,
                p.LowerBound
            })
        };

        Console.WriteLine(JsonSerializer.Serialize(payload, options));
    }

    private static void OutputAsTable(string dataFile, string column, OneShotAnomalyResult result, string? outputPath)
    {
        AnsiConsole.Write(new Rule($"[cyan]One-Shot Anomaly Detection - {Path.GetFileName(dataFile)}[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"Column: [cyan]{column}[/]  Points: [cyan]{result.Points.Count}[/]  " +
                               $"Anomalies: [{(result.AnomalyCount > 0 ? "red" : "green")}]{result.AnomalyCount}[/]  " +
                               $"Period: [cyan]{(result.Period > 0 ? result.Period.ToString() : "none")}[/]");
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
            table.AddColumn("Lower");
            table.AddColumn("Upper");

            const int maxRows = 50;
            foreach (var p in result.Points.Where(p => p.IsAnomaly).Take(maxRows))
            {
                table.AddRow(
                    p.Index.ToString(),
                    p.Value.ToString("G6", CultureInfo.InvariantCulture),
                    p.Score.ToString("F3", CultureInfo.InvariantCulture),
                    p.ExpectedValue.ToString("G6", CultureInfo.InvariantCulture),
                    p.LowerBound.ToString("G6", CultureInfo.InvariantCulture),
                    p.UpperBound.ToString("G6", CultureInfo.InvariantCulture));
            }

            AnsiConsole.Write(table);
            if (result.AnomalyCount > maxRows)
                AnsiConsole.MarkupLine($"[grey]Showing first {maxRows} of {result.AnomalyCount} anomalies " +
                                       "(use --output or --json for the full list)[/]");
        }

        AnsiConsole.WriteLine();
        if (outputPath != null)
            AnsiConsole.MarkupLine($"[green]Full per-point result written to:[/] {outputPath}");
        else
            AnsiConsole.MarkupLine("[grey]Use [blue]--output result.csv[/] for per-point bounds or [blue]--json[/] for machine-readable output.[/]");
    }

    private static void WriteError(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            // The JSON envelope stays on stdout for --json consumers; the cause is mirrored to
            // stderr so "exit != 0 ⇒ stderr has a cause" holds in every mode.
            Console.WriteLine(JsonSerializer.Serialize(new { error = message }));
            Console.Error.WriteLine(message);
        }
        else
        {
            ErrorConsole.Out.MarkupLineInterpolated($"[red]Error:[/] {message}");
        }
    }
}
