using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Display;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for managing prediction feedback.
/// Supports recording feedback, listing entries, and calculating metrics.
/// </summary>
public static class FeedbackCommand
{
    public static Command Create()
    {
        var command = new Command("feedback", "Manage prediction feedback for model monitoring");

        command.Subcommands.Add(CreateAddCommand());
        command.Subcommands.Add(CreateListCommand());
        command.Subcommands.Add(CreateMetricsCommand());

        return command;
    }

    private static Command CreateAddCommand()
    {
        var predictionIdOption = new Option<string?>("--prediction-id")
        {
            Description = "ID of the prediction to provide feedback for (with --actual-value; or give --from-file for many)"
        };

        var actualValueOption = new Option<string?>("--actual-value")
        {
            Description = "The actual (ground truth) value"
        };

        var sourceOption = new Option<string?>("--source")
        {
            Description = "Where the value came from (e.g., 'user', 'reviewer-llm', 'system'). With --from-file: for the rows whose source cell is empty"
        };

        var weightOption = new Option<double?>("--weight", "-w")
        {
            Description = "How much this row counts in training and evaluation built from feedback "
                + "(for a row reviewed with probability p, 1/p). Default 1. With --from-file: for the rows whose weight cell is empty"
        };

        var fromFileOption = new Option<string?>("--from-file")
        {
            Description = "Record feedback for many predictions from a table (csv, xlsx, tsv, ...): one row per prediction, "
                + "its prediction id and the actual value in columns. Rows that cannot be recorded are listed by row number; the others are recorded"
        };

        var idColumnOption = new Option<string?>("--id-column")
        {
            Description = $"With --from-file: the column holding the prediction id (default {FeedbackFileColumns.Id})"
        };

        var actualColumnOption = new Option<string?>("--actual-column")
        {
            Description = $"With --from-file: the column holding the actual value (default {FeedbackFileColumns.Actual})"
        };

        var sourceColumnOption = new Option<string?>("--source-column")
        {
            Description = $"With --from-file: the column holding where each value came from (default {FeedbackFileColumns.Source}, read when the file has it)"
        };

        var weightColumnOption = new Option<string?>("--weight-column")
        {
            Description = $"With --from-file: the column holding each row's weight (default {FeedbackFileColumns.Weight}, read when the file has it)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the recorded feedback as JSON to stdout (narration goes to stderr)"
        };

        var command = new Command("add", "Record feedback for a prediction, or for many from a file");
        command.Options.Add(predictionIdOption);
        command.Options.Add(actualValueOption);
        command.Options.Add(sourceOption);
        command.Options.Add(weightOption);
        command.Options.Add(fromFileOption);
        command.Options.Add(idColumnOption);
        command.Options.Add(actualColumnOption);
        command.Options.Add(sourceColumnOption);
        command.Options.Add(weightColumnOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var json = parseResult.GetValue(jsonOption);
            var predictionId = parseResult.GetValue(predictionIdOption);
            var actualValue = parseResult.GetValue(actualValueOption);
            var source = parseResult.GetValue(sourceOption);
            var weight = parseResult.GetValue(weightOption);
            var fromFile = parseResult.GetValue(fromFileOption);
            var columns = new FeedbackFileColumns(
                parseResult.GetValue(idColumnOption),
                parseResult.GetValue(actualColumnOption),
                parseResult.GetValue(sourceColumnOption),
                parseResult.GetValue(weightColumnOption));
            return ExecuteAddAsync(predictionId, actualValue, source, weight, fromFile, columns, json);
        });

        return command;
    }

    private static Command CreateListCommand()
    {
        var nameOption = new Option<string>("--name", "-n")
        {
            Description = "Model name",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
        };

        var limitOption = new Option<int>("--limit")
        {
            Description = "Maximum number of entries to show",
            DefaultValueFactory = _ => 20
        };

        var fromOption = new Option<DateTime?>("--from")
        {
            Description = "Start date filter (yyyy-MM-dd)"
        };

        var toOption = new Option<DateTime?>("--to")
        {
            Description = "End date filter (yyyy-MM-dd)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("list", "List feedback entries for a model");
        command.Options.Add(nameOption);
        command.Options.Add(limitOption);
        command.Options.Add(fromOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var limit = parseResult.GetValue(limitOption);
            var from = parseResult.GetValue(fromOption);
            var to = parseResult.GetValue(toOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteListAsync(modelName, limit, from, to, json);
        });

        return command;
    }

    private static Command CreateMetricsCommand()
    {
        var nameOption = new Option<string>("--name", "-n")
        {
            Description = "Model name",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
        };

        var fromOption = new Option<DateTime?>("--from")
        {
            Description = "Start date filter (yyyy-MM-dd)"
        };

        var toOption = new Option<DateTime?>("--to")
        {
            Description = "End date filter (yyyy-MM-dd)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("metrics", "Calculate accuracy metrics from feedback");
        command.Options.Add(nameOption);
        command.Options.Add(fromOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var from = parseResult.GetValue(fromOption);
            var to = parseResult.GetValue(toOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteMetricsAsync(modelName, from, to, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAddAsync(
        string? predictionId,
        string? actualValue,
        string? source,
        double? weight,
        string? fromFile,
        FeedbackFileColumns columns,
        bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        if (fromFile is null && columns.AnyNamed)
        {
            ErrorConsole.Error(
                "--id-column, --actual-column, --source-column and --weight-column name columns of a file.",
                "Give the file with --from-file, or drop the column options.");
            return 1;
        }

        if (fromFile is not null)
        {
            if (predictionId is not null || actualValue is not null)
            {
                ErrorConsole.Error(
                    "--from-file cannot be combined with --prediction-id or --actual-value.",
                    "The file holds one row per prediction — name its columns with --id-column and --actual-column.");
                return 1;
            }

            return await ExecuteAddFromFileAsync(fromFile, source, weight, columns, jsonOutput);
        }

        if (predictionId is null || actualValue is null)
        {
            ErrorConsole.Error(
                "`feedback add` needs the prediction it answers and the actual value.",
                "Give --prediction-id and --actual-value for one prediction, or --from-file <table> for many.");
            return 1;
        }

        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            await collector.RecordFeedbackAsync(
                predictionId,
                actualValue,
                source,
                weight);

            AnsiConsole.MarkupLine($"[green]✓[/] Feedback recorded for prediction [cyan]{Markup.Escape(predictionId)}[/]");
            AnsiConsole.MarkupLine($"  Actual value: [yellow]{Markup.Escape(actualValue)}[/]");
            if (source != null)
            {
                AnsiConsole.MarkupLine($"  Source: [grey]{Markup.Escape(source)}[/]");
            }
            if (weight is { } w)
            {
                AnsiConsole.MarkupLine($"  Weight: [grey]{w.ToString(CultureInfo.InvariantCulture)}[/]");
            }

            if (jsonOutput)
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    predictionId,
                    actualValue,
                    source,
                    weight,
                    warnings = JsonOutputScope.Warnings
                }, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                }));

            return 0;
        }
        catch (ArgumentOutOfRangeException ex) when (ex.ParamName == "weight")
        {
            ErrorConsole.Error(
                $"--weight {weight?.ToString(CultureInfo.InvariantCulture)} cannot weight a row.",
                "A weight is a finite number that is not negative — for a row reviewed with probability p, 1/p.");
            return 1;
        }
        catch (PredictionNotFoundException ex)
        {
            ErrorConsole.Error(
                Markup.Escape(ex.Message),
                "Prediction ids are in the ID column of `mloop logs`. Only logged predictions have one — "
                + "`mloop predict --log`, or any prediction served by `mloop serve`.");
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "feedback");
            return 1;
        }
    }

    /// <summary>The names of the columns a feedback file is read by, as given on the command line; null for the default.</summary>
    private sealed record FeedbackFileColumns(string? IdName, string? ActualName, string? SourceName, string? WeightName)
    {
        public const string Id = "prediction_id";
        public const string Actual = "actual";
        public const string Source = "source";
        public const string Weight = "weight";

        public bool AnyNamed => IdName is not null || ActualName is not null || SourceName is not null || WeightName is not null;
    }

    private static int ColumnIndex(string[] header, string name)
    {
        var wanted = CanonicalText.Nfc(name.Trim());
        for (var i = 0; i < header.Length; i++)
            if (string.Equals(CanonicalText.Nfc(header[i].Trim().TrimStart('\uFEFF')), wanted, StringComparison.OrdinalIgnoreCase))
                return i;
        return -1;
    }

    private static string Cell(string[] record, int index) => index >= 0 && index < record.Length ? record[index].Trim() : "";

    private static string Shown(string text) => text.Length > 16 ? text[..16] + "…" : text;

    /// <summary>
    /// Feedback for many predictions from one table: a row names a prediction by its id and carries the actual value. The columns are
    /// named, never guessed, and what a row cannot be recorded for is reported by its row number while the others are recorded.
    /// </summary>
    private static async Task<int> ExecuteAddFromFileAsync(
        string fromFile,
        string? defaultSource,
        double? defaultWeight,
        FeedbackFileColumns columns,
        bool jsonOutput)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var path = ProjectPaths.Resolve(fromFile, projectRoot);
            if (!TabularDataFile.Exists(path))
            {
                ErrorConsole.Error(
                    $"Feedback file not found: {fromFile}",
                    "Give the path of a table with one row per prediction — its prediction id and the actual value in columns.");
                return 1;
            }

            FeedbackEntry.ValidateWeight(defaultWeight);

            var records = TableRows.ReadRecords(await TabularDataFile.AsCsvAsync(path), includeHeader: true).ToList();
            if (records.Count == 0)
            {
                ErrorConsole.Error($"{fromFile} has no header row.", "Give a table whose first row names its columns.");
                return 1;
            }

            var header = records[0];
            var idName = columns.IdName ?? FeedbackFileColumns.Id;
            var actualName = columns.ActualName ?? FeedbackFileColumns.Actual;
            var idIndex = ColumnIndex(header, idName);
            var actualIndex = ColumnIndex(header, actualName);
            if (idIndex < 0 || actualIndex < 0)
            {
                var missing = idIndex < 0 ? idName : actualName;
                var option = idIndex < 0 ? "--id-column" : "--actual-column";
                var holds = idIndex < 0 ? "prediction id" : "actual value";
                ErrorConsole.Error(
                    $"{fromFile} has no column named '{missing}'.",
                    $"Its columns: {ColumnsOf(header)}. Name the column that holds the {holds} with {option}.");
                return 1;
            }

            if (idIndex == actualIndex)
            {
                ErrorConsole.Error(
                    $"The prediction id and the actual value are read from the same column ('{header[idIndex].Trim()}') of {fromFile}.",
                    "Name two different columns with --id-column and --actual-column.");
                return 1;
            }

            // The optional columns are read when the file has them; one the caller named and the file lacks is a mistake.
            int OptionalColumn(string? named, string fallback, string option)
            {
                var index = ColumnIndex(header, named ?? fallback);
                if (index < 0 && named is not null)
                    throw new FeedbackFileException(named, option);
                return index;
            }

            int sourceIndex, weightIndex;
            try
            {
                sourceIndex = OptionalColumn(columns.SourceName, FeedbackFileColumns.Source, "--source-column");
                weightIndex = OptionalColumn(columns.WeightName, FeedbackFileColumns.Weight, "--weight-column");
            }
            catch (FeedbackFileException ex)
            {
                ErrorConsole.Error(
                    $"{fromFile} has no column named '{ex.Column}'.",
                    $"Its columns: {ColumnsOf(header)}. Name the right one with {ex.Option}, or drop the option.");
                return 1;
            }

            if (records.Count == 1)
            {
                ErrorConsole.Error($"{fromFile} has a header and no rows.", "Add a row per prediction: its id and the actual value.");
                return 1;
            }

            var rejected = new List<FeedbackRejection>();
            var rows = new List<FeedbackRow>(records.Count - 1);
            for (var i = 1; i < records.Count; i++)
            {
                var record = records[i];
                var id = Cell(record, idIndex);
                var weight = defaultWeight;
                var weightCell = Cell(record, weightIndex);
                if (weightCell.Length > 0)
                {
                    if (!double.TryParse(weightCell, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
                    {
                        rejected.Add(new FeedbackRejection(i, id, $"the weight '{weightCell}' is not a number"));
                        continue;
                    }

                    weight = parsed;
                }

                var sourceCell = Cell(record, sourceIndex);
                rows.Add(new FeedbackRow(i, id, Cell(record, actualIndex), sourceCell.Length > 0 ? sourceCell : defaultSource, weight));
            }

            var result = await new FileFeedbackCollector(projectRoot).RecordFeedbackBatchAsync(rows);
            var allRejected = rejected.Concat(result.Rejected).OrderBy(r => r.Row).ToList();
            var total = records.Count - 1;

            if (result.Repeats.Count > 0)
            {
                var first = result.Repeats[0];
                WarningConsole.Warn(
                    $"{result.Repeats.Count} prediction(s) are named by more than one row; the last row for each stands "
                    + $"(first: {Shown(first.PredictionId)} in rows {string.Join(" and ", first.Rows)}).");
            }

            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    file = fromFile,
                    rows = total,
                    recorded = result.Recorded,
                    rejected = allRejected.Select(r => new { row = r.Row, predictionId = r.PredictionId, reason = r.Reason }),
                    repeated = result.Repeats.Select(r => new { predictionId = r.PredictionId, rows = r.Rows }),
                    warnings = JsonOutputScope.Warnings
                }, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                }));
            }
            else
            {
                AnsiConsole.MarkupLine($"[green]✓[/] Feedback recorded for [cyan]{result.Recorded}[/] of {total} row(s) of {Markup.Escape(fromFile)}");
                if (allRejected.Count > 0)
                {
                    var listed = string.Join('\n', allRejected.Take(10).Select(r => $"  row {r.Row}  {Shown(r.PredictionId)}  {r.Reason}"));
                    if (allRejected.Count > 10)
                        listed += $"\n  … and {allRejected.Count - 10} more — `--json` lists them all";
                    var notFound = allRejected.Any(r => r.Reason.StartsWith("no logged prediction", StringComparison.Ordinal));
                    ErrorConsole.Error(
                        $"{allRejected.Count} row(s) of {fromFile} were not recorded (rows count from 1 under the header):\n{listed}",
                        notFound
                            ? "Prediction ids are in the ID column of `mloop logs`. Only logged predictions have one — `mloop predict --log`, or any prediction served by `mloop serve`."
                            : "Fix those rows and run the file again — recording a prediction again corrects it, the latest entry standing.");
                }
            }

            return allRejected.Count == 0 ? 0 : 1;
        }
        catch (ArgumentOutOfRangeException ex) when (ex.ParamName == "weight")
        {
            ErrorConsole.Error(
                $"--weight {defaultWeight?.ToString(CultureInfo.InvariantCulture)} cannot weight a row.",
                "A weight is a finite number that is not negative — for a row reviewed with probability p, 1/p.");
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "feedback");
            return 1;
        }
    }

    private sealed class FeedbackFileException(string column, string option) : Exception($"no column named '{column}'")
    {
        public string Column { get; } = column;
        public string Option { get; } = option;
    }

    private static string ColumnsOf(string[] header) =>
        string.Join(", ", header.Take(12).Select(h => h.Trim())) + (header.Length > 12 ? ", …" : "");

    private static async Task<int> ExecuteListAsync(
        string modelName,
        int limit,
        DateTime? from,
        DateTime? to,
        bool jsonOutput)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            var (fromOffset, toOffset) = DateRangeFilter.ToFilterRange(from, to);

            var feedback = await collector.GetFeedbackAsync(
                MLoop.Core.Storage.ModelName.Require(modelName),
                fromOffset,
                toOffset,
                limit);

            // Empty is an ordinary state; --json still owes the caller a payload, not prose.
            if (!feedback.Any() && !jsonOutput)
            {
                AnsiConsole.MarkupLine($"[yellow]No feedback found for model '[cyan]{modelName}[/]'.[/]");
                AnsiConsole.MarkupLine("[grey]Use [blue]mloop feedback add[/] to record feedback.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputListAsJson(feedback);
            }
            else
            {
                OutputListAsTable(feedback, modelName);
            }

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "feedback");
            return 1;
        }
    }

    private static async Task<int> ExecuteMetricsAsync(
        string modelName,
        DateTime? from,
        DateTime? to,
        bool jsonOutput)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            var (fromOffset, toOffset) = DateRangeFilter.ToFilterRange(from, to);

            var metrics = await collector.CalculateMetricsAsync(
                MLoop.Core.Storage.ModelName.Require(modelName),
                fromOffset,
                toOffset);

            // Empty is an ordinary state; --json still owes the caller a payload, not prose.
            if (metrics.TotalFeedback == 0 && !jsonOutput)
            {
                AnsiConsole.MarkupLine($"[yellow]No feedback data found for model '[cyan]{modelName}[/]'.[/]");
                AnsiConsole.MarkupLine("[grey]Record feedback using [blue]mloop feedback add[/] first.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputMetricsAsJson(metrics);
            }
            else
            {
                OutputMetricsAsPanel(metrics);
            }

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "feedback");
            return 1;
        }
    }

    private static string? FindProjectRoot()
    {
        var fileSystem = new FileSystemManager();
        var projectDiscovery = new ProjectDiscovery(fileSystem);

        try
        {
            return projectDiscovery.FindRoot();
        }
        catch (InvalidOperationException)
        {
            ErrorConsole.Error(
                ProjectDiscovery.NotInsideProjectCause,
                ProjectDiscovery.NotInsideProjectGuidance);
            return null;
        }
    }

    private static void OutputListAsJson(IReadOnlyList<FeedbackEntry> feedback)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var output = feedback.Select(f => new
        {
            f.PredictionId,
            f.ModelName,
            f.PredictedValue,
            f.ActualValue,
            f.Source,
            f.Weight,
            f.ExperimentId,
            Timestamp = f.Timestamp.ToString("o")
        });

        Console.WriteLine(JsonSerializer.Serialize(output, options));
    }

    private static void OutputListAsTable(IReadOnlyList<FeedbackEntry> feedback, string modelName)
    {
        AnsiConsole.Write(new Rule($"[cyan]Feedback - {modelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn(TimestampDisplay.ZoneHeading("Timestamp"));
        table.AddColumn("Prediction ID");
        table.AddColumn("Predicted");
        table.AddColumn("Actual");
        table.AddColumn("Match");
        table.AddColumn("Source");
        var weighted = feedback.Any(e => e.Weight is not null);
        if (weighted)
            table.AddColumn(new TableColumn("Weight").RightAligned());

        foreach (var entry in feedback)
        {
            var predictedStr = entry.PredictedValue?.ToString() ?? "-";
            var actualStr = entry.ActualValue?.ToString() ?? "-";
            var isMatch = FeedbackValueComparer.ValuesMatch(entry.PredictedValue, entry.ActualValue);
            var matchStr = isMatch ? "[green]✓[/]" : "[red]✗[/]";

            // Values are the caller's text — escaped, or a '[' in a label is read as markup.
            var cells = new List<string>
            {
                TimestampDisplay.LocalWithoutZone(entry.Timestamp),
                Markup.Escape(entry.PredictionId.Length > 12 ? entry.PredictionId[..12] + "..." : entry.PredictionId),
                Markup.Escape(predictedStr.Length > 20 ? predictedStr[..20] + "..." : predictedStr),
                Markup.Escape(actualStr.Length > 20 ? actualStr[..20] + "..." : actualStr),
                matchStr,
                Markup.Escape(entry.Source ?? "-")
            };
            if (weighted)
                cells.Add(entry.EffectiveWeight.ToString("0.###", CultureInfo.CurrentCulture));
            table.AddRow(cells.ToArray());
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Showing {feedback.Count} entries[/]");
    }

    private static void OutputMetricsAsJson(FeedbackMetrics metrics)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            metrics.ModelName,
            metrics.TotalFeedback,
            metrics.Accuracy,
            metrics.Weighted,
            metrics.EffectiveSampleSize,
            metrics.ExperimentId,
            metrics.UnattributedFeedback,
            CalculatedAt = metrics.CalculatedAt.ToString("o")
        }, options));
    }

    private static void OutputMetricsAsPanel(FeedbackMetrics metrics)
    {
        AnsiConsole.Write(new Rule($"[cyan]Feedback Metrics - {metrics.ModelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var grid = new Grid();
        grid.AddColumn();
        grid.AddColumn();

        grid.AddRow("[grey]Total Feedback Entries:[/]", $"[white]{metrics.TotalFeedback}[/]");

        if (metrics.Accuracy.HasValue)
        {
            var accuracyColor = metrics.Accuracy.Value >= 0.9 ? "green"
                : metrics.Accuracy.Value >= 0.7 ? "yellow"
                : "red";
            grid.AddRow(metrics.Weighted ? "[grey]Accuracy (weighted):[/]" : "[grey]Accuracy:[/]",
                $"[{accuracyColor}]{metrics.Accuracy.Value:P2}[/]");
            if (metrics.Weighted && metrics.EffectiveSampleSize is { } effective)
                grid.AddRow("[grey]Effective rows:[/]", $"[white]{effective:0.#}[/] [grey](what the weighted accuracy is worth as evidence)[/]");
        }
        else
        {
            grid.AddRow("[grey]Accuracy:[/]", "[grey]N/A[/]");
        }

        grid.AddRow("[grey]Calculated At:[/]", $"[grey]{TimestampDisplay.Local(metrics.CalculatedAt)}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }
}
