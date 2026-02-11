using System.CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop info - Display dataset profiling information
/// </summary>
public static class InfoCommand
{
    public static Command Create()
    {
        var dataFileArg = new Argument<string>("data-file")
        {
            Description = "Path to data file to analyze"
        };

        var labelOption = new Option<string?>("--label", "-l")
        {
            Description = "Label column name (overrides mloop.yaml setting)"
        };

        var nameOption = new Option<string>("--name", "-n")
        {
            Description = "Model name to read label configuration from mloop.yaml",
            DefaultValueFactory = _ => "default"
        };

        var command = new Command("info", "Display dataset profiling information");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(labelOption);
        command.Options.Add(nameOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            return ExecuteAsync(dataFile, label, modelName);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(string dataFile, string? labelOption, string modelName)
    {
        try
        {
            // Initialize components
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            // Find project root (optional for this command)
            string? projectRoot = null;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch
            {
                // Not in a project, that's ok for info command
            }

            // Resolve data file path
            string resolvedDataFile;
            if (projectRoot != null && !Path.IsPathRooted(dataFile))
            {
                resolvedDataFile = Path.Combine(projectRoot, dataFile);
            }
            else
            {
                resolvedDataFile = dataFile;
            }

            if (!File.Exists(resolvedDataFile))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] File not found: {resolvedDataFile}");
                return 1;
            }

            // Determine label column: 1) --label option, 2) mloop.yaml, 3) first column (fallback)
            string? labelColumn = labelOption;
            string? labelSource = labelOption != null ? "--label option" : null;

            if (labelColumn == null && projectRoot != null)
            {
                // Try to load label from mloop.yaml
                var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
                var config = await configLoader.LoadUserConfigAsync();

                if (config?.Models != null && config.Models.TryGetValue(modelName, out var modelDef))
                {
                    if (!string.IsNullOrEmpty(modelDef.Label))
                    {
                        labelColumn = modelDef.Label;
                        labelSource = $"mloop.yaml (model: {modelName})";
                    }
                }
            }

            AnsiConsole.MarkupLine($"[blue]Analyzing:[/] [cyan]{Path.GetFileName(resolvedDataFile)}[/]");
            if (labelSource != null)
            {
                AnsiConsole.MarkupLine($"[blue]Label column:[/] [green]{labelColumn}[/] (from {labelSource})");
            }
            AnsiConsole.WriteLine();

            // Profile the dataset
            await ProfileDatasetAsync(resolvedDataFile, labelColumn);

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "info");
            return 1;
        }
    }

    private static async Task ProfileDatasetAsync(string dataFile, string? labelColumn)
    {
        await Task.Run(() =>
        {
            var mlContext = new MLContext(seed: 42);

            // Detect and convert encoding to UTF-8 with BOM for ML.NET compatibility
            var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(dataFile);
            if (detection.WasConverted && detection.EncodingName != "UTF-8")
            {
                AnsiConsole.MarkupLine($"[green]Info:[/] Converted {detection.EncodingName} → UTF-8");
            }

            // Use converted path for all operations
            dataFile = convertedPath;

            // Read file info and count lines in a single pass
            var fileInfo = new FileInfo(dataFile);

            int lineCount = 0;
            string? firstLine = null;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine(); // Read header
                while (reader.ReadLine() != null)
                {
                    lineCount++;
                }
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                AnsiConsole.MarkupLine("[red]File is empty[/]");
                return;
            }

            AnsiConsole.Write(new Rule("[yellow]File Information[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var fileTable = new Table().Border(TableBorder.Rounded);
            fileTable.AddColumn("Property");
            fileTable.AddColumn("Value");

            fileTable.AddRow("File Size", FormatFileSize(fileInfo.Length));
            fileTable.AddRow("Rows (excluding header)", lineCount.ToString("N0"));
            fileTable.AddRow("Last Modified", fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"));

            AnsiConsole.Write(fileTable);
            AnsiConsole.WriteLine();

            var columns = MLoop.CLI.Infrastructure.ML.CsvFieldParser.ParseFields(firstLine);

            // Determine label for InferColumns:
            // 1) Use provided labelColumn if it exists in the file
            // 2) Fall back to first column
            string inferLabel;
            if (!string.IsNullOrEmpty(labelColumn) && columns.Contains(labelColumn))
            {
                inferLabel = labelColumn;
            }
            else
            {
                inferLabel = columns.Length > 0 ? columns[0] : "dummy";
                if (!string.IsNullOrEmpty(labelColumn) && !columns.Contains(labelColumn))
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] Label column '{labelColumn}' not found in file, using '{inferLabel}'");
                }
            }

            var columnInference = mlContext.Auto().InferColumns(
                dataFile,
                labelColumnName: inferLabel,
                separatorChar: ',');

            // Ensure RFC 4180 compliance: handle commas inside quoted fields
            columnInference.TextLoaderOptions.AllowQuoting = true;

            AnsiConsole.Write(new Rule("[yellow]Column Information[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var columnTable = new Table().Border(TableBorder.Rounded);
            columnTable.AddColumn("#");
            columnTable.AddColumn("Column Name");
            columnTable.AddColumn("Data Type");
            columnTable.AddColumn("Purpose");

            for (int colIdx = 0; colIdx < columns.Length; colIdx++)
            {
                var column = columns[colIdx];
                var dataType = InferDisplayType(column, columnInference, colIdx);
                var purpose = GetColumnPurpose(column, columnInference.ColumnInformation, dataType);

                columnTable.AddRow(
                    (colIdx + 1).ToString(),
                    column,
                    dataType,
                    purpose);
            }

            AnsiConsole.Write(columnTable);
            AnsiConsole.WriteLine();

            // Calculate statistics from raw CSV (avoids ML.NET DataView type compatibility issues)
            AnsiConsole.Write(new Rule("[yellow]Data Statistics[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var (columnStats, labelDistribution) = CalculateColumnStats(dataFile, columns, labelColumn);

            var statsTable = new Table().Border(TableBorder.Rounded);
            statsTable.AddColumn("Column");
            statsTable.AddColumn("Missing Count");
            statsTable.AddColumn("Missing %");
            statsTable.AddColumn("Unique Values (sample)");

            foreach (var colName in columns)
            {
                if (columnStats.TryGetValue(colName, out var stats))
                {
                    statsTable.AddRow(
                        colName,
                        stats.MissingCount.ToString("N0"),
                        $"{(stats.MissingCount / (double)lineCount) * 100:F2}%",
                        stats.UniqueCount > 0 ? $"~{stats.UniqueCount}" : "N/A");
                }
            }

            AnsiConsole.Write(statsTable);
            AnsiConsole.WriteLine();

            // Show label column class distribution
            if (labelDistribution != null && labelDistribution.Count > 0)
            {
                // IMP-8: Detect regression target (many unique numeric values) and show stats instead
                var nonEmptyClasses = labelDistribution.Where(p => p.Key != "(empty)").ToList();
                bool isLikelyRegression = nonEmptyClasses.Count > 20
                    && nonEmptyClasses.All(p => double.TryParse(p.Key, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out _));

                if (isLikelyRegression)
                {
                    AnsiConsole.Write(new Rule("[yellow]Label Statistics (Regression)[/]").LeftJustified());
                    AnsiConsole.WriteLine();

                    var values = nonEmptyClasses
                        .SelectMany(p => Enumerable.Repeat(double.Parse(p.Key, System.Globalization.CultureInfo.InvariantCulture), p.Value))
                        .OrderBy(v => v)
                        .ToList();

                    var mean = values.Average();
                    var stdDev = Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
                    var median = values.Count % 2 == 0
                        ? (values[values.Count / 2 - 1] + values[values.Count / 2]) / 2.0
                        : values[values.Count / 2];

                    var statsTable2 = new Table().Border(TableBorder.Rounded);
                    statsTable2.AddColumn("Statistic");
                    statsTable2.AddColumn("Value");

                    statsTable2.AddRow("Count", values.Count.ToString("N0"));
                    statsTable2.AddRow("Unique Values", nonEmptyClasses.Count.ToString("N0"));
                    statsTable2.AddRow("Min", values.First().ToString("F4"));
                    statsTable2.AddRow("Max", values.Last().ToString("F4"));
                    statsTable2.AddRow("Mean", mean.ToString("F4"));
                    statsTable2.AddRow("Median", median.ToString("F4"));
                    statsTable2.AddRow("Std Dev", stdDev.ToString("F4"));

                    AnsiConsole.Write(statsTable2);
                    AnsiConsole.WriteLine();

                    if (labelDistribution.ContainsKey("(empty)"))
                    {
                        var emptyCount = labelDistribution["(empty)"];
                        AnsiConsole.MarkupLine($"[yellow]Warning:[/] {emptyCount} rows have empty label values. These will be dropped during training.");
                        AnsiConsole.WriteLine();
                    }

                    AnsiConsole.MarkupLine("[green]Info:[/] Continuous label detected — suitable for [blue]regression[/] task.");
                    AnsiConsole.MarkupLine("[grey]  Tip: [blue]mloop init . --task regression[/][/]");
                    AnsiConsole.WriteLine();
                }
                else
                {
                AnsiConsole.Write(new Rule("[yellow]Label Distribution[/]").LeftJustified());
                AnsiConsole.WriteLine();

                var labelTable = new Table().Border(TableBorder.Rounded);
                labelTable.AddColumn("Class");
                labelTable.AddColumn("Count");
                labelTable.AddColumn("Percentage");

                foreach (var pair in labelDistribution.OrderByDescending(p => p.Value))
                {
                    var percent = (pair.Value / (double)lineCount) * 100;
                    labelTable.AddRow(pair.Key, pair.Value.ToString("N0"), $"{percent:F2}%");
                }

                AnsiConsole.Write(labelTable);
                AnsiConsole.WriteLine();

                // Imbalance warning for binary classification
                var realClasses = labelDistribution.Where(p => p.Key != "(empty)").OrderByDescending(p => p.Value).ToList();
                if (realClasses.Count == 2)
                {
                    var ratio = (double)realClasses[0].Value / realClasses[1].Value;
                    if (ratio > 10)
                    {
                        AnsiConsole.MarkupLine($"[yellow]Warning:[/] Severe class imbalance detected (ratio {ratio:F1}:1). Consider using [blue]--balance auto[/] during training.");
                    }
                    else if (ratio > 3)
                    {
                        AnsiConsole.MarkupLine($"[yellow]Warning:[/] Class imbalance detected (ratio {ratio:F1}:1). Consider using [blue]--balance auto[/] during training.");
                    }
                    AnsiConsole.WriteLine();
                }

                // IMP-7: Warn about empty label values
                if (labelDistribution.ContainsKey("(empty)"))
                {
                    var emptyCount = labelDistribution["(empty)"];
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] {emptyCount} rows have empty label values. These will be dropped during training.");
                    AnsiConsole.MarkupLine("[grey]  Tip: Clean data with [blue]mloop train --drop-missing-labels[/] (default for classification)[/]");
                    AnsiConsole.WriteLine();
                }
                } // end else (classification branch)
            }

            // BUG-8: Warn about potential date columns
            for (int ci = 0; ci < columns.Length; ci++)
            {
                var colName = columns[ci].ToLowerInvariant();
                if (colName.Contains("date") || colName.Contains("time") || colName.Contains("timestamp"))
                {
                    AnsiConsole.MarkupLine($"[yellow]Note:[/] Column '[cyan]{columns[ci]}[/]' appears to be a date/time column. ML.NET will treat it as text — consider excluding it if not relevant to the prediction task.");
                    AnsiConsole.WriteLine();
                    break; // Only warn once
                }
            }
        });
    }

    private static string GetColumnPurpose(string columnName, Microsoft.ML.AutoML.ColumnInformation columnInfo, string dataType)
    {
        if (columnInfo.LabelColumnName == columnName)
            return "[green]Label[/]";
        if (columnInfo.IgnoredColumnNames?.Contains(columnName) == true)
            return "[grey]Ignored[/]";
        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "[yellow]Categorical Feature[/]";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "[cyan]Numeric Feature[/]";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "[blue]Text Feature[/]";

        // Fall back: use inferred data type for purpose display
        return dataType switch
        {
            "Numeric" or "Integer" => "[cyan]Numeric Feature[/]",
            "Text" => "[blue]Text Feature[/]",
            "Boolean" => "[cyan]Numeric Feature[/]",
            _ => "[grey]Feature[/]"
        };
    }

    private static string InferDisplayType(string columnName, Microsoft.ML.AutoML.ColumnInferenceResults results, int csvColumnIndex)
    {
        var columnInfo = results.ColumnInformation;

        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "Text/Categorical";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "Numeric";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Text";

        // Fall back to TextLoaderOptions for columns not in ColumnInformation
        if (results.TextLoaderOptions?.Columns != null)
        {
            foreach (var col in results.TextLoaderOptions.Columns)
            {
                bool matched = false;

                // Match by name
                if (col.Name == columnName)
                {
                    matched = true;
                }
                // Match by source column index
                else if (col.Source != null)
                {
                    foreach (var range in col.Source)
                    {
                        if (csvColumnIndex >= range.Min && csvColumnIndex <= (range.Max ?? range.Min))
                        {
                            matched = true;
                            break;
                        }
                    }
                }

                if (matched)
                {
                    return col.DataKind switch
                    {
                        DataKind.Single or DataKind.Double => "Numeric",
                        DataKind.Int32 or DataKind.Int64 or DataKind.UInt32 or DataKind.UInt64 => "Integer",
                        DataKind.String => "Text",
                        DataKind.Boolean => "Boolean",
                        _ => col.DataKind.ToString()
                    };
                }
            }
        }

        return "Unknown";
    }

    private record ColumnStatInfo(long MissingCount, int UniqueCount);

    private static (Dictionary<string, ColumnStatInfo> Stats, Dictionary<string, int>? LabelDistribution)
        CalculateColumnStats(string dataFile, string[] columns, string? labelColumn, int maxUniqueRows = 10000)
    {
        var missingCounts = new long[columns.Length];
        var uniqueSets = new HashSet<string>[columns.Length];
        for (int i = 0; i < columns.Length; i++)
            uniqueSets[i] = new HashSet<string>();

        int labelIndex = labelColumn != null ? Array.IndexOf(columns, labelColumn) : -1;
        Dictionary<string, int>? labelDistribution = labelIndex >= 0 ? new Dictionary<string, int>() : null;

        int rowCount = 0;
        using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
        {
            reader.ReadLine(); // skip header
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                rowCount++;
                var fields = CsvFieldParser.ParseFields(line);
                for (int i = 0; i < Math.Min(fields.Length, columns.Length); i++)
                {
                    if (string.IsNullOrWhiteSpace(fields[i]))
                        missingCounts[i]++;
                    if (rowCount <= maxUniqueRows && !string.IsNullOrWhiteSpace(fields[i]))
                        uniqueSets[i].Add(fields[i]);
                }

                // Label distribution (all rows)
                if (labelDistribution != null && labelIndex >= 0 && labelIndex < fields.Length)
                {
                    var value = fields[labelIndex];
                    if (string.IsNullOrWhiteSpace(value)) value = "(empty)";
                    labelDistribution.TryGetValue(value, out var count);
                    labelDistribution[value] = count + 1;
                }
            }
        }

        var result = new Dictionary<string, ColumnStatInfo>();
        for (int i = 0; i < columns.Length; i++)
        {
            result[columns[i]] = new ColumnStatInfo(missingCounts[i], uniqueSets[i].Count);
        }

        return (result, labelDistribution);
    }

    private static string FormatFileSize(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB", "TB" };
        double len = bytes;
        int order = 0;

        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len = len / 1024;
        }

        return $"{len:0.##} {sizes[order]}";
    }
}
