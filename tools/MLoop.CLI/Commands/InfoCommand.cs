using System.CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
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
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);
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

            // Read file info
            var fileInfo = new FileInfo(dataFile);

            // Count lines with UTF-8 encoding
            int lineCount = 0;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                while (reader.ReadLine() != null)
                {
                    lineCount++;
                }
            }
            lineCount--; // Exclude header

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

            // Infer columns with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                AnsiConsole.MarkupLine("[red]File is empty[/]");
                return;
            }

            var columns = firstLine.Split(',');

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

            AnsiConsole.Write(new Rule("[yellow]Column Information[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var columnTable = new Table().Border(TableBorder.Rounded);
            columnTable.AddColumn("#");
            columnTable.AddColumn("Column Name");
            columnTable.AddColumn("Data Type");
            columnTable.AddColumn("Purpose");

            int colIndex = 1;
            foreach (var column in columns)
            {
                var purpose = GetColumnPurpose(column, columnInference.ColumnInformation);
                var dataType = InferDisplayType(column, columnInference);

                columnTable.AddRow(
                    colIndex.ToString(),
                    column,
                    dataType,
                    purpose);

                colIndex++;
            }

            AnsiConsole.Write(columnTable);
            AnsiConsole.WriteLine();

            // Load data for statistics
            try
            {
                var textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
                var dataView = textLoader.Load(dataFile);

                AnsiConsole.Write(new Rule("[yellow]Data Statistics[/]").LeftJustified());
                AnsiConsole.WriteLine();

                // Calculate null/missing percentages for each column
                var statsTable = new Table().Border(TableBorder.Rounded);
                statsTable.AddColumn("Column");
                statsTable.AddColumn("Missing Count");
                statsTable.AddColumn("Missing %");
                statsTable.AddColumn("Unique Values (sample)");

                foreach (var colName in columns.Take(10))
                {
                    var schema = dataView.Schema;
                    var colIndex2 = schema.GetColumnOrNull(colName);

                    if (colIndex2.HasValue)
                    {
                        // Count missing values (simplified)
                        long missingCount = CountMissingValues(dataView, colName);
                        double missingPercent = (missingCount / (double)lineCount) * 100;

                        // Sample unique values for categorical
                        int uniqueCount = CountUniqueValues(dataView, colName, maxSample: 1000);

                        statsTable.AddRow(
                            colName,
                            missingCount.ToString("N0"),
                            $"{missingPercent:F2}%",
                            uniqueCount > 0 ? $"~{uniqueCount}" : "N/A");
                    }
                }

                AnsiConsole.Write(statsTable);
                AnsiConsole.WriteLine();

                if (columns.Length > 10)
                {
                    AnsiConsole.MarkupLine("[grey]Showing first 10 columns. Total columns: {0}[/]",
                        columns.Length);
                    AnsiConsole.WriteLine();
                }
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[yellow]Could not load detailed statistics: {ex.Message}[/]");
            }
        });
    }

    private static string GetColumnPurpose(string columnName, Microsoft.ML.AutoML.ColumnInformation columnInfo)
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

        return "[grey]Feature[/]";
    }

    private static string InferDisplayType(string columnName, Microsoft.ML.AutoML.ColumnInferenceResults results)
    {
        var columnInfo = results.ColumnInformation;

        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "Text/Categorical";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "Numeric";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Text";

        return "Unknown";
    }

    private static long CountMissingValues(IDataView dataView, string columnName)
    {
        // Simplified: count rows where value is empty/null
        // In practice, ML.NET handles missing values differently per type
        long missingCount = 0;

        try
        {
            var schema = dataView.Schema;
            var colIndex = schema.GetColumnOrNull(columnName);

            if (!colIndex.HasValue)
                return 0;

            using (var cursor = dataView.GetRowCursor(dataView.Schema))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(colIndex.Value);
                var value = new ReadOnlyMemory<char>();

                while (cursor.MoveNext())
                {
                    try
                    {
                        getter(ref value);
                        if (value.IsEmpty || value.Length == 0)
                        {
                            missingCount++;
                        }
                    }
                    catch
                    {
                        missingCount++;
                    }
                }
            }
        }
        catch
        {
            // Column type not compatible with text getter
            return 0;
        }

        return missingCount;
    }

    private static int CountUniqueValues(IDataView dataView, string columnName, int maxSample)
    {
        var uniqueValues = new HashSet<string>();
        int sampledRows = 0;

        try
        {
            var schema = dataView.Schema;
            var colIndex = schema.GetColumnOrNull(columnName);

            if (!colIndex.HasValue)
                return 0;

            using (var cursor = dataView.GetRowCursor(dataView.Schema))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(colIndex.Value);
                var value = new ReadOnlyMemory<char>();

                while (cursor.MoveNext() && sampledRows < maxSample)
                {
                    try
                    {
                        getter(ref value);
                        if (!value.IsEmpty)
                        {
                            uniqueValues.Add(value.ToString());
                        }
                        sampledRows++;
                    }
                    catch
                    {
                        // Skip incompatible types
                        break;
                    }
                }
            }
        }
        catch
        {
            return 0;
        }

        return uniqueValues.Count;
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
