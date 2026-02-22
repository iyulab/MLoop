using System.CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using DataLens;
using DataLens.Models;
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

        var analyzeOption = new Option<bool>("--analyze", "-a")
        {
            Description = "Run deep analysis: correlation, feature importance, distribution, anomaly detection"
        };

        var sampleSizeOption = new Option<int>("--sample-size")
        {
            Description = "Maximum rows to sample for deep analysis",
            DefaultValueFactory = _ => 50_000
        };

        var command = new Command("info", "Display dataset profiling information");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(labelOption);
        command.Options.Add(nameOption);
        command.Options.Add(analyzeOption);
        command.Options.Add(sampleSizeOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var analyze = parseResult.GetValue(analyzeOption);
            var sampleSize = parseResult.GetValue(sampleSizeOption);
            return ExecuteAsync(dataFile, label, modelName, analyze, sampleSize);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string dataFile, string? labelOption, string modelName,
        bool analyze, int sampleSize)
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
            await ProfileDatasetAsync(resolvedDataFile, labelColumn, analyze, sampleSize);

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "info");
            return 1;
        }
    }

    private static async Task ProfileDatasetAsync(
        string dataFile, string? labelColumn, bool analyze, int sampleSize)
    {
        var mlContext = new MLContext(seed: 42);

        // Keep original path for DataLens (which requires .csv extension via CsvBridge)
        var originalDataFile = dataFile;

        // Detect and convert encoding to UTF-8 with BOM for ML.NET compatibility
        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(dataFile);
        if (detection.WasConverted && detection.EncodingName != "UTF-8")
        {
            AnsiConsole.MarkupLine($"[green]Info:[/] Converted {detection.EncodingName} -> UTF-8");
        }

        // Use converted path for ML.NET operations
        dataFile = convertedPath;

        // Flatten multi-line quoted headers (ML.NET doesn't support them)
        dataFile = CsvDataLoader.FlattenMultiLineHeaders(dataFile);

        // Remove unnamed/pandas index columns (matches CsvDataLoader.LoadData behavior)
        var preIndexPath = dataFile;
        dataFile = CsvDataLoader.RemoveIndexColumns(dataFile);
        if (dataFile != preIndexPath)
        {
            AnsiConsole.MarkupLine("[green]Info:[/] Removed unnamed index column(s) (pandas artifact)");
        }

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

        // 1. File Information
        InfoPresenter.DisplayFileInfo(
            Path.GetFileName(dataFile), fileInfo.Length, lineCount, fileInfo.LastWriteTime);

        var columns = CsvFieldParser.ParseFields(firstLine);

        // Determine label for InferColumns
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

        // Ensure RFC 4180 compliance
        columnInference.TextLoaderOptions.AllowQuoting = true;

        // 2. Column Information
        InfoPresenter.DisplayColumnInfo(
            columns,
            (colName, colIdx) => InferDisplayType(colName, columnInference, colIdx),
            (colName, dataType) => GetColumnPurpose(colName, columnInference.ColumnInformation, dataType));

        // 3. DataLens Profile (always-on, nullable)
        // DataLens uses its own CSV loading (FilePrepper CsvBridge) which requires .csv extension,
        // so we pass the original file path instead of the ML.NET-converted .tmp path.
        var dataLens = new DataLensAnalyzer();
        ProfileReport? profile = null;

        if (dataLens.IsAvailable)
        {
            // Size gate: skip profiling for files > 200MB (unless --analyze forces deep analysis)
            if (fileInfo.Length <= 200 * 1024 * 1024 || analyze)
            {
                profile = await dataLens.ProfileAsync(originalDataFile);
            }

            if (dataLens.Version != null)
            {
                AnsiConsole.MarkupLine($"[grey]DataLens {dataLens.Version} active[/]");
                AnsiConsole.WriteLine();
            }
        }

        // 4. Calculate column stats from raw CSV
        var (columnStats, labelDistribution) = CalculateColumnStats(dataFile, columns, labelColumn);

        // Convert to the format InfoPresenter expects
        var statsDict = columnStats.ToDictionary(
            kvp => kvp.Key,
            kvp => (kvp.Value.MissingCount, kvp.Value.UniqueCount));

        // 5. Display enhanced stats (with profile data if available)
        InfoPresenter.DisplayDataStatistics(columns, statsDict, lineCount, profile);

        // 6. Label distribution
        if (labelDistribution != null && labelDistribution.Count > 0)
        {
            InfoPresenter.DisplayLabelDistribution(labelDistribution, lineCount);
        }

        // 7. Date column warning (uses shared DateTimeDetector for consistent detection)
        for (int ci = 0; ci < columns.Length; ci++)
        {
            if (DateTimeDetector.IsDateTimeColumnName(columns[ci]))
            {
                AnsiConsole.MarkupLine($"[yellow]Note:[/] Column '[cyan]{columns[ci]}[/]' appears to be a date/time column. ML.NET will treat it as text -- consider excluding it if not relevant to the prediction task.");
                AnsiConsole.WriteLine();
                break;
            }
        }

        // 8. Deep analysis (--analyze)
        if (analyze)
        {
            await RunDeepAnalysisAsync(dataLens, originalDataFile, labelColumn);
        }
    }

    private static async Task RunDeepAnalysisAsync(
        DataLensAnalyzer dataLens, string dataFile, string? labelColumn)
    {
        if (!dataLens.IsAvailable)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --analyze requires DataLens library. Skipping deep analysis.");
            AnsiConsole.MarkupLine("[grey]  Install DataLens NuGet package to enable.[/]");
            AnsiConsole.WriteLine();
            return;
        }

        AnsiConsole.Write(new Rule("[blue]Deep Analysis (DataLens)[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var options = new AnalysisOptions
        {
            IncludeProfiling = false, // Already done above
            IncludeDescriptive = true,
            IncludeCorrelation = true,
            IncludeDistribution = true,
            IncludeOutliers = true,
            IncludeFeatures = !string.IsNullOrEmpty(labelColumn),
            IncludeRegression = false,
            IncludeClustering = false,
            IncludePca = false,
            TargetColumn = labelColumn
        };

        var result = await dataLens.AnalyzeAsync(dataFile, options);
        if (result == null)
        {
            AnsiConsole.MarkupLine("[yellow]Deep analysis returned no results.[/]");
            AnsiConsole.WriteLine();
            return;
        }

        // a. Descriptive Statistics (quartiles, skewness, kurtosis)
        if (result.Descriptive?.Columns is { Count: > 0 })
        {
            InfoPresenter.DisplayDescriptiveStatistics(result.Descriptive);
        }

        // b. Correlation
        if (result.Correlation != null)
        {
            InfoPresenter.DisplayCorrelation(result.Correlation);
        }

        // c. Feature Importance
        if (result.Features?.Importance != null)
        {
            InfoPresenter.DisplayFeatureImportance(result.Features.Importance);
        }

        // d. Distribution
        if (result.Distribution?.Columns is { Count: > 0 })
        {
            InfoPresenter.DisplayDistributions(result.Distribution);
        }

        // e. Outlier detection
        if (result.Outliers != null)
        {
            InfoPresenter.DisplayOutlierSummary(result.Outliers);
        }
    }

    private static string GetColumnPurpose(string columnName, ColumnInformation columnInfo, string dataType)
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

        return dataType switch
        {
            "Numeric" or "Integer" => "[cyan]Numeric Feature[/]",
            "Text" => "[blue]Text Feature[/]",
            "Boolean" => "[cyan]Numeric Feature[/]",
            _ => "[grey]Feature[/]"
        };
    }

    private static string InferDisplayType(string columnName, ColumnInferenceResults results, int csvColumnIndex)
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

                if (col.Name == columnName)
                {
                    matched = true;
                }
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
}
