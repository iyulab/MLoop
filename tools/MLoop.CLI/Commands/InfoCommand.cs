using System.CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
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

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the profiling result as JSON to stdout instead of the human report. " +
                          "Progress and narration go to stderr."
        };

        var command = new Command("info", "Display dataset profiling information");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(labelOption);
        command.Options.Add(nameOption);
        command.Options.Add(analyzeOption);
        command.Options.Add(sampleSizeOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg)!;
            var label = parseResult.GetValue(labelOption);
            var modelName = parseResult.GetValue(nameOption)!;
            var analyze = parseResult.GetValue(analyzeOption);
            var sampleSize = parseResult.GetValue(sampleSizeOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(dataFile, label, modelName, analyze, sampleSize, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string dataFile, string? labelOption, string modelName,
        bool analyze, int sampleSize, bool jsonOutput = false)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

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
                ErrorConsole.Error(
                    $"File not found: {resolvedDataFile}",
                    projectRoot != null ? ErrorConsole.PathNotFoundTip(projectRoot) : ErrorConsole.PathNotFoundTipCwd());
                EmitJson(resolvedDataFile, null, null, analyze, null, null, null, null, null,
                    [$"File not found: {resolvedDataFile}"], jsonOutput);
                return 1;
            }

            // Determine label column: 1) --label option, 2) mloop.yaml, 3) first column (fallback)
            string? labelColumn = labelOption;
            string? labelSource = labelOption != null ? "--label option" : null;
            MLoopConfig? config = null;

            if (projectRoot != null)
            {
                // Try to load config from mloop.yaml
                var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
                config = await configLoader.LoadUserConfigAsync();
            }

            if (labelColumn == null && config?.Models != null &&
                config.Models.TryGetValue(modelName, out var modelDef))
            {
                if (!string.IsNullOrEmpty(modelDef.Label))
                {
                    labelColumn = modelDef.Label;
                    labelSource = $"mloop.yaml (model: {modelName})";
                }
            }

            AnsiConsole.MarkupLine($"[blue]Analyzing:[/] [cyan]{Path.GetFileName(resolvedDataFile)}[/]");
            if (labelSource != null)
            {
                AnsiConsole.MarkupLine($"[blue]Label column:[/] [green]{labelColumn}[/] (from {labelSource})");
            }
            AnsiConsole.WriteLine();

            // Get column overrides from config
            Dictionary<string, string>? columnOverrides = null;
            if (config?.Models != null && config.Models.TryGetValue(modelName, out var modelDefForOverrides))
            {
                columnOverrides = modelDefForOverrides.Columns?.ToDictionary(
                    kvp => kvp.Key, kvp => kvp.Value.Type);
            }

            // Profile the dataset
            return await ProfileDatasetAsync(resolvedDataFile, labelColumn, labelSource, analyze, sampleSize, columnOverrides, jsonOutput);
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "info");
            return 1;
        }
    }

    private static async Task<int> ProfileDatasetAsync(
        string dataFile, string? labelColumn, string? labelSource, bool analyze, int sampleSize,
        Dictionary<string, string>? columnOverrides = null, bool jsonOutput = false)
    {
        var reportedDataFile = dataFile;
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

        // Flatten multi-line quoted fields in data rows (RFC 4180 multiline support)
        dataFile = CsvDataLoader.FlattenMultiLineQuotedFields(dataFile, CoreNarration.Sink);

        // Flatten multi-line quoted headers (ML.NET doesn't support them)
        dataFile = CsvDataLoader.FlattenMultiLineHeaders(dataFile, CoreNarration.Sink);

        // Remove unnamed/pandas index columns (matches CsvDataLoader.LoadData behavior)
        var preIndexPath = dataFile;
        dataFile = CsvDataLoader.RemoveIndexColumns(dataFile, CoreNarration.Sink);
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
            EmitJson(reportedDataFile, labelColumn, labelSource, analyze, null, null, null, null, null,
                ["File is empty"], jsonOutput);
            return 1;
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
            if (!string.IsNullOrEmpty(labelColumn) && !columns.Contains(labelColumn))
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] Label column '{labelColumn}' not found in file");
            }

            // Heuristic: prefer common label column names (case-insensitive)
            var commonLabelNames = new[] { "label", "target", "class", "category", "y", "output" };
            var matched = columns.FirstOrDefault(c =>
                commonLabelNames.Contains(c, StringComparer.OrdinalIgnoreCase));

            if (matched != null)
            {
                inferLabel = matched;
            }
            else
            {
                // Fall back to last column (ML convention: label is typically the last column)
                inferLabel = columns.Length > 0 ? columns[^1] : "dummy";
            }
        }

        var columnInference = mlContext.Auto().InferColumns(
            dataFile,
            labelColumnName: inferLabel,
            separatorChar: ',');

        // Ensure RFC 4180 compliance
        columnInference.TextLoaderOptions.AllowQuoting = true;

        // Sample data lines for text-likeness analysis (reuses TrainingEngine logic)
        var sampleLines = ReadSampleLines(dataFile, 200);

        // 2. Column Information
        InfoPresenter.DisplayColumnInfo(
            columns,
            (colName, colIdx) => InferDisplayType(colName, columnInference, colIdx, sampleLines),
            (colName, dataType) => GetColumnPurpose(colName, columnInference.ColumnInformation, dataType),
            columnOverrides);

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

        // Same per-column classification InfoPresenter.DisplayColumnInfo just rendered, built again
        // here as plain data for --json (the lambdas are pure and cheap per column; keeping the
        // display call's own signature untouched avoids threading a view model through it).
        var columnRows = new List<ColumnInfoRow>();
        for (int ci = 0; ci < columns.Length; ci++)
        {
            var dataType = InferDisplayType(columns[ci], columnInference, ci, sampleLines);
            var purpose = GetColumnPurpose(columns[ci], columnInference.ColumnInformation, dataType);
            string? overrideType = columnOverrides != null && columnOverrides.TryGetValue(columns[ci], out var ot)
                ? ot
                : null;
            var stat = columnStats.TryGetValue(columns[ci], out var s) ? s : new ColumnStatInfo(0, 0);
            columnRows.Add(new ColumnInfoRow(columns[ci], dataType, purpose, overrideType, stat.MissingCount, stat.UniqueCount));
        }

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
        AnalysisResult? analysisResult = null;
        if (analyze)
        {
            analysisResult = await RunDeepAnalysisAsync(dataLens, originalDataFile, labelColumn);
        }

        EmitJson(
            reportedDataFile, labelColumn, labelSource, analyze,
            new FileInfoRow(Path.GetFileName(dataFile), fileInfo.Length, lineCount, fileInfo.LastWriteTime),
            columnRows, labelDistribution, profile, analysisResult, null, jsonOutput);

        return 0;
    }

    private static async Task<AnalysisResult?> RunDeepAnalysisAsync(
        DataLensAnalyzer dataLens, string dataFile, string? labelColumn)
    {
        if (!dataLens.IsAvailable)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --analyze requires DataLens library. Skipping deep analysis.");
            AnsiConsole.MarkupLine("[grey]  Install DataLens NuGet package to enable.[/]");
            AnsiConsole.WriteLine();
            return null;
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
            return null;
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

        return result;
    }

    /// <summary>
    /// Plain semantic classification of a column's role (Label / Ignored / Text Feature / ...).
    /// Display-only coloring is applied by <see cref="InfoPresenter"/>, not here, so this value is
    /// also what the <c>--json</c> payload reports for each column.
    /// </summary>
    internal static string GetColumnPurpose(string columnName, ColumnInformation columnInfo, string dataType)
    {
        if (columnInfo.LabelColumnName == columnName)
            return "Label";
        if (columnInfo.IgnoredColumnNames?.Contains(columnName) == true)
            return "Ignored";

        // Use dataType (which incorporates text-likeness reclassification) over raw ML.NET inference
        if (dataType == "Text")
            return "Text Feature";
        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "Categorical Feature";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "Numeric Feature";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Text Feature";

        return dataType switch
        {
            "Numeric" or "Integer" => "Numeric Feature",
            "Boolean" => "Numeric Feature",
            _ => "Feature"
        };
    }

    private static string InferDisplayType(string columnName, ColumnInferenceResults results, int csvColumnIndex, string[] sampleLines)
    {
        var columnInfo = results.ColumnInformation;

        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
        {
            // Apply text-likeness check: reclassify categorical columns that look like text
            var uniqueValues = new HashSet<string>();
            foreach (var line in sampleLines)
            {
                var fields = CsvFieldParser.ParseFields(line);
                if (csvColumnIndex < fields.Length)
                {
                    var value = fields[csvColumnIndex].Trim();
                    if (!string.IsNullOrEmpty(value))
                        uniqueValues.Add(value);
                }
            }
            if (TrainingEngine.LooksLikeText(csvColumnIndex, sampleLines, uniqueValues.Count))
                return "Text";
            return "Text/Categorical";
        }
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

    private static string[] ReadSampleLines(string dataFile, int maxLines)
    {
        var lines = new List<string>();
        using var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        reader.ReadLine(); // skip header
        string? line;
        while ((line = reader.ReadLine()) != null && lines.Count < maxLines)
        {
            lines.Add(line);
        }
        return lines.ToArray();
    }

    internal record ColumnStatInfo(long MissingCount, int UniqueCount);

    internal static (Dictionary<string, ColumnStatInfo> Stats, Dictionary<string, int>? LabelDistribution)
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

    internal record FileInfoRow(string FileName, long SizeBytes, int LineCount, DateTime LastModified);

    internal record ColumnInfoRow(
        string Name, string DataType, string Purpose, string? Override, long MissingCount, int UniqueCount);

    private static void EmitJson(
        string dataFile,
        string? labelColumn,
        string? labelSource,
        bool analyze,
        FileInfoRow? fileInfo,
        List<ColumnInfoRow>? columns,
        Dictionary<string, int>? labelDistribution,
        ProfileReport? profile,
        AnalysisResult? analysis,
        List<string>? errors,
        bool jsonOutput)
    {
        if (!jsonOutput)
            return;

        var payload = new
        {
            DataFile = dataFile,
            LabelColumn = labelColumn,
            LabelSource = labelSource,
            Analyze = analyze,
            FileInfo = fileInfo,
            Columns = columns,
            LabelDistribution = labelDistribution,
            Profile = profile,
            Analysis = analysis,
            Errors = errors
        };
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(payload, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
            // Descriptive stats (e.g. kurtosis on a near-constant or tiny sample) and DataLens's own
            // profile numerics can legitimately be NaN/Infinity — the default serializer throws
            // rather than emit them. This writes them as the quoted strings "NaN"/"Infinity"/
            // "-Infinity" instead, matching the values themselves being non-numbers.
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
            Converters = { new Double2DArrayJsonConverter() }
        }));
    }

    /// <summary>
    /// <c>System.Text.Json</c> has no built-in support for multi-dimensional arrays (thrown as
    /// "Serialization and deserialization of 'System.Double[,]' instances is not supported") —
    /// DataLens's <c>CorrelationReport.Matrix</c> and <c>PcaReport.Loadings</c> are both <see
    /// cref="double"/>[,]. Writes rows as nested JSON arrays; a general fix rather than reshaping
    /// each report type individually, since any future DataLens report field of this shape would
    /// hit the same gap.
    /// </summary>
    internal sealed class Double2DArrayJsonConverter : System.Text.Json.Serialization.JsonConverter<double[,]>
    {
        public override double[,] Read(
            ref System.Text.Json.Utf8JsonReader reader, Type typeToConvert, System.Text.Json.JsonSerializerOptions options)
            => throw new NotSupportedException("info --json output is not deserialized back into this type.");

        public override void Write(System.Text.Json.Utf8JsonWriter writer, double[,] value, System.Text.Json.JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            for (int r = 0; r < value.GetLength(0); r++)
            {
                writer.WriteStartArray();
                for (int c = 0; c < value.GetLength(1); c++)
                    WriteDouble(writer, value[r, c]);
                writer.WriteEndArray();
            }
            writer.WriteEndArray();
        }

        // A zero-variance column makes Pearson correlation 0/0 = NaN, so this is reachable on
        // ordinary data, not just constructed edge cases. Matches the scalar-double path elsewhere
        // in this payload (JsonNumberHandling.AllowNamedFloatingPointLiterals), which writes these
        // as the quoted strings "NaN"/"Infinity"/"-Infinity" — bare, unquoted NaN is not valid JSON.
        private static void WriteDouble(System.Text.Json.Utf8JsonWriter writer, double value)
        {
            if (double.IsNaN(value)) writer.WriteStringValue("NaN");
            else if (double.IsPositiveInfinity(value)) writer.WriteStringValue("Infinity");
            else if (double.IsNegativeInfinity(value)) writer.WriteStringValue("-Infinity");
            else writer.WriteNumberValue(value);
        }
    }
}
