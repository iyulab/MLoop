using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Data;

/// <summary>
/// Implementation of CSV merger for combining multiple CSV files with same schema.
/// Supports automatic detection of mergeable files and pattern recognition.
/// </summary>
public class CsvMerger : ICsvMerger
{
    private readonly ICsvHelper _csvHelper;

    // Patterns for filename-based merge detection
    private static readonly Regex NormalOutlierPattern = new(
        @"(normal|outlier|anomal|fault|defect|good|bad)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex DatePattern = new(
        @"(\d{4}[-._]\d{2}[-._]\d{2}|\d{2}[-._]\d{2}[-._]\d{4})",
        RegexOptions.Compiled);

    private static readonly Regex SequencePattern = new(
        @"[_-](\d+)[_.]",
        RegexOptions.Compiled);

    public CsvMerger(ICsvHelper csvHelper)
    {
        _csvHelper = csvHelper ?? throw new ArgumentNullException(nameof(csvHelper));
    }

    /// <inheritdoc />
    public async Task<List<CsvMergeGroup>> DiscoverMergeableCsvsAsync(
        string directory,
        CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Directory not found: {directory}");
        }

        var csvFiles = Directory.GetFiles(directory, "*.csv", SearchOption.TopDirectoryOnly)
            .Where(f => !Path.GetFileName(f).Equals("train.csv", StringComparison.OrdinalIgnoreCase))
            .Where(f => !Path.GetFileName(f).Equals("validation.csv", StringComparison.OrdinalIgnoreCase))
            .Where(f => !Path.GetFileName(f).Equals("test.csv", StringComparison.OrdinalIgnoreCase))
            .Where(f => !Path.GetFileName(f).Equals("predict.csv", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (csvFiles.Count < 2)
        {
            return new List<CsvMergeGroup>();
        }

        // Group by schema
        var schemaGroups = new Dictionary<string, List<(string Path, List<string> Headers)>>();

        foreach (var file in csvFiles)
        {
            try
            {
                var headers = await _csvHelper.ReadHeadersAsync(file, cancellationToken: cancellationToken).ConfigureAwait(false);
                if (headers.Count == 0) continue;

                var schemaId = ComputeSchemaId(headers);

                if (!schemaGroups.ContainsKey(schemaId))
                {
                    schemaGroups[schemaId] = new List<(string, List<string>)>();
                }
                schemaGroups[schemaId].Add((file, headers));
            }
            catch
            {
                // Skip files that can't be read
                continue;
            }
        }

        // Convert to CsvMergeGroup and detect patterns
        var result = new List<CsvMergeGroup>();

        foreach (var (schemaId, files) in schemaGroups)
        {
            if (files.Count < 2) continue;

            var (pattern, confidence) = DetectMergePattern(files.Select(f => f.Path).ToList());

            result.Add(new CsvMergeGroup
            {
                SchemaId = schemaId,
                Columns = files.First().Headers,
                FilePaths = files.Select(f => f.Path).OrderBy(f => f).ToList(),
                DetectedPattern = pattern,
                PatternConfidence = confidence
            });
        }

        return result.OrderByDescending(g => g.FilePaths.Count).ToList();
    }

    /// <inheritdoc />
    public async Task<CsvMergeResult> MergeAsync(
        IEnumerable<string> sourcePaths,
        string outputPath,
        IReadOnlyDictionary<string, string>? columnDefaults = null,
        CancellationToken cancellationToken = default)
    {
        var paths = sourcePaths.ToList();
        if (paths.Count == 0)
        {
            return new CsvMergeResult
            {
                Success = false,
                Error = "No source files provided"
            };
        }

        try
        {
            // Validate schema compatibility first
            var validation = await ValidateSchemaCompatibilityAsync(paths, columnDefaults, cancellationToken).ConfigureAwait(false);
            if (!validation.IsCompatible)
            {
                return new CsvMergeResult
                {
                    Success = false,
                    Error = validation.Message ?? "Schema mismatch between files"
                };
            }

            // Every file has the same columns (validated above); the output takes the first file's order, with the
            // defaults it lacks after them.
            var firstHeaders = await _csvHelper.ReadHeadersAsync(paths[0], cancellationToken: cancellationToken).ConfigureAwait(false);
            var columns = firstHeaders.Concat(MissingDefaults(firstHeaders, columnDefaults).Select(kv => kv.Key)).ToList();
            var (rowsPerFile, totalRows) = await WriteMergedAsync(paths, outputPath, columns,
                (_, headers) => Projection(columns, headers, MissingDefaults(headers, columnDefaults)), cancellationToken).ConfigureAwait(false);

            if (totalRows == 0)
            {
                return new CsvMergeResult
                {
                    Success = false,
                    Error = "No data found in source files"
                };
            }

            return new CsvMergeResult
            {
                Success = true,
                OutputPath = Path.GetFullPath(outputPath),
                TotalRows = totalRows,
                SourceFileCount = paths.Count,
                RowsPerFile = rowsPerFile
            };
        }
        catch (Exception ex)
        {
            return new CsvMergeResult
            {
                Success = false,
                Error = $"Merge failed: {ex.Message}"
            };
        }
    }

    /// <inheritdoc />
    public async Task<CsvSchemaValidation> ValidateSchemaCompatibilityAsync(
        IEnumerable<string> paths,
        IReadOnlyDictionary<string, string>? columnDefaults = null,
        CancellationToken cancellationToken = default)
    {
        var pathList = paths.ToList();
        if (pathList.Count == 0)
        {
            return new CsvSchemaValidation
            {
                IsCompatible = false,
                Message = "No files provided for validation"
            };
        }

        var allHeaders = new Dictionary<string, List<string>>();
        var unreadable = new Dictionary<string, string>();

        foreach (var path in pathList)
        {
            try
            {
                var headers = await _csvHelper.ReadHeadersAsync(path, cancellationToken: cancellationToken).ConfigureAwait(false);
                // A column with a default is one every file has: those that lack it will be given it.
                allHeaders[path] = headers.Concat(MissingDefaults(headers, columnDefaults).Select(kv => kv.Key)).ToList();
            }
            catch (Exception ex)
            {
                unreadable[path] = ex.Message;
            }
        }

        if (allHeaders.Count == 0)
        {
            return new CsvSchemaValidation
            {
                IsCompatible = false,
                UnreadableFiles = unreadable,
                Message = "No files could be read"
            };
        }

        // Find common columns
        var commonColumns = allHeaders.Values.First().ToHashSet();
        foreach (var headers in allHeaders.Values.Skip(1))
        {
            commonColumns.IntersectWith(headers);
        }

        // Find mismatched columns
        var mismatched = new Dictionary<string, List<string>>();
        foreach (var (path, headers) in allHeaders)
        {
            var extra = headers.Except(commonColumns).ToList();
            if (extra.Count > 0)
            {
                mismatched[Path.GetFileName(path)] = extra;
            }
        }

        var isCompatible = commonColumns.Count > 0 && mismatched.Count == 0;

        return new CsvSchemaValidation
        {
            IsCompatible = isCompatible,
            CommonColumns = commonColumns.ToList(),
            MismatchedColumns = mismatched,
            UnreadableFiles = unreadable,
            Message = isCompatible
                ? $"All {allHeaders.Count} files have compatible schemas with {commonColumns.Count} columns"
                : $"Schema mismatch: {mismatched.Count} files have different columns"
        };
    }

    private static IEnumerable<KeyValuePair<string, string>> MissingDefaults(
        IEnumerable<string> headers, IReadOnlyDictionary<string, string>? columnDefaults) =>
        columnDefaults is null
            ? []
            : columnDefaults.Where(kv => !headers.Contains(kv.Key, StringComparer.OrdinalIgnoreCase));

    /// <summary>
    /// For one file: where each output column's value comes from — the file's own column, or a constant (a default
    /// the file lacks, or filename metadata). A constant wins over a column of the same name, as the merges always had it.
    /// </summary>
    private static Func<string[], string>[] Projection(
        IReadOnlyList<string> columns, IReadOnlyList<string> headers, IEnumerable<KeyValuePair<string, string>> constants)
    {
        var fixedValues = new Dictionary<string, string>();
        foreach (var (column, value) in constants)
            fixedValues[column] = value;
        return [.. columns.Select<string, Func<string[], string>>(column =>
        {
            if (fixedValues.TryGetValue(column, out var value))
                return _ => value;
            var index = IndexOf(headers, column);
            return index < 0 ? _ => string.Empty : row => row[index];
        })];
    }

    /// <summary>
    /// Streams every file's rows, projected onto <paramref name="columns"/>, into one file — written beside
    /// <paramref name="outputPath"/> and moved into place only when it holds a row, so an empty merge leaves no file.
    /// The rows are never held.
    /// </summary>
    private async Task<(Dictionary<string, int> RowsPerFile, int TotalRows)> WriteMergedAsync(
        IReadOnlyList<string> paths,
        string outputPath,
        IReadOnlyList<string> columns,
        Func<string, IReadOnlyList<string>, Func<string[], string>[]> projectionFor,
        CancellationToken cancellationToken)
    {
        var rowsPerFile = new Dictionary<string, int>();
        var total = 0;
        var directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        Directory.CreateDirectory(directory);
        var temporary = Path.Combine(directory, $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.tmp");

        async IAsyncEnumerable<string[]> Rows()
        {
            foreach (var path in paths)
            {
                Func<string[], string>[] projection = [];
                var count = 0;
                await foreach (var row in _csvHelper.ReadRowsAsync(path, headers => projection = projectionFor(path, headers),
                    cancellationToken: cancellationToken).ConfigureAwait(false))
                {
                    var values = new string[projection.Length];
                    for (var i = 0; i < projection.Length; i++)
                        values[i] = projection[i](row);
                    count++;
                    yield return values;
                }
                rowsPerFile[Path.GetFileName(path)] = count;
                total += count;
            }
        }

        try
        {
            await _csvHelper.WriteRowsAsync(temporary, columns, Rows(), cancellationToken).ConfigureAwait(false);
            if (total > 0)
                File.Move(temporary, outputPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary))
                File.Delete(temporary);
        }
        return (rowsPerFile, total);
    }

    private static int IndexOf(IReadOnlyList<string> headers, string name)
    {
        for (var i = 0; i < headers.Count; i++)
            if (headers[i] == name)
                return i;
        return -1;
    }

    private static string ComputeSchemaId(List<string> headers)
    {
        var normalized = string.Join("|", headers.Select(h => h.Trim().ToLowerInvariant()).OrderBy(h => h));
        var bytes = Encoding.UTF8.GetBytes(normalized);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash)[..16];
    }

    /// <inheritdoc />
    public async Task<CsvMergeResult> MergeWithMetadataAsync(
        IEnumerable<string> sourcePaths,
        string outputPath,
        FilenameMetadataOptions metadataOptions,
        CancellationToken cancellationToken = default)
    {
        var paths = sourcePaths.ToList();
        if (paths.Count == 0)
        {
            return new CsvMergeResult
            {
                Success = false,
                Error = "No source files provided"
            };
        }

        try
        {
            // Validate schema compatibility first
            var validation = await ValidateSchemaCompatibilityAsync(paths, cancellationToken: cancellationToken).ConfigureAwait(false);
            if (!validation.IsCompatible)
            {
                return new CsvMergeResult
                {
                    Success = false,
                    Error = validation.Message ?? "Schema mismatch between files"
                };
            }

            // The metadata columns first, then the first file's order; each file's rows carry the values its name
            // gives (empty where the name does not match).
            var metadataColumns = GetMetadataColumns(metadataOptions);
            var firstHeaders = await _csvHelper.ReadHeadersAsync(paths[0], cancellationToken: cancellationToken).ConfigureAwait(false);
            var columns = metadataColumns.Concat(firstHeaders).ToList();
            var (rowsPerFile, totalRows) = await WriteMergedAsync(paths, outputPath, columns, (path, headers) =>
            {
                var metadata = ExtractFilenameMetadata(Path.GetFileName(path), metadataOptions);
                var constants = metadataColumns.Select(c => KeyValuePair.Create(c, metadata.GetValueOrDefault(c, string.Empty)));
                return Projection(columns, headers, constants);
            }, cancellationToken).ConfigureAwait(false);

            if (totalRows == 0)
            {
                return new CsvMergeResult
                {
                    Success = false,
                    Error = "No data found in source files"
                };
            }

            return new CsvMergeResult
            {
                Success = true,
                OutputPath = Path.GetFullPath(outputPath),
                TotalRows = totalRows,
                SourceFileCount = paths.Count,
                RowsPerFile = rowsPerFile
            };
        }
        catch (Exception ex)
        {
            return new CsvMergeResult
            {
                Success = false,
                Error = $"Merge failed: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Gets the list of metadata columns that will be added based on options.
    /// </summary>
    private static List<string> GetMetadataColumns(FilenameMetadataOptions options)
    {
        var columns = new List<string>();

        if (options.AddSourceColumn)
        {
            columns.Add(options.SourceColumnName);
        }

        if (options.ExtractDate || options.Preset is FilenameMetadataPreset.DateOnly or FilenameMetadataPreset.SensorDate)
        {
            columns.Add(options.DateColumnName);
        }

        // Add preset-specific columns
        var presetPatterns = GetPresetPatterns(options.Preset);
        foreach (var (columnName, _) in presetPatterns)
        {
            if (!columns.Contains(columnName))
            {
                columns.Add(columnName);
            }
        }

        // Add custom pattern columns
        if (options.CustomPatterns != null)
        {
            foreach (var columnName in options.CustomPatterns.Keys)
            {
                if (!columns.Contains(columnName))
                {
                    columns.Add(columnName);
                }
            }
        }

        return columns;
    }

    /// <summary>
    /// Extracts metadata from filename based on options.
    /// </summary>
    private static Dictionary<string, string> ExtractFilenameMetadata(string filename, FilenameMetadataOptions options)
    {
        var metadata = new Dictionary<string, string>();

        // Add source filename
        if (options.AddSourceColumn)
        {
            metadata[options.SourceColumnName] = filename;
        }

        // Extract date if configured
        if (options.ExtractDate || options.Preset is FilenameMetadataPreset.DateOnly or FilenameMetadataPreset.SensorDate)
        {
            var datePattern = options.DatePattern ?? GetDefaultDatePattern(options.Preset);
            if (!string.IsNullOrEmpty(datePattern))
            {
                var match = Regex.Match(filename, datePattern);
                if (match.Success && match.Groups.Count > 1)
                {
                    metadata[options.DateColumnName] = match.Groups[1].Value;
                }
                else
                {
                    metadata[options.DateColumnName] = "";
                }
            }
        }

        // Apply preset patterns
        var presetPatterns = GetPresetPatterns(options.Preset);
        foreach (var (columnName, pattern) in presetPatterns)
        {
            if (metadata.ContainsKey(columnName)) continue;

            var match = Regex.Match(filename, pattern, RegexOptions.IgnoreCase);
            metadata[columnName] = match.Success && match.Groups.Count > 1 ? match.Groups[1].Value : "";
        }

        // Apply custom patterns
        if (options.CustomPatterns != null)
        {
            foreach (var (columnName, pattern) in options.CustomPatterns)
            {
                if (metadata.ContainsKey(columnName)) continue;

                var match = Regex.Match(filename, pattern, RegexOptions.IgnoreCase);
                metadata[columnName] = match.Success && match.Groups.Count > 1 ? match.Groups[1].Value : "";
            }
        }

        return metadata;
    }

    /// <summary>
    /// Gets the default date pattern for a preset.
    /// </summary>
    private static string? GetDefaultDatePattern(FilenameMetadataPreset preset)
    {
        return preset switch
        {
            FilenameMetadataPreset.DateOnly => @"(\d{4}[.\-]?\d{2}[.\-]?\d{2})",
            FilenameMetadataPreset.SensorDate => @"(\d{4}\.\d{2}\.\d{2})",
            _ => null
        };
    }

    /// <summary>
    /// Gets regex patterns for preset metadata types.
    /// </summary>
    private static List<(string ColumnName, string Pattern)> GetPresetPatterns(FilenameMetadataPreset preset)
    {
        return preset switch
        {
            FilenameMetadataPreset.DateOnly => new List<(string, string)>
            {
                ("FileDate", @"(\d{4}[.\-]?\d{2}[.\-]?\d{2})")
            },
            FilenameMetadataPreset.SensorDate => new List<(string, string)>
            {
                ("FileDate", @"(\d{4}\.\d{2}\.\d{2})")
            },
            FilenameMetadataPreset.Manufacturing => new List<(string, string)>
            {
                ("BatchId", @"batch[_\-]?(\d+)"),
                ("Category", @"(normal|outlier|train|test|valid)")
            },
            FilenameMetadataPreset.Category => new List<(string, string)>
            {
                ("Category", @"(normal|outlier|train|test|valid)")
            },
            _ => new List<(string, string)>()
        };
    }

    /// <summary>
    /// Detects merge pattern from filenames.
    /// </summary>
    private static (string Pattern, double Confidence) DetectMergePattern(List<string> paths)
    {
        var fileNames = paths
            .Select(p => Path.GetFileNameWithoutExtension(p) ?? string.Empty)
            .Where(f => !string.IsNullOrEmpty(f))
            .ToList();

        if (fileNames.Count == 0)
        {
            return ("generic", 0.5);
        }

        // Check for normal/outlier pattern (common in anomaly detection datasets)
        var normalOutlierMatches = fileNames.Count(f => NormalOutlierPattern.IsMatch(f));
        if (normalOutlierMatches >= 2)
        {
            return ("normal_outlier", 0.9);
        }

        // Check for date-based pattern
        var dateMatches = fileNames.Count(f => DatePattern.IsMatch(f));
        if (dateMatches == fileNames.Count)
        {
            return ("date_series", 0.85);
        }

        // Check for sequence pattern (file_1.csv, file_2.csv, etc.)
        var sequenceMatches = fileNames.Count(f => SequencePattern.IsMatch(f));
        if (sequenceMatches == fileNames.Count)
        {
            return ("sequence", 0.8);
        }

        // Default: generic merge
        return ("generic", 0.6);
    }
}
