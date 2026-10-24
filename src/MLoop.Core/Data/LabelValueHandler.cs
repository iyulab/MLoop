using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Data;

/// <summary>
/// Handles missing values in label column for classification/regression tasks.
/// ML training fails when the label column has empty/missing values.
/// </summary>
public class LabelValueHandler
{
    private readonly ICsvHelper _csvHelper;
    private readonly ILogger _logger;

    public LabelValueHandler(ICsvHelper csvHelper, ILogger logger)
    {
        _csvHelper = csvHelper ?? throw new ArgumentNullException(nameof(csvHelper));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Analyzes the label column for missing values.
    /// </summary>
    /// <param name="csvPath">Path to CSV file</param>
    /// <param name="labelColumn">Name of the label column</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Analysis result with statistics</returns>
    public async Task<LabelAnalysisResult> AnalyzeLabelColumnAsync(
        string csvPath,
        string labelColumn,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"CSV file not found: {csvPath}");
        }

        // Streamed: only the label column's counts are kept, never the rows.
        int totalRows = 0;
        int missingCount = 0;
        var valueCounts = new Dictionary<string, int>();
        var labelIndex = -1;
        await foreach (var row in _csvHelper.ReadRowsAsync(csvPath, headers => labelIndex = IndexOf(headers, labelColumn),
            cancellationToken: cancellationToken).ConfigureAwait(false))
        {
            totalRows++;
            if (labelIndex < 0)
            {
                continue;
            }

            var value = row[labelIndex];
            if (string.IsNullOrWhiteSpace(value))
            {
                missingCount++;
            }
            else
            {
                valueCounts.TryGetValue(value, out var count);
                valueCounts[value] = count + 1;
            }
        }

        if (totalRows == 0)
        {
            return new LabelAnalysisResult
            {
                TotalRows = 0,
                MissingCount = 0,
                ValidCount = 0,
                HasMissingValues = false,
                LabelColumn = labelColumn
            };
        }

        // Check if label column exists
        if (labelIndex < 0)
        {
            return new LabelAnalysisResult
            {
                TotalRows = totalRows,
                MissingCount = 0,
                ValidCount = 0,
                HasMissingValues = false,
                LabelColumn = labelColumn,
                Error = $"Label column '{labelColumn}' not found in data"
            };
        }

        return new LabelAnalysisResult
        {
            TotalRows = totalRows,
            MissingCount = missingCount,
            ValidCount = totalRows - missingCount,
            HasMissingValues = missingCount > 0,
            MissingPercentage = missingCount * 100.0 / totalRows,
            LabelColumn = labelColumn,
            UniqueValueCount = valueCounts.Count,
            ValueDistribution = valueCounts
        };
    }

    /// <summary>
    /// Drops rows with missing label values and saves to a new file.
    /// </summary>
    /// <param name="csvPath">Path to input CSV file</param>
    /// <param name="outputPath">Path for output CSV file (can be same as input to overwrite)</param>
    /// <param name="labelColumn">Name of the label column</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Result with statistics</returns>
    public async Task<LabelCleanResult> DropMissingLabelsAsync(
        string csvPath,
        string outputPath,
        string labelColumn,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"CSV file not found: {csvPath}");
        }

        // One streamed pass writes the kept rows to a temporary file beside the output and counts both kinds;
        // the file takes the output's place only when something was dropped and something is left — so the
        // output may be the input itself, and the rows are never held.
        var directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        Directory.CreateDirectory(directory);
        var temporary = Path.Combine(directory, $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.tmp");
        int originalCount = 0, keptCount = 0;
        var labelIndex = -1;
        try
        {
            var headers = await _csvHelper.ReadHeadersAsync(csvPath, cancellationToken: cancellationToken).ConfigureAwait(false);
            labelIndex = headers.IndexOf(labelColumn);
            if (labelIndex >= 0)
            {
                await _csvHelper.WriteRowsAsync(temporary, headers, Kept(), cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await foreach (var _ in _csvHelper.ReadRowsAsync(csvPath, cancellationToken: cancellationToken).ConfigureAwait(false))
                {
                    originalCount++;
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            File.Delete(temporary);
            return new LabelCleanResult
            {
                Success = false,
                Error = $"Failed to write cleaned data: {ex.Message}",
                OriginalRowCount = originalCount,
                DroppedRowCount = originalCount - keptCount,
                FinalRowCount = keptCount
            };
        }

        async IAsyncEnumerable<string[]> Kept()
        {
            await foreach (var row in _csvHelper.ReadRowsAsync(csvPath, cancellationToken: cancellationToken).ConfigureAwait(false))
            {
                originalCount++;
                if (!string.IsNullOrWhiteSpace(row[labelIndex]))
                {
                    keptCount++;
                    yield return row;
                }
            }
        }

        var droppedCount = originalCount - keptCount;
        LabelCleanResult? unchanged = originalCount == 0
            ? new LabelCleanResult
            {
                Success = false,
                Error = "Data file is empty",
                OriginalRowCount = 0,
                DroppedRowCount = 0,
                FinalRowCount = 0
            }
            : labelIndex < 0
            ? new LabelCleanResult
            {
                Success = false,
                Error = $"Label column '{labelColumn}' not found in data",
                OriginalRowCount = originalCount,
                DroppedRowCount = 0,
                FinalRowCount = originalCount
            }
            : droppedCount == 0
            ? new LabelCleanResult
            {
                Success = true,
                OutputPath = csvPath,
                OriginalRowCount = originalCount,
                DroppedRowCount = 0,
                FinalRowCount = originalCount,
                Message = "No missing labels found, no changes made"
            }
            : keptCount == 0
            ? new LabelCleanResult
            {
                Success = false,
                Error = "All rows have missing labels - cannot create training data",
                OriginalRowCount = originalCount,
                DroppedRowCount = droppedCount,
                FinalRowCount = 0
            }
            : null;
        if (unchanged is not null)
        {
            File.Delete(temporary);
            return unchanged;
        }

        File.Move(temporary, outputPath, overwrite: true);
        _logger.Warning($"Dropped {droppedCount}/{originalCount} rows with missing labels ({droppedCount * 100.0 / originalCount:F1}%)");

        return new LabelCleanResult
        {
            Success = true,
            OutputPath = Path.GetFullPath(outputPath),
            OriginalRowCount = originalCount,
            DroppedRowCount = droppedCount,
            FinalRowCount = keptCount,
            DroppedPercentage = droppedCount * 100.0 / originalCount,
            Message = $"Dropped {droppedCount} rows with missing labels"
        };
    }


    private static int IndexOf(IReadOnlyList<string> headers, string name)
    {
        for (var i = 0; i < headers.Count; i++)
        {
            if (headers[i] == name)
            {
                return i;
            }
        }
        return -1;
    }
}

/// <summary>
/// Result of label column analysis.
/// </summary>
public class LabelAnalysisResult
{
    /// <summary>
    /// Total number of rows in the dataset.
    /// </summary>
    public int TotalRows { get; init; }

    /// <summary>
    /// Number of rows with missing label values.
    /// </summary>
    public int MissingCount { get; init; }

    /// <summary>
    /// Number of rows with valid label values.
    /// </summary>
    public int ValidCount { get; init; }

    /// <summary>
    /// Whether any missing values were found.
    /// </summary>
    public bool HasMissingValues { get; init; }

    /// <summary>
    /// Percentage of rows with missing values.
    /// </summary>
    public double MissingPercentage { get; init; }

    /// <summary>
    /// Name of the label column.
    /// </summary>
    public required string LabelColumn { get; init; }

    /// <summary>
    /// Number of unique label values (excluding missing).
    /// </summary>
    public int UniqueValueCount { get; init; }

    /// <summary>
    /// Distribution of label values.
    /// </summary>
    public Dictionary<string, int> ValueDistribution { get; init; } = new();

    /// <summary>
    /// Error message if analysis failed.
    /// </summary>
    public string? Error { get; init; }
}

/// <summary>
/// Result of dropping rows with missing labels.
/// </summary>
public class LabelCleanResult
{
    /// <summary>
    /// Whether the operation was successful.
    /// </summary>
    public bool Success { get; init; }

    /// <summary>
    /// Path to the output file.
    /// </summary>
    public string? OutputPath { get; init; }

    /// <summary>
    /// Original number of rows.
    /// </summary>
    public int OriginalRowCount { get; init; }

    /// <summary>
    /// Number of rows dropped due to missing labels.
    /// </summary>
    public int DroppedRowCount { get; init; }

    /// <summary>
    /// Final number of rows after cleaning.
    /// </summary>
    public int FinalRowCount { get; init; }

    /// <summary>
    /// Percentage of rows dropped.
    /// </summary>
    public double DroppedPercentage { get; init; }

    /// <summary>
    /// Description message.
    /// </summary>
    public string? Message { get; init; }

    /// <summary>
    /// Error message if operation failed.
    /// </summary>
    public string? Error { get; init; }
}
