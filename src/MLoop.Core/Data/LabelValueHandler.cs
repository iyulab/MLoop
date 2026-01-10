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

        var data = await _csvHelper.ReadAsync(csvPath, cancellationToken: cancellationToken);

        if (data.Count == 0)
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
        if (!data[0].ContainsKey(labelColumn))
        {
            return new LabelAnalysisResult
            {
                TotalRows = data.Count,
                MissingCount = 0,
                ValidCount = 0,
                HasMissingValues = false,
                LabelColumn = labelColumn,
                Error = $"Label column '{labelColumn}' not found in data"
            };
        }

        int missingCount = 0;
        var valueCounts = new Dictionary<string, int>();

        foreach (var row in data)
        {
            var value = row.TryGetValue(labelColumn, out var v) ? v : null;

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

        return new LabelAnalysisResult
        {
            TotalRows = data.Count,
            MissingCount = missingCount,
            ValidCount = data.Count - missingCount,
            HasMissingValues = missingCount > 0,
            MissingPercentage = data.Count > 0 ? (missingCount * 100.0 / data.Count) : 0,
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

        var data = await _csvHelper.ReadAsync(csvPath, cancellationToken: cancellationToken);

        if (data.Count == 0)
        {
            return new LabelCleanResult
            {
                Success = false,
                Error = "Data file is empty",
                OriginalRowCount = 0,
                DroppedRowCount = 0,
                FinalRowCount = 0
            };
        }

        // Check if label column exists
        if (!data[0].ContainsKey(labelColumn))
        {
            return new LabelCleanResult
            {
                Success = false,
                Error = $"Label column '{labelColumn}' not found in data",
                OriginalRowCount = data.Count,
                DroppedRowCount = 0,
                FinalRowCount = data.Count
            };
        }

        var originalCount = data.Count;

        // Filter out rows with missing labels
        var cleanedData = data
            .Where(row =>
            {
                var value = row.TryGetValue(labelColumn, out var v) ? v : null;
                return !string.IsNullOrWhiteSpace(value);
            })
            .ToList();

        var droppedCount = originalCount - cleanedData.Count;

        if (droppedCount == 0)
        {
            return new LabelCleanResult
            {
                Success = true,
                OutputPath = csvPath,
                OriginalRowCount = originalCount,
                DroppedRowCount = 0,
                FinalRowCount = originalCount,
                Message = "No missing labels found, no changes made"
            };
        }

        if (cleanedData.Count == 0)
        {
            return new LabelCleanResult
            {
                Success = false,
                Error = "All rows have missing labels - cannot create training data",
                OriginalRowCount = originalCount,
                DroppedRowCount = droppedCount,
                FinalRowCount = 0
            };
        }

        // Write cleaned data
        try
        {
            var finalPath = await _csvHelper.WriteAsync(outputPath, cleanedData, cancellationToken: cancellationToken);

            _logger.Warning($"Dropped {droppedCount}/{originalCount} rows with missing labels ({droppedCount * 100.0 / originalCount:F1}%)");

            return new LabelCleanResult
            {
                Success = true,
                OutputPath = finalPath,
                OriginalRowCount = originalCount,
                DroppedRowCount = droppedCount,
                FinalRowCount = cleanedData.Count,
                DroppedPercentage = droppedCount * 100.0 / originalCount,
                Message = $"Dropped {droppedCount} rows with missing labels"
            };
        }
        catch (Exception ex)
        {
            return new LabelCleanResult
            {
                Success = false,
                Error = $"Failed to write cleaned data: {ex.Message}",
                OriginalRowCount = originalCount,
                DroppedRowCount = droppedCount,
                FinalRowCount = cleanedData.Count
            };
        }
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
