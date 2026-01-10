namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Integration interface for FilePrepper - fast data preprocessing library (20x faster than pandas).
/// </summary>
/// <remarks>
/// FilePrepper provides optimized operations for common preprocessing tasks:
/// - Encoding normalization (UTF-8 conversion, BOM handling)
/// - Missing value imputation
/// - Type inference and conversion
/// - Duplicate detection and removal
/// - Outlier detection
/// </remarks>
public interface IFilePrepper
{
    /// <summary>
    /// Normalizes CSV file encoding to UTF-8 with proper BOM handling.
    /// </summary>
    /// <param name="inputPath">Input CSV path</param>
    /// <param name="outputPath">Output CSV path</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to normalized file</returns>
    Task<string> NormalizeEncodingAsync(
        string inputPath,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects and removes duplicate rows.
    /// </summary>
    /// <param name="inputPath">Input CSV path</param>
    /// <param name="outputPath">Output CSV path</param>
    /// <param name="keyColumns">Columns to use for duplicate detection (null = all columns)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Tuple of (output path, number of duplicates removed)</returns>
    Task<(string Path, int DuplicatesRemoved)> RemoveDuplicatesAsync(
        string inputPath,
        string outputPath,
        string[]? keyColumns = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Infers and converts column data types.
    /// </summary>
    /// <param name="inputPath">Input CSV path</param>
    /// <param name="outputPath">Output CSV path</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Dictionary of column name to inferred type</returns>
    Task<Dictionary<string, string>> InferAndConvertTypesAsync(
        string inputPath,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs simple unpivot (wide-to-long) transformation.
    /// Converts columns like "Q1,Q2,Q3,Q4" into rows with "Quarter" and "Value" columns.
    /// </summary>
    /// <param name="inputPath">Input CSV path</param>
    /// <param name="outputPath">Output CSV path</param>
    /// <param name="baseColumns">Columns to keep as-is (e.g., ["Region", "Product"])</param>
    /// <param name="unpivotColumns">Columns to unpivot (e.g., ["Q1", "Q2", "Q3", "Q4"])</param>
    /// <param name="indexColumn">Name for the new index column (e.g., "Quarter")</param>
    /// <param name="valueColumn">Name for the new value column (e.g., "Sales")</param>
    /// <param name="skipEmptyRows">Whether to skip rows with empty values (default: true)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to transformed file and row count change info</returns>
    Task<UnpivotResult> UnpivotSimpleAsync(
        string inputPath,
        string outputPath,
        string[] baseColumns,
        string[] unpivotColumns,
        string indexColumn = "Index",
        string valueColumn = "Value",
        bool skipEmptyRows = true,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Parses Korean time format (오전/오후) to DateTime.
    /// Handles formats like "오전 9:30", "오후 2:45".
    /// </summary>
    /// <param name="inputPath">Input CSV path</param>
    /// <param name="outputPath">Output CSV path</param>
    /// <param name="sourceColumn">Column containing Korean time string</param>
    /// <param name="targetColumn">Name for parsed DateTime column</param>
    /// <param name="baseDate">Base date to use for time-only values (default: 2000-01-01)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to transformed file and conversion stats</returns>
    Task<KoreanTimeParseResult> ParseKoreanTimeAsync(
        string inputPath,
        string outputPath,
        string sourceColumn,
        string targetColumn,
        DateTime? baseDate = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Result of unpivot operation.
/// </summary>
public class UnpivotResult
{
    /// <summary>
    /// Path to the output file.
    /// </summary>
    public string OutputPath { get; set; } = string.Empty;

    /// <summary>
    /// Number of rows in the original file.
    /// </summary>
    public int OriginalRowCount { get; set; }

    /// <summary>
    /// Number of rows in the transformed file.
    /// </summary>
    public int TransformedRowCount { get; set; }

    /// <summary>
    /// Number of empty rows skipped.
    /// </summary>
    public int SkippedEmptyRows { get; set; }

    /// <summary>
    /// Whether the operation was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if operation failed.
    /// </summary>
    public string? Error { get; set; }
}

/// <summary>
/// Result of Korean time parsing operation.
/// </summary>
public class KoreanTimeParseResult
{
    /// <summary>
    /// Path to the output file.
    /// </summary>
    public string OutputPath { get; set; } = string.Empty;

    /// <summary>
    /// Total number of rows processed.
    /// </summary>
    public int TotalRows { get; set; }

    /// <summary>
    /// Number of successfully parsed values.
    /// </summary>
    public int SuccessfullyParsed { get; set; }

    /// <summary>
    /// Number of values that could not be parsed.
    /// </summary>
    public int FailedToParse { get; set; }

    /// <summary>
    /// Whether the operation was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if operation failed.
    /// </summary>
    public string? Error { get; set; }
}
