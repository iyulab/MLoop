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
}
