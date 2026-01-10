namespace MLoop.Core.Data;

/// <summary>
/// Interface for merging multiple CSV files with same schema.
/// Supports automatic detection and merging of compatible CSV files.
/// </summary>
public interface ICsvMerger
{
    /// <summary>
    /// Discovers CSV files in a directory that can be merged (same schema).
    /// </summary>
    /// <param name="directory">Directory to search for CSV files</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Groups of CSV files that share the same schema</returns>
    Task<List<CsvMergeGroup>> DiscoverMergeableCsvsAsync(
        string directory,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Merges multiple CSV files into a single file.
    /// All files must have the same schema (column headers).
    /// </summary>
    /// <param name="sourcePaths">Paths to source CSV files</param>
    /// <param name="outputPath">Path for merged output file</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Merge result with statistics</returns>
    Task<CsvMergeResult> MergeAsync(
        IEnumerable<string> sourcePaths,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Validates that multiple CSV files have compatible schemas.
    /// </summary>
    /// <param name="paths">Paths to CSV files to validate</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Validation result with details</returns>
    Task<CsvSchemaValidation> ValidateSchemaCompatibilityAsync(
        IEnumerable<string> paths,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Merges multiple CSV files with filename metadata extraction.
    /// Extracts date, category, or custom patterns from filenames and adds as columns.
    /// </summary>
    /// <param name="sourcePaths">Paths to source CSV files</param>
    /// <param name="outputPath">Path for merged output file</param>
    /// <param name="metadataOptions">Options for filename metadata extraction</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Merge result with statistics</returns>
    Task<CsvMergeResult> MergeWithMetadataAsync(
        IEnumerable<string> sourcePaths,
        string outputPath,
        FilenameMetadataOptions metadataOptions,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Group of CSV files that share the same schema and can be merged.
/// </summary>
public class CsvMergeGroup
{
    /// <summary>
    /// Schema identifier (hash of column names).
    /// </summary>
    public required string SchemaId { get; init; }

    /// <summary>
    /// Column headers shared by all files in this group.
    /// </summary>
    public required List<string> Columns { get; init; }

    /// <summary>
    /// Paths to CSV files in this group.
    /// </summary>
    public required List<string> FilePaths { get; init; }

    /// <summary>
    /// Detected merge pattern (e.g., "normal_outlier", "date_series", "generic").
    /// </summary>
    public string? DetectedPattern { get; init; }

    /// <summary>
    /// Confidence score for the detected pattern (0-1).
    /// </summary>
    public double PatternConfidence { get; init; }
}

/// <summary>
/// Result of a CSV merge operation.
/// </summary>
public class CsvMergeResult
{
    /// <summary>
    /// Whether the merge was successful.
    /// </summary>
    public bool Success { get; init; }

    /// <summary>
    /// Path to the merged output file.
    /// </summary>
    public string? OutputPath { get; init; }

    /// <summary>
    /// Total number of rows in the merged file (excluding header).
    /// </summary>
    public int TotalRows { get; init; }

    /// <summary>
    /// Number of source files merged.
    /// </summary>
    public int SourceFileCount { get; init; }

    /// <summary>
    /// Row counts per source file.
    /// </summary>
    public Dictionary<string, int> RowsPerFile { get; init; } = new();

    /// <summary>
    /// Error message if merge failed.
    /// </summary>
    public string? Error { get; init; }
}

/// <summary>
/// Result of schema compatibility validation.
/// </summary>
public class CsvSchemaValidation
{
    /// <summary>
    /// Whether all files have compatible schemas.
    /// </summary>
    public bool IsCompatible { get; init; }

    /// <summary>
    /// Common columns across all files.
    /// </summary>
    public List<string> CommonColumns { get; init; } = new();

    /// <summary>
    /// Columns present in some files but not others.
    /// </summary>
    public Dictionary<string, List<string>> MismatchedColumns { get; init; } = new();

    /// <summary>
    /// Files that couldn't be read.
    /// </summary>
    public Dictionary<string, string> UnreadableFiles { get; init; } = new();

    /// <summary>
    /// Validation message describing the result.
    /// </summary>
    public string? Message { get; init; }
}

/// <summary>
/// Configuration for filename metadata extraction during merge.
/// Extracts structured information from filenames and adds as columns.
/// </summary>
public class FilenameMetadataOptions
{
    /// <summary>
    /// Whether to add a source filename column (default: true).
    /// </summary>
    public bool AddSourceColumn { get; set; } = true;

    /// <summary>
    /// Name for the source filename column (default: "SourceFile").
    /// </summary>
    public string SourceColumnName { get; set; } = "SourceFile";

    /// <summary>
    /// Whether to extract date from filename (default: false).
    /// </summary>
    public bool ExtractDate { get; set; } = false;

    /// <summary>
    /// Name for the extracted date column (default: "FileDate").
    /// </summary>
    public string DateColumnName { get; set; } = "FileDate";

    /// <summary>
    /// Regex pattern for date extraction. First capture group is used.
    /// Example: @"(\d{4}[.\-]?\d{2}[.\-]?\d{2})"
    /// </summary>
    public string? DatePattern { get; set; }

    /// <summary>
    /// Custom regex patterns for metadata extraction.
    /// Key: output column name, Value: regex pattern with capture group.
    /// Example: { "BatchId", @"batch[_-]?(\d+)" }
    /// </summary>
    public Dictionary<string, string>? CustomPatterns { get; set; }

    /// <summary>
    /// Preset pattern for common filename formats.
    /// </summary>
    public FilenameMetadataPreset Preset { get; set; } = FilenameMetadataPreset.None;
}

/// <summary>
/// Preset patterns for common filename metadata formats.
/// </summary>
public enum FilenameMetadataPreset
{
    /// <summary>
    /// No preset pattern.
    /// </summary>
    None,

    /// <summary>
    /// Date pattern: yyyy.MM.dd or yyyy-MM-dd or yyyyMMdd.
    /// Extracts: FileDate
    /// </summary>
    DateOnly,

    /// <summary>
    /// Sensor data pattern: sensor-yyyy.MM.dd.csv.
    /// Extracts: FileDate
    /// </summary>
    SensorDate,

    /// <summary>
    /// Manufacturing data pattern: batch_123_normal.csv.
    /// Extracts: BatchId, Category
    /// </summary>
    Manufacturing,

    /// <summary>
    /// Category pattern: train.csv, test.csv, normal.csv, outlier.csv.
    /// Extracts: Category
    /// </summary>
    Category
}
