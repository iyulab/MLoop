namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Execution context for preprocessing scripts providing access to I/O, helpers, and metadata.
/// </summary>
public class PreprocessContext
{
    /// <summary>
    /// Path to input CSV file (from previous script or original user input).
    /// </summary>
    public required string InputPath { get; init; }

    /// <summary>
    /// Directory for intermediate output files (`.mloop/temp/`).
    /// Scripts should write output files here using unique names (e.g., "01_joined.csv", "02_features.csv").
    /// </summary>
    public required string OutputDirectory { get; init; }

    /// <summary>
    /// Project root directory (location of `.mloop/` folder).
    /// Use for accessing additional data files or configuration.
    /// </summary>
    public required string ProjectRoot { get; init; }

    /// <summary>
    /// CSV helper for reading and writing CSV files with proper encoding and parsing.
    /// </summary>
    /// <remarks>
    /// Provides convenient methods:
    /// - <c>ReadAsync(path)</c>: Read CSV to List&lt;Dictionary&lt;string, string&gt;&gt;
    /// - <c>WriteAsync(path, data)</c>: Write data to CSV with proper formatting
    /// </remarks>
    public required ICsvHelper Csv { get; init; }

    /// <summary>
    /// FilePrepper integration for advanced data preprocessing (encoding, cleaning, etc.).
    /// </summary>
    /// <remarks>
    /// FilePrepper provides 20x faster preprocessing than pandas for common operations:
    /// - CSV encoding normalization
    /// - Missing value handling
    /// - Type inference and conversion
    /// </remarks>
    public IFilePrepper? FilePrepper { get; init; }

    /// <summary>
    /// Logger for progress reporting and debugging.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Zero-based index of current script in execution sequence.
    /// First script (01_*.cs) has index 0, second (02_*.cs) has index 1, etc.
    /// </summary>
    public int ScriptIndex { get; init; }

    /// <summary>
    /// Name of current script file (e.g., "01_join_files.cs").
    /// Useful for logging and debugging.
    /// </summary>
    public required string ScriptName { get; init; }

    /// <summary>
    /// Optional metadata from previous scripts or training configuration.
    /// Scripts can store intermediate state here for coordination.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; init; }
}
