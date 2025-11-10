namespace MLoop.Extensibility;

/// <summary>
/// Provides context and utilities for preprocessing script execution.
/// Contains all information and helpers needed for data transformation operations.
/// </summary>
public class PreprocessContext
{
    /// <summary>
    /// Gets the absolute path to the input CSV file.
    /// For the first script (01_*.cs), this is the original raw data file.
    /// For subsequent scripts, this is the output from the previous script.
    /// </summary>
    public required string InputPath { get; init; }

    /// <summary>
    /// Gets the absolute path to the directory where output files should be written.
    /// Each script should write its output to this directory with a unique filename.
    /// Recommended pattern: Path.Combine(OutputDirectory, "01_scriptname.csv")
    /// </summary>
    public required string OutputDirectory { get; init; }

    /// <summary>
    /// Gets the absolute path to the project root directory.
    /// Useful for accessing other project files or directories.
    /// </summary>
    public required string ProjectRoot { get; init; }

    /// <summary>
    /// Gets the CSV helper for reading and writing CSV files.
    /// Provides high-performance CSV operations with automatic type inference.
    /// </summary>
    public required ICsvHelper Csv { get; init; }

    /// <summary>
    /// Gets the logger for outputting messages to the user.
    /// Use this to provide feedback about preprocessing progress and issues.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Gets read-only metadata about the current operation.
    /// Common keys include:
    /// - "LabelColumn": The name of the label column (if specified)
    /// - "ScriptSequence": The sequence number of this script (e.g., 1 for 01_*.cs)
    /// - "TotalScripts": Total number of scripts to execute
    /// </summary>
    public IReadOnlyDictionary<string, object> Metadata { get; private set; } =
        new Dictionary<string, object>();

    private readonly Dictionary<string, object> _mutableMetadata = new();

    /// <summary>
    /// Initializes the metadata dictionary.
    /// </summary>
    /// <param name="metadata">Initial metadata values.</param>
    public void InitializeMetadata(Dictionary<string, object> metadata)
    {
        _mutableMetadata.Clear();
        foreach (var kvp in metadata)
        {
            _mutableMetadata[kvp.Key] = kvp.Value;
        }
        Metadata = _mutableMetadata;
    }

    /// <summary>
    /// Sets or updates a metadata value.
    /// Useful for passing data between scripts or storing context for later scripts.
    /// </summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="value">The metadata value.</param>
    public void SetMetadata(string key, object value)
    {
        _mutableMetadata[key] = value;
    }

    /// <summary>
    /// Gets a metadata value with type casting.
    /// </summary>
    /// <typeparam name="T">The expected type of the metadata value.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The metadata value cast to type T, or default(T) if not found.</returns>
    public T? GetMetadata<T>(string key)
    {
        if (Metadata.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return default;
    }
}

/// <summary>
/// Helper interface for CSV file operations.
/// Provides high-performance reading and writing of CSV data.
/// </summary>
public interface ICsvHelper
{
    /// <summary>
    /// Reads a CSV file and returns the data as a list of dictionaries.
    /// Each dictionary represents a row, with column names as keys.
    /// </summary>
    /// <param name="filePath">The absolute or relative path to the CSV file.</param>
    /// <param name="hasHeader">Whether the CSV file has a header row (default: true).</param>
    /// <returns>A list of dictionaries representing the CSV data.</returns>
    Task<List<Dictionary<string, string>>> ReadAsync(string filePath, bool hasHeader = true);

    /// <summary>
    /// Writes data to a CSV file.
    /// Column names are taken from the dictionary keys in the first row.
    /// </summary>
    /// <param name="filePath">The absolute or relative path where the CSV file should be written.</param>
    /// <param name="data">The data to write, as a list of dictionaries.</param>
    /// <returns>The absolute path to the written file.</returns>
    Task<string> WriteAsync(string filePath, List<Dictionary<string, string>> data);

    /// <summary>
    /// Writes data to a CSV file with custom options.
    /// </summary>
    /// <param name="filePath">The absolute or relative path where the CSV file should be written.</param>
    /// <param name="data">The data to write, as a list of dictionaries.</param>
    /// <param name="delimiter">The delimiter to use (default: comma).</param>
    /// <param name="includeHeader">Whether to include a header row (default: true).</param>
    /// <returns>The absolute path to the written file.</returns>
    Task<string> WriteAsync(
        string filePath,
        List<Dictionary<string, string>> data,
        char delimiter = ',',
        bool includeHeader = true);
}
