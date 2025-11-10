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
/// Extension methods for safer dictionary operations with better error messages.
/// </summary>
public static class DictionaryExtensions
{
    /// <summary>
    /// Safely gets a value from a dictionary with helpful error messages if key is missing.
    /// Includes available keys and fuzzy matching suggestions.
    /// </summary>
    public static string GetValueOrThrow(
        this Dictionary<string, string> row,
        string key,
        ILogger? logger = null)
    {
        if (!row.ContainsKey(key))
        {
            var available = row.Keys.ToArray();
            var suggestion = FindClosestMatch(key, available);

            var errorMessage = $"Column '{key}' not found in CSV row.\n" +
                             $"Available columns: {string.Join(", ", available)}";

            if (suggestion != null)
            {
                errorMessage += $"\nDid you mean '{suggestion}'?";
            }

            logger?.Error(errorMessage);
            throw new KeyNotFoundException(errorMessage);
        }

        return row[key];
    }

    /// <summary>
    /// Finds the closest matching string using Levenshtein distance.
    /// Returns null if no good match is found (distance > 3).
    /// </summary>
    private static string? FindClosestMatch(string target, string[] options)
    {
        if (options.Length == 0) return null;

        var closest = options
            .Select(opt => new { Option = opt, Distance = LevenshteinDistance(target, opt) })
            .OrderBy(x => x.Distance)
            .First();

        // Only suggest if distance is reasonable (≤ 3 edits)
        return closest.Distance <= 3 ? closest.Option : null;
    }

    /// <summary>
    /// Calculates Levenshtein distance between two strings.
    /// </summary>
    private static int LevenshteinDistance(string source, string target)
    {
        if (string.IsNullOrEmpty(source))
            return string.IsNullOrEmpty(target) ? 0 : target.Length;

        if (string.IsNullOrEmpty(target))
            return source.Length;

        var distance = new int[source.Length + 1, target.Length + 1];

        for (int i = 0; i <= source.Length; i++)
            distance[i, 0] = i;

        for (int j = 0; j <= target.Length; j++)
            distance[0, j] = j;

        for (int i = 1; i <= source.Length; i++)
        {
            for (int j = 1; j <= target.Length; j++)
            {
                var cost = target[j - 1] == source[i - 1] ? 0 : 1;

                distance[i, j] = Math.Min(
                    Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1),
                    distance[i - 1, j - 1] + cost);
            }
        }

        return distance[source.Length, target.Length];
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
