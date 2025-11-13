using System.Text;

namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Helper for reading and writing CSV files with proper encoding and parsing.
/// </summary>
public interface ICsvHelper
{
    /// <summary>
    /// Reads CSV file into memory as list of dictionaries (column name → value).
    /// </summary>
    /// <param name="path">Path to CSV file (absolute or relative to project root)</param>
    /// <param name="encoding">File encoding (defaults to UTF-8 with BOM detection)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of rows, each row as dictionary of column name to value</returns>
    /// <exception cref="FileNotFoundException">CSV file not found</exception>
    /// <exception cref="CsvParsingException">CSV parsing failed (invalid format, encoding issues)</exception>
    /// <remarks>
    /// <para>
    /// Example usage:
    /// <code>
    /// var data = await context.Csv.ReadAsync("datasets/raw/machines.csv");
    /// foreach (var row in data)
    /// {
    ///     Console.WriteLine($"Machine: {row["machine"]}, Capacity: {row["capacity"]}");
    /// }
    /// </code>
    /// </para>
    /// <para>
    /// For large files (>100MB), consider streaming or FilePrepper integration.
    /// </para>
    /// </remarks>
    Task<List<Dictionary<string, string>>> ReadAsync(
        string path,
        Encoding? encoding = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Writes data to CSV file with proper formatting and encoding.
    /// </summary>
    /// <param name="path">Output file path</param>
    /// <param name="data">Data to write (list of dictionaries)</param>
    /// <param name="encoding">Output encoding (defaults to UTF-8)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to written file (same as input path)</returns>
    /// <exception cref="IOException">Failed to write file</exception>
    /// <remarks>
    /// <para>
    /// Automatically creates parent directories if needed.
    /// Overwrites existing file.
    /// </para>
    /// <para>
    /// Example usage:
    /// <code>
    /// var output = new List&lt;Dictionary&lt;string, string&gt;&gt;
    /// {
    ///     new() { ["id"] = "1", ["name"] = "Item1" },
    ///     new() { ["id"] = "2", ["name"] = "Item2" }
    /// };
    /// await context.Csv.WriteAsync("output.csv", output);
    /// </code>
    /// </para>
    /// </remarks>
    Task<string> WriteAsync(
        string path,
        List<Dictionary<string, string>> data,
        Encoding? encoding = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Reads CSV file headers only (first row).
    /// Useful for schema validation without loading entire file.
    /// </summary>
    /// <param name="path">Path to CSV file</param>
    /// <param name="encoding">File encoding</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of column names</returns>
    Task<List<string>> ReadHeadersAsync(
        string path,
        Encoding? encoding = null,
        CancellationToken cancellationToken = default);
}
