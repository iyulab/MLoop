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
    /// This holds the whole file. For a large file, stream it with <see cref="ReadRowsAsync"/>.
    /// </para>
    /// </remarks>
    Task<List<Dictionary<string, string>>> ReadAsync(
        string path,
        Encoding? encoding = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Streams a CSV file's rows under the same reading rules as <see cref="ReadAsync"/>, without holding the file:
    /// each row is its values in header order (a new array per row). Stop early by leaving the loop.
    /// </summary>
    /// <param name="path">Path to CSV file</param>
    /// <param name="onHeaders">Receives the header before the first row</param>
    /// <param name="encoding">File encoding (detected when null)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <remarks>
    /// <code>
    /// var kept = 0;
    /// await foreach (var row in context.Csv.ReadRowsAsync("datasets/raw/big.csv"))
    ///     if (row[0] != "") kept++;
    /// </code>
    /// </remarks>
    IAsyncEnumerable<string[]> ReadRowsAsync(
        string path,
        Action<IReadOnlyList<string>>? onHeaders = null,
        Encoding? encoding = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Writes rows given as values in <paramref name="headers"/> order (UTF-8, comma-delimited, header first) —
    /// the streaming counterpart of <see cref="WriteAsync(string, List{Dictionary{string, string}}, List{string}?, CancellationToken)"/>.
    /// </summary>
    /// <param name="path">Output file path (parent directories are created)</param>
    /// <param name="headers">Column names, in the order the values come</param>
    /// <param name="rows">Rows, each its values in <paramref name="headers"/> order; enumerated once</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Absolute path of the written file</returns>
    Task<string> WriteRowsAsync(
        string path,
        IReadOnlyList<string> headers,
        IAsyncEnumerable<string[]> rows,
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
    /// Writes data to CSV file with specified column order.
    /// </summary>
    /// <param name="path">Output file path</param>
    /// <param name="data">Data to write (list of dictionaries)</param>
    /// <param name="columnOrder">Ordered list of column names to write (determines header and data order)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Path to written file</returns>
    Task<string> WriteAsync(
        string path,
        List<Dictionary<string, string>> data,
        List<string>? columnOrder,
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
