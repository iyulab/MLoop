using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.Extensibility;

namespace MLoop.Core.Data;

/// <summary>
/// High-performance CSV helper implementation using CsvHelper library.
/// Provides simple dictionary-based CSV reading and writing for preprocessing scripts.
/// </summary>
public class CsvHelperImpl : ICsvHelper
{
    /// <summary>
    /// Reads a CSV file and returns the data as a list of dictionaries.
    /// Each dictionary represents a row, with column names as keys.
    /// </summary>
    public async Task<List<Dictionary<string, string>>> ReadAsync(string filePath, bool hasHeader = true)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"CSV file not found: {filePath}");
        }

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = hasHeader,
            MissingFieldFound = null, // Ignore missing fields
            BadDataFound = null, // Ignore bad data
            TrimOptions = TrimOptions.Trim
        };

        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, config);

        var records = new List<Dictionary<string, string>>();

        await csv.ReadAsync();
        csv.ReadHeader();
        var headers = csv.HeaderRecord ?? throw new InvalidOperationException("No headers found in CSV");

        while (await csv.ReadAsync())
        {
            var record = new Dictionary<string, string>();
            foreach (var header in headers)
            {
                record[header] = csv.GetField(header) ?? string.Empty;
            }
            records.Add(record);
        }

        return records;
    }

    /// <summary>
    /// Writes data to a CSV file.
    /// Column names are taken from the dictionary keys in the first row.
    /// </summary>
    public async Task<string> WriteAsync(string filePath, List<Dictionary<string, string>> data)
    {
        return await WriteAsync(filePath, data, ',', true);
    }

    /// <summary>
    /// Writes data to a CSV file with custom options.
    /// </summary>
    public async Task<string> WriteAsync(
        string filePath,
        List<Dictionary<string, string>> data,
        char delimiter = ',',
        bool includeHeader = true)
    {
        if (data == null || data.Count == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }

        // Ensure directory exists
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = delimiter.ToString(),
            HasHeaderRecord = includeHeader
        };

        using var writer = new StreamWriter(filePath);
        using var csv = new CsvWriter(writer, config);

        // Get headers from first record
        var headers = data[0].Keys.ToList();

        // Write header
        if (includeHeader)
        {
            foreach (var header in headers)
            {
                csv.WriteField(header);
            }
            await csv.NextRecordAsync();
        }

        // Write records
        foreach (var record in data)
        {
            foreach (var header in headers)
            {
                csv.WriteField(record.TryGetValue(header, out var value) ? value : string.Empty);
            }
            await csv.NextRecordAsync();
        }

        // Return absolute path
        return Path.GetFullPath(filePath);
    }
}
