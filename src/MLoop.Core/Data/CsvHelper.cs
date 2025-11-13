using System.Globalization;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.Extensibility;
using MLoop.Extensibility.Preprocessing;

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
    public async Task<List<Dictionary<string, string>>> ReadAsync(
        string path,
        System.Text.Encoding? encoding = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"CSV file not found: {path}");
        }

        var targetEncoding = encoding ?? System.Text.Encoding.UTF8;

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            MissingFieldFound = null, // Ignore missing fields
            BadDataFound = null, // Ignore bad data
            TrimOptions = TrimOptions.Trim,
            Encoding = targetEncoding
        };

        using var reader = new StreamReader(path, targetEncoding, detectEncodingFromByteOrderMarks: true);
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
                var value = csv.GetField(header) ?? string.Empty;
                // Automatically clean comma-formatted numbers (e.g., "2,000" → "2000")
                record[header] = CleanNumericString(value);
            }
            records.Add(record);
        }

        return records;
    }

    /// <summary>
    /// Cleans numeric strings by removing thousand separators (commas).
    /// This enables ML.NET to correctly infer numeric types from Korean-formatted numbers.
    /// Examples: "1,000" → "1000", "2,000.5" → "2000.5"
    /// </summary>
    private static string CleanNumericString(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return input;
        }

        var trimmed = input.Trim();

        // Detect comma-formatted numbers: "1,000", "2,000.5", "-1,000"
        // Pattern: optional sign, digits with optional commas, optional decimal part
        if (Regex.IsMatch(trimmed, @"^-?[\d,]+\.?\d*$"))
        {
            return trimmed.Replace(",", "");
        }

        return trimmed;
    }

    /// <summary>
    /// Writes data to a CSV file.
    /// Column names are taken from the dictionary keys in the first row.
    /// </summary>
    public async Task<string> WriteAsync(
        string path,
        List<Dictionary<string, string>> data,
        System.Text.Encoding? encoding = null,
        CancellationToken cancellationToken = default)
    {
        if (data == null || data.Count == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }

        var targetEncoding = encoding ?? System.Text.Encoding.UTF8;

        // Ensure directory exists
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = ",",
            HasHeaderRecord = true
        };

        using var writer = new StreamWriter(path, false, targetEncoding);
        using var csv = new CsvWriter(writer, config);

        // Get headers from first record
        var headers = data[0].Keys.ToList();

        // Write header (always included)
        foreach (var header in headers)
        {
            csv.WriteField(header);
        }
        await csv.NextRecordAsync();

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
        return Path.GetFullPath(path);
    }

    /// <summary>
    /// Reads CSV file headers only (first row).
    /// Useful for schema validation without loading entire file.
    /// </summary>
    public async Task<List<string>> ReadHeadersAsync(
        string path,
        System.Text.Encoding? encoding = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"CSV file not found: {path}");
        }

        var targetEncoding = encoding ?? System.Text.Encoding.UTF8;

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            MissingFieldFound = null,
            BadDataFound = null,
            TrimOptions = TrimOptions.Trim,
            Encoding = targetEncoding
        };

        using var reader = new StreamReader(path, targetEncoding, detectEncodingFromByteOrderMarks: true);
        using var csv = new CsvReader(reader, config);

        await csv.ReadAsync();
        csv.ReadHeader();

        return csv.HeaderRecord?.ToList() ?? new List<string>();
    }
}
