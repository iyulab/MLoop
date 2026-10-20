using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace MLoop.Core.Data;

/// <summary>One part of a CSV file: the rows whose partitioning column holds <see cref="Value"/>.</summary>
public sealed record CsvSlice(string Value, string Path, int Rows);

/// <summary>
/// Splits a CSV file into one file per distinct value of a column, each with the original header, so
/// every part can go through exactly the path the whole file goes through (loader, schema checks,
/// evaluation) — a per-slice computation over an already-transformed view would be a second
/// implementation of that path.
/// </summary>
public static class CsvPartition
{
    /// <summary>
    /// Writes the parts under <paramref name="directory"/> and returns them largest first. Values are
    /// compared as written, after trimming; an empty cell is its own part, named <c>(empty)</c>.
    /// Throws <see cref="ArgumentException"/> when the column is absent, and
    /// <see cref="InvalidOperationException"/> when it has more than <paramref name="maxParts"/> values
    /// — a per-value table that long is not a slice report.
    /// </summary>
    public static async Task<IReadOnlyList<CsvSlice>> ByColumnAsync(
        string csvPath, string column, string directory, int maxParts = 50, CancellationToken cancellationToken = default)
    {
        var (readPath, detection) = EncodingDetector.ConvertToUtf8WithBom(csvPath);
        try
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                MissingFieldFound = null,
                BadDataFound = null,
            };

            using var reader = new StreamReader(readPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            using var csv = new CsvReader(reader, config);

            if (!await csv.ReadAsync().ConfigureAwait(false) || !csv.ReadHeader() || csv.HeaderRecord is null)
                throw new ArgumentException($"'{csvPath}' has no header row.");

            var header = csv.HeaderRecord;
            var index = Array.FindIndex(header, h => h.Trim().Equals(column, StringComparison.OrdinalIgnoreCase));
            if (index < 0)
                throw new ArgumentException(
                    $"Column '{column}' is not in the data. Columns: {string.Join(", ", header)}.");

            var parts = new Dictionary<string, List<string[]>>(StringComparer.Ordinal);
            while (await csv.ReadAsync().ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var record = csv.Parser.Record ?? [];
                var value = index < record.Length ? record[index].Trim() : "";
                if (value.Length == 0)
                    value = "(empty)";

                if (!parts.TryGetValue(value, out var rows))
                {
                    if (parts.Count == maxParts)
                        throw new InvalidOperationException(
                            $"Column '{column}' has more than {maxParts} distinct values — slice by a column with fewer.");
                    parts[value] = rows = [];
                }
                rows.Add(record);
            }

            Directory.CreateDirectory(directory);
            var slices = new List<CsvSlice>();
            var n = 0;
            foreach (var (value, rows) in parts.OrderByDescending(p => p.Value.Count).ThenBy(p => p.Key, StringComparer.Ordinal))
            {
                var path = Path.Combine(directory, $"slice-{n++:D3}.csv");
                await using (var writer = new StreamWriter(path, false, new System.Text.UTF8Encoding(false)))
                await using (var output = new CsvWriter(writer, CultureInfo.InvariantCulture))
                {
                    foreach (var h in header)
                        output.WriteField(h);
                    await output.NextRecordAsync().ConfigureAwait(false);
                    foreach (var row in rows)
                    {
                        foreach (var field in row)
                            output.WriteField(field);
                        await output.NextRecordAsync().ConfigureAwait(false);
                    }
                }
                slices.Add(new CsvSlice(value, path, rows.Count));
            }

            return slices;
        }
        finally
        {
            if (detection.WasConverted && File.Exists(readPath))
                File.Delete(readPath);
        }
    }
}
