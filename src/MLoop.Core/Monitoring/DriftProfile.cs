using System.Globalization;
using System.Text.Json.Serialization;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Monitoring;

/// <summary>
/// How the inputs a model was trained on were distributed, column by column — the reference a later
/// window of inputs is compared against for drift. A property of the training, like the saved schema:
/// it is captured when the model is trained and read back, never re-derived from the data being scored.
/// </summary>
public sealed record DriftProfile
{
    /// <summary>Format of the profile.</summary>
    public int Version { get; init; } = 1;

    /// <summary>Rows of the training file the profile was built from (a sample when the file is large).</summary>
    public required int Rows { get; init; }

    /// <summary>One entry per profiled feature column.</summary>
    public required IReadOnlyList<FeatureProfile> Features { get; init; }

    /// <summary>The feature columns of <paramref name="schema"/> a drift profile covers: numeric, categorical and boolean features. Free text has no distribution worth comparing bin by bin.</summary>
    public static IReadOnlyList<(string Column, bool Numeric)> ColumnsOf(InputSchemaInfo schema)
        => schema.Columns
            .Where(c => c.Purpose == "Feature")
            .Where(c => c.DataType is SchemaDataTypes.Numeric or SchemaDataTypes.Categorical or SchemaDataTypes.Boolean)
            .Select(c => (c.Name, c.DataType == SchemaDataTypes.Numeric))
            .ToList();

    /// <summary>
    /// Builds a profile from <paramref name="path"/> (a CSV with a header), reading every row once and
    /// keeping a fixed-seed reservoir sample of <paramref name="sampleSize"/> rows — memory stays flat
    /// on a large file, and the same file gives the same profile.
    /// </summary>
    public static async Task<DriftProfile> FromCsvAsync(
        string path, IReadOnlyList<(string Column, bool Numeric)> columns, int sampleSize = 20_000,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(columns);
        var (readPath, detection) = EncodingDetector.ConvertToUtf8WithBom(path);
        try
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                MissingFieldFound = null,
                BadDataFound = null,
                TrimOptions = TrimOptions.Trim,
            };
            using var reader = new StreamReader(readPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            using var csv = new CsvReader(reader, config);
            if (!await csv.ReadAsync().ConfigureAwait(false))
                return Build([], columns);
            csv.ReadHeader();
            var header = csv.HeaderRecord ?? [];
            var index = columns
                .Select(c => Array.FindIndex(header, h => string.Equals(h, c.Column, StringComparison.OrdinalIgnoreCase)))
                .ToArray();

            var random = new Random(20261021);
            var reservoir = new List<string?[]>();
            var seen = 0;
            while (await csv.ReadAsync().ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var row = new string?[columns.Count];
                for (var i = 0; i < columns.Count; i++)
                    row[i] = index[i] >= 0 ? csv.GetField(index[i]) : null;
                seen++;
                if (reservoir.Count < sampleSize)
                    reservoir.Add(row);
                else if (random.Next(seen) is var slot && slot < sampleSize)
                    reservoir[slot] = row;
            }
            return Build(reservoir, columns);
        }
        finally
        {
            if (detection.WasConverted)
                File.Delete(readPath);
        }
    }

    /// <summary>Builds a profile from rows of raw values, one array per row in the order of <paramref name="columns"/>.</summary>
    public static DriftProfile Build(IReadOnlyList<string?[]> rows, IReadOnlyList<(string Column, bool Numeric)> columns)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(columns);
        var features = new List<FeatureProfile>(columns.Count);
        for (var i = 0; i < columns.Count; i++)
        {
            var values = rows.Select(r => r[i]).ToList();
            features.Add(columns[i].Numeric ? FeatureProfile.Numeric(columns[i].Column, values) : FeatureProfile.Categorical(columns[i].Column, values));
        }
        return new DriftProfile { Rows = rows.Count, Features = features };
    }
}

/// <summary>
/// One column's reference distribution: bins and the share of training rows in each, the last share
/// being the missing values'. A value is placed with <see cref="BinOf"/> — the same rule for the
/// training rows and the rows compared against them.
/// </summary>
public sealed record FeatureProfile
{
    /// <summary>Distinct categories kept by name; the rest share one "other" bin.</summary>
    public const int MaxCategories = 20;

    /// <summary>Bins a numeric column is cut into, at most (fewer when values repeat).</summary>
    public const int NumericBins = 10;

    public required string Column { get; init; }

    /// <summary>Numeric (cut by <see cref="Edges"/>) or categorical (matched to <see cref="Categories"/>).</summary>
    public required bool IsNumeric { get; init; }

    /// <summary>Numeric: ascending inclusive upper edges of every bin but the last.</summary>
    public IReadOnlyList<double>? Edges { get; init; }

    /// <summary>
    /// Categorical: the categories kept by name, most frequent first; one more bin holds the rest.
    /// Matched ignoring case — a CSV's <c>true</c> and a JSON log's <c>True</c> are one value.
    /// </summary>
    public IReadOnlyList<string>? Categories { get; init; }

    /// <summary>Share of training rows per bin, the last being missing values. Sums to 1 over a non-empty profile.</summary>
    public required IReadOnlyList<double> Shares { get; init; }

    /// <summary>How many bins there are, the missing bin included.</summary>
    [JsonIgnore]
    public int BinCount => Shares.Count;

    /// <summary>The bin <paramref name="value"/> falls in; the last bin for a missing (empty or unparseable numeric) value.</summary>
    public int BinOf(string? value)
    {
        var missing = BinCount - 1;
        if (string.IsNullOrWhiteSpace(value))
            return missing;
        if (IsNumeric)
        {
            if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) || double.IsNaN(x))
                return missing;
            var edges = Edges ?? [];
            for (var i = 0; i < edges.Count; i++)
                if (x <= edges[i])
                    return i;
            return edges.Count;
        }
        var categories = Categories ?? [];
        for (var i = 0; i < categories.Count; i++)
            if (string.Equals(categories[i], value.Trim(), StringComparison.OrdinalIgnoreCase))
                return i;
        return categories.Count; // other
    }

    internal static FeatureProfile Numeric(string column, IReadOnlyList<string?> values)
    {
        var parsed = new List<double>();
        foreach (var v in values)
            if (!string.IsNullOrWhiteSpace(v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) && !double.IsNaN(x))
                parsed.Add(x);
        parsed.Sort();

        // Inner quantile cut points, without repeats: a value repeated across a cut stays in one bin.
        var edges = new List<double>();
        for (var k = 1; k < NumericBins && parsed.Count > 0; k++)
        {
            var edge = parsed[Math.Min(parsed.Count - 1, (int)Math.Floor(k * parsed.Count / (double)NumericBins))];
            if ((edges.Count == 0 || edge > edges[^1]) && edge < parsed[^1])
                edges.Add(edge);
        }
        var profile = new FeatureProfile { Column = column, IsNumeric = true, Edges = edges, Shares = new double[edges.Count + 2] };
        return profile with { Shares = SharesOf(profile, values) };
    }

    internal static FeatureProfile Categorical(string column, IReadOnlyList<string?> values)
    {
        var categories = values
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .Select(v => v!.Trim())
            .GroupBy(v => v, StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .Take(MaxCategories)
            .Select(g => g.Key)
            .ToList();
        var profile = new FeatureProfile { Column = column, IsNumeric = false, Categories = categories, Shares = new double[categories.Count + 2] };
        return profile with { Shares = SharesOf(profile, values) };
    }

    /// <summary>The share of <paramref name="values"/> in each of this profile's bins.</summary>
    public double[] SharesOf(IReadOnlyCollection<string?> values)
        => SharesOf(this, values);

    private static double[] SharesOf(FeatureProfile profile, IReadOnlyCollection<string?> values)
    {
        var counts = new double[profile.BinCount];
        foreach (var v in values)
            counts[profile.BinOf(v)]++;
        if (values.Count > 0)
            for (var i = 0; i < counts.Length; i++)
                counts[i] /= values.Count;
        return counts;
    }
}
