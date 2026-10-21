using System.Globalization;
using MLoop.Core.Prediction;

namespace MLoop.Core.Data;

/// <summary>
/// Which string feature columns hold categories — codes, levels, names from a short list — rather
/// than language. A category is one-hot encoded; language is featurized as text (word and character
/// n-grams).
/// </summary>
/// <remarks>
/// <para>
/// Training featurized every string column as text, so a column like <c>sex</c> or
/// <c>workclass</c> went through TF-IDF n-grams while the saved schema recorded it as
/// <c>Categorical</c> — two answers to one question. Text featurization of a category is not wrong
/// so much as roundabout (a one-token "document" per row), but it is also the one step ML.NET cannot
/// export to ONNX, so any model with a category column could not leave ML.NET.
/// </para>
/// <para>
/// The decision is made once per run, from the whole training file, like the feature exclusions
/// (<see cref="CsvDataLoader.DetermineExcludedColumns"/>): deciding per partition would let a column
/// be a category in one slice and text in another. <see cref="TextLikeness"/> says whether values read
/// as language; a column that does not, and has few enough distinct values to enumerate, is a
/// category.
/// </para>
/// </remarks>
public static class CategoryColumns
{
    /// <summary>A column with at most this many distinct values is small enough to one-hot encode…</summary>
    public const int MaxCategories = 100;

    /// <summary>…or one whose distinct values are under this share of its rows.</summary>
    public const double MaxDistinctShare = 0.1;

    // Distinct values are counted up to here; past it no rule can call the column a category.
    private const int DistinctCountCap = 100_000;

    /// <summary>Whether a string column with these values is a category rather than text.</summary>
    /// <param name="sampleValues">Non-empty values in row order (the first <see cref="TextLikeness.SampleSize"/> are read).</param>
    /// <param name="uniqueCount">Distinct non-empty values.</param>
    /// <param name="rowCount">Rows the count was taken over.</param>
    public static bool IsCategory(IReadOnlyList<string> sampleValues, int uniqueCount, int rowCount)
    {
        if (rowCount == 0 || uniqueCount == 0)
            return false;
        if (TextLikeness.LooksLikeText(sampleValues, uniqueCount, rowCount))
            return false;
        return uniqueCount <= MaxCategories || uniqueCount < rowCount * MaxDistinctShare;
    }

    /// <summary>
    /// The string columns of <paramref name="csvPath"/> that are categories, in header order.
    /// </summary>
    /// <param name="csvPath">The training data (CSV with a header).</param>
    /// <param name="skip">
    /// Columns not to judge — the label, columns excluded from features, columns the user typed.
    /// </param>
    /// <remarks>
    /// A column is a string column when some non-empty value does not parse as a number; a column of
    /// numbers is not a candidate however few values it has (the loader reads it as a number).
    /// </remarks>
    public static IReadOnlyList<string> Decide(string csvPath, IEnumerable<string>? skip = null)
    {
        var skipped = new HashSet<string>(skip ?? [], StringComparer.OrdinalIgnoreCase);

        var encoding = EncodingDetector.DetectEncoding(csvPath).Encoding;
        using var reader = new StreamReader(csvPath, encoding, detectEncodingFromByteOrderMarks: true);
        var headerLine = reader.ReadLine();
        if (string.IsNullOrEmpty(headerLine))
            return [];

        var header = CsvFieldParser.ParseFields(headerLine);
        var candidates = Enumerable.Range(0, header.Length)
            .Where(i => !skipped.Contains(header[i].Trim()))
            .ToArray();
        var stats = candidates.ToDictionary(i => i, _ => new ColumnStats());

        var rows = 0;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.Length == 0)
                continue;
            rows++;
            var fields = CsvFieldParser.ParseFields(line);
            foreach (var i in candidates)
            {
                if (i >= fields.Length)
                    continue;
                var value = fields[i].Trim();
                if (value.Length == 0)
                    continue;
                var s = stats[i];
                if (!s.HasText && !double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                    s.HasText = true;
                if (s.Sample.Count < TextLikeness.SampleSize)
                    s.Sample.Add(value);
                if (s.Distinct.Count < DistinctCountCap)
                    s.Distinct.Add(value);
            }
        }

        return candidates
            .Where(i => stats[i].HasText && IsCategory(stats[i].Sample, stats[i].Distinct.Count, rows))
            .Select(i => header[i].Trim())
            .ToList();
    }

    private sealed class ColumnStats
    {
        public bool HasText;
        public readonly List<string> Sample = [];
        public readonly HashSet<string> Distinct = new(StringComparer.Ordinal);
    }
}
