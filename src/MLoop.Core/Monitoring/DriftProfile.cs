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

    /// <summary>
    /// One entry per free-text feature column — described by what a bin-by-bin comparison can read of
    /// it (<see cref="TextProfile"/>). <c>null</c> for profiles written before text was profiled.
    /// </summary>
    public IReadOnlyList<TextProfile>? Texts { get; init; }

    /// <summary>
    /// How the model's own answers were distributed over the same rows — the predicted class for a
    /// classifier, the predicted value for a regressor — in the form a prediction log records them
    /// (<see cref="Prediction.PredictionLogOutput"/>). <c>null</c> for a task whose answer is neither,
    /// and for profiles written before answers were profiled.
    /// </summary>
    public FeatureProfile? Prediction { get; init; }

    /// <summary>The name <see cref="Prediction"/> is reported under — no input column can carry it.</summary>
    public const string PredictionColumn = "(prediction)";

    /// <summary>
    /// The answer profile of <paramref name="answers"/> — the logged output of each scored row — or
    /// <c>null</c> when <paramref name="task"/>'s answer is not a class or a value.
    /// </summary>
    public static FeatureProfile? PredictionProfileOf(string? task, IReadOnlyCollection<object?> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);
        if (!HasAnswerProfile(task))
            return null;
        var values = answers.Select(DriftMeasure.Text).ToList();
        return Training.TrainingRunner.IsClassification(task)
            ? FeatureProfile.Categorical(PredictionColumn, values)
            : FeatureProfile.Numeric(PredictionColumn, values);
    }

    /// <summary>Whether <paramref name="task"/>'s answer is a class or a value — the answers an answer profile describes.</summary>
    public static bool HasAnswerProfile(string? task)
        => Training.TrainingRunner.IsClassification(task) || TaskTypes.Canonical(task) == "regression";

    /// <summary>The feature columns of <paramref name="schema"/> a drift profile covers: numeric, categorical and boolean features. Free text is profiled apart (<see cref="TextColumnsOf"/>).</summary>
    public static IReadOnlyList<(string Column, bool Numeric)> ColumnsOf(InputSchemaInfo schema)
        => schema.Columns
            .Where(c => c.Purpose == "Feature")
            .Where(c => c.DataType is SchemaDataTypes.Numeric or SchemaDataTypes.Categorical or SchemaDataTypes.Boolean)
            .Select(c => (c.Name, c.DataType == SchemaDataTypes.Numeric))
            .ToList();

    /// <summary>The free-text feature columns of <paramref name="schema"/> — profiled by <see cref="TextProfile"/>, not bin by bin.</summary>
    public static IReadOnlyList<string> TextColumnsOf(InputSchemaInfo schema)
        => schema.Columns
            .Where(c => c.Purpose == "Feature" && c.DataType == SchemaDataTypes.Text)
            .Select(c => c.Name)
            .ToList();

    /// <summary>
    /// The profile of <paramref name="sample"/>'s rows for the feature columns of <paramref name="schema"/>:
    /// numeric, categorical and boolean ones bin by bin (<see cref="ColumnsOf"/>), free text by
    /// <see cref="TextProfile"/> (<see cref="TextColumnsOf"/>).
    /// </summary>
    public static DriftProfile Of(CsvSample sample, InputSchemaInfo schema)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(schema);
        var columns = ColumnsOf(schema);
        var texts = TextColumnsOf(schema);
        var textRows = sample.Project(texts);
        return Build(sample.Project(columns.Select(c => c.Column).ToList()), columns) with
        {
            Texts = texts.Select((column, i) => TextProfile.Build(column, textRows.Select(r => r[i]).ToList())).ToList(),
        };
    }

    /// <summary>
    /// Builds a profile from <paramref name="path"/> (a CSV with a header), reading every row once and
    /// keeping a fixed-seed reservoir sample of <paramref name="sampleSize"/> rows — memory stays flat
    /// on a large file, and the same file gives the same profile.
    /// </summary>
    public static async Task<DriftProfile> FromCsvAsync(
        string path, IReadOnlyList<(string Column, bool Numeric)> columns, int sampleSize = DefaultSampleSize,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(columns);
        var sample = await SampleCsvAsync(path, sampleSize, cancellationToken).ConfigureAwait(false);
        return Build(sample.Project(columns.Select(c => c.Column).ToList()), columns);
    }

    /// <summary>Rows of the training file a profile is built from, at most.</summary>
    public const int DefaultSampleSize = 20_000;

    /// <summary>
    /// Reads every row of <paramref name="path"/> (a CSV with a header) once and keeps a fixed-seed
    /// reservoir sample of <paramref name="sampleSize"/> whole rows — memory stays flat on a large
    /// file, and the same file gives the same sample. The rows an input profile is built from and the
    /// rows the model is scored on for its answer profile are this one sample.
    /// </summary>
    public static async Task<CsvSample> SampleCsvAsync(string path, int sampleSize = DefaultSampleSize, CancellationToken cancellationToken = default)
    {
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
                return new CsvSample([], []);
            csv.ReadHeader();
            var header = csv.HeaderRecord ?? [];

            var random = new Random(20261021);
            var reservoir = new List<string?[]>();
            var seen = 0;
            while (await csv.ReadAsync().ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var row = new string?[header.Length];
                for (var i = 0; i < header.Length; i++)
                    row[i] = csv.TryGetField<string>(i, out var field) ? field : null;
                seen++;
                if (reservoir.Count < sampleSize)
                    reservoir.Add(row);
                else if (random.Next(seen) is var slot && slot < sampleSize)
                    reservoir[slot] = row;
            }
            return new CsvSample(header, reservoir);
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
/// A free-text column's reference, read through two numeric descriptors a bin-by-bin comparison can
/// hold: each value's length in characters (<see cref="Length"/>), and the share of its words not in
/// the training vocabulary (<see cref="Unseen"/>) — the signal that the text is about something the
/// model never read. Both are ordinary numeric <see cref="FeatureProfile"/>s, so drift, slices and
/// <c>input_drift</c> treat them as two more columns, named <see cref="LengthColumn"/> and
/// <see cref="UnseenColumn"/>.
/// </summary>
/// <remarks>
/// The vocabulary is built from every other sampled row and the unseen-share reference from the rows
/// in between. Built and measured on the same rows, every training row would read 0 % unseen and any
/// live text would look like drift; split, the reference is the share a row the vocabulary did not
/// come from shows — which is what a live row is.
/// </remarks>
public sealed record TextProfile
{
    /// <summary>Words kept in the vocabulary, most frequent first.</summary>
    public const int MaxVocabulary = 10_000;

    /// <summary>
    /// Where <see cref="Unseen"/> is cut: none unseen, then rising shares. Fixed rather than the
    /// reference's deciles — the reference is usually almost all 0 %, and deciles of a near-constant
    /// column leave one bin that every later share falls into, so text full of new words would read
    /// as no change.
    /// </summary>
    public static readonly IReadOnlyList<double> UnseenEdges = [0, 0.1, 0.25, 0.5, 0.75];

    /// <summary>The text column.</summary>
    public required string Column { get; init; }

    /// <summary>The training vocabulary: lower-case words, most frequent first (<see cref="Words"/>).</summary>
    public required IReadOnlyList<string> Vocabulary { get; init; }

    /// <summary>Length in characters of each value.</summary>
    public required FeatureProfile Length { get; init; }

    /// <summary>Share of each value's words not in <see cref="Vocabulary"/>; a value without words is missing.</summary>
    public required FeatureProfile Unseen { get; init; }

    /// <summary>The name <see cref="Length"/> is reported under.</summary>
    public static string LengthColumn(string column) => $"{column} (length)";

    /// <summary>The name <see cref="Unseen"/> is reported under.</summary>
    public static string UnseenColumn(string column) => $"{column} (unseen words)";

    /// <summary>The profile of <paramref name="values"/>, a text column's sampled values.</summary>
    public static TextProfile Build(string column, IReadOnlyList<string?> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var vocabulary = values.Where((_, i) => i % 2 == 0)
            .SelectMany(v => Words(v))
            .GroupBy(w => w, StringComparer.Ordinal)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .Take(MaxVocabulary)
            .Select(g => g.Key)
            .ToList();
        var known = vocabulary.ToHashSet(StringComparer.Ordinal);
        var reference = values.Count > 1 ? values.Where((_, i) => i % 2 == 1).ToList() : values;
        return new TextProfile
        {
            Column = column,
            Vocabulary = vocabulary,
            Length = FeatureProfile.Numeric(LengthColumn(column), values.Select(LengthOf).ToList()),
            Unseen = FeatureProfile.Numeric(UnseenColumn(column), UnseenEdges, reference.Select(v => UnseenShareOf(v, known)).ToList()),
        };
    }

    /// <summary>The descriptors of <paramref name="values"/>, as the raw values <see cref="Length"/> and <see cref="Unseen"/> bin.</summary>
    public (IReadOnlyList<string?> Lengths, IReadOnlyList<string?> UnseenShares) Describe(IReadOnlyCollection<string?> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var known = Vocabulary.ToHashSet(StringComparer.Ordinal);
        return (values.Select(LengthOf).ToList(), values.Select(v => UnseenShareOf(v, known)).ToList());
    }

    /// <summary>
    /// The words of <paramref name="text"/>, lower-cased: runs of letters and digits. A Han, kana or
    /// Thai character is a word on its own — those scripts do not separate words with spaces, and a
    /// whole unspaced sentence read as one word would be unseen every time.
    /// </summary>
    public static IEnumerable<string> Words(string? text)
    {
        if (string.IsNullOrEmpty(text))
            yield break;
        var start = -1;
        for (var i = 0; i <= text.Length; i++)
        {
            var c = i < text.Length ? text[i] : ' ';
            if (char.IsLetterOrDigit(c) && !IsUnspacedScript(c))
            {
                if (start < 0)
                    start = i;
                continue;
            }
            if (start >= 0)
            {
                yield return text[start..i].ToLowerInvariant();
                start = -1;
            }
            if (i < text.Length && char.IsLetter(c))
                yield return c.ToString();
        }
    }

    private static bool IsUnspacedScript(char c)
        => c is >= '\u3040' and <= '\u30FF'   // hiragana, katakana
            or >= '\u3400' and <= '\u4DBF'    // CJK extension A
            or >= '\u4E00' and <= '\u9FFF'    // CJK unified ideographs
            or >= '\u0E00' and <= '\u0E7F';   // Thai

    private static string? LengthOf(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim().Length.ToString(CultureInfo.InvariantCulture);

    private static string? UnseenShareOf(string? value, HashSet<string> known)
    {
        var words = Words(value).ToList();
        return words.Count == 0 ? null
            : (words.Count(w => !known.Contains(w)) / (double)words.Count).ToString("R", CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// Whole rows drawn from a CSV file — its header, and the sampled rows in reservoir order. Any subset
/// of the rows is itself a uniform sample of the file.
/// </summary>
public sealed record CsvSample(IReadOnlyList<string> Header, IReadOnlyList<string?[]> Rows)
{
    /// <summary>The rows cut down to <paramref name="columns"/>, in that order; a column the file lacks is missing in every row.</summary>
    public IReadOnlyList<string?[]> Project(IReadOnlyList<string> columns)
    {
        var index = columns
            .Select(c => Header.ToList().FindIndex(h => string.Equals(h, c, StringComparison.OrdinalIgnoreCase)))
            .ToArray();
        return Rows.Select(r => index.Select(i => i >= 0 && i < r.Length ? r[i] : null).ToArray()).ToList();
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
    /// Numeric: the smallest and largest value the profile was built from (single precision, as compared). The bins are open at both ends, so
    /// without these a value far past the training range sits in the end bin beside the rows that were there — and when the largest
    /// value is tied across many rows, a whole window of new values past it moves that bin's share very little. Absent from profiles
    /// written before; those read as they did.
    /// </summary>
    public double? Min { get; init; }

    /// <inheritdoc cref="Min"/>
    public double? Max { get; init; }

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
            x = AsSingle(x);
            for (var i = 0; i < edges.Count; i++)
                if (x <= AsSingle(edges[i]))
                    return i;
            return edges.Count;
        }
        var categories = Categories ?? [];
        var text = value.Trim();
        for (var i = 0; i < categories.Count; i++)
            if (string.Equals(categories[i], text, StringComparison.OrdinalIgnoreCase))
                return i;
        // In a yes/no column, a value written another way — training's CSV said "yes", a JSON request sends
        // true — is the same value: the loader reads both as one boolean, and so does the model.
        if (categories.Count > 0 && Data.ColumnLayoutValues.TryReadBoolean(text, out var truth)
            && categories.All(c => Data.ColumnLayoutValues.TryReadBoolean(c, out _)))
            for (var i = 0; i < categories.Count; i++)
                if (Data.ColumnLayoutValues.TryReadBoolean(categories[i], out var category) && category == truth)
                    return i;
        return categories.Count; // other
    }

    /// <summary>
    /// A number as the model reads it. The loaders read every numeric input as a single-precision float, so a training file's
    /// <c>0.999999993939</c> is the <c>1</c> that the prediction log records, and <c>15.0013</c> written as <c>15.001299999999999</c> is the
    /// same value as <c>15.0013</c>. Cut points and values are compared after this rounding: compared in double precision, a column
    /// whose values differ below single precision read as drifted on rows drawn from the training data.
    /// </summary>
    internal static double AsSingle(double value) => (float)value;

    internal static FeatureProfile Numeric(string column, IReadOnlyList<string?> values)
    {
        var parsed = new List<double>();
        foreach (var v in values)
            if (!string.IsNullOrWhiteSpace(v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) && !double.IsNaN(x))
                parsed.Add(AsSingle(x));
        parsed.Sort();

        // Inner quantile cut points, without repeats: a value repeated across a cut stays in one bin.
        var edges = new List<double>();
        for (var k = 1; k < NumericBins && parsed.Count > 0; k++)
        {
            var edge = parsed[Math.Min(parsed.Count - 1, (int)Math.Floor(k * parsed.Count / (double)NumericBins))];
            if ((edges.Count == 0 || edge > edges[^1]) && edge < parsed[^1])
                edges.Add(edge);
        }
        var profile = new FeatureProfile
        {
            Column = column,
            IsNumeric = true,
            Edges = edges,
            Min = parsed.Count > 0 ? parsed[0] : null,
            Max = parsed.Count > 0 ? parsed[^1] : null,
            Shares = new double[edges.Count + 2],
        };
        return profile with { Shares = SharesOf(profile, values) };
    }

    /// <summary>A numeric profile cut at the given <paramref name="edges"/> (ascending inclusive upper edges) instead of the values' deciles.</summary>
    internal static FeatureProfile Numeric(string column, IReadOnlyList<double> edges, IReadOnlyList<string?> values)
    {
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

    /// <summary>
    /// The shares of <paramref name="values"/> (all of them, missing included) below the training minimum and above the training maximum;
    /// zero for both when the profile has no range or the column is not numeric.
    /// </summary>
    public (double Below, double Above) OutsideRange(IReadOnlyCollection<string?> values)
    {
        if (!IsNumeric || Min is not { } lo || Max is not { } hi || values.Count == 0)
            return (0, 0);
        double below = 0, above = 0;
        foreach (var v in values)
        {
            if (string.IsNullOrWhiteSpace(v) || !double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) || double.IsNaN(x))
                continue;
            x = AsSingle(x);
            if (x < lo)
                below++;
            else if (x > hi)
                above++;
        }
        return (below / values.Count, above / values.Count);
    }

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
