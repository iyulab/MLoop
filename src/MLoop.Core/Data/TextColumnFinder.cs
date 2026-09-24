using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// Chooses the columns a text model reads (text classification, NER, sentence similarity, question
/// answering).
/// </summary>
/// <remarks>
/// <para>A string column is not necessarily text: a source tag or a category code loads as a
/// string too. Taking the first string columns in order once handed a sentence-similarity model a
/// source tag and one of its two sentences — and the tag correlated with the label well enough that
/// the model scored well and was promoted, having never seen the second sentence.</para>
/// <para>So the choice runs in this order: columns the user declared <c>text</c> in
/// <c>mloop.yaml</c> (in declaration order), then string columns whose values read as language
/// (<see cref="TextLikeness"/>), in schema order. A column declared anything other than text is
/// never chosen. When that still leaves too few, the remaining string columns fill in and the
/// choice is reported as a guess, with how to declare it.</para>
/// </remarks>
public static class TextColumnFinder
{
    /// <summary>Rows read to judge whether a column reads as language.</summary>
    private const int ProbeRows = 1000;

    /// <summary>The single text column for a one-input text model.</summary>
    public static string? FindFirst(
        IDataView data, string labelColumn,
        IReadOnlyDictionary<string, string>? columnOverrides = null, Action<string>? log = null) =>
        Find(data, labelColumn, 1, columnOverrides, log).FirstOrDefault();

    /// <summary>Up to <paramref name="count"/> text columns, most text-like first by the rule above.</summary>
    public static List<string> Find(
        IDataView data, string labelColumn, int count,
        IReadOnlyDictionary<string, string>? columnOverrides = null, Action<string>? log = null)
    {
        var strings = data.Schema
            .Where(c => !c.IsHidden
                        && c.Type is TextDataViewType
                        && !c.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
            .Select(c => c.Name)
            .ToList();

        var declared = new List<string>();
        var declaredOther = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (column, type) in columnOverrides ?? new Dictionary<string, string>())
        {
            var match = strings.FirstOrDefault(s => s.Equals(column, StringComparison.OrdinalIgnoreCase));
            if (match is null)
                continue;
            if (type.Equals("text", StringComparison.OrdinalIgnoreCase))
                declared.Add(match);
            else
                declaredOther.Add(match);
        }

        var chosen = declared.Take(count).ToList();
        if (chosen.Count == count)
            return chosen;

        var undeclared = strings.Where(s => !declared.Contains(s) && !declaredOther.Contains(s)).ToList();
        chosen.AddRange(undeclared.Where(s => ReadsAsLanguage(data, s)).Take(count - chosen.Count));
        if (chosen.Count == count)
            return chosen;

        var guesses = undeclared.Where(s => !chosen.Contains(s)).Take(count - chosen.Count).ToList();
        if (guesses.Count > 0)
        {
            log?.Invoke(
                $"[Warning] No column reads as free text beyond {(chosen.Count == 0 ? "none" : string.Join(", ", chosen))}; " +
                $"using {string.Join(", ", guesses)} as text. To choose them, declare each text column in " +
                "mloop.yaml under columns: <name>: type: text");
            chosen.AddRange(guesses);
        }
        return chosen;
    }

    private static bool ReadsAsLanguage(IDataView data, string column)
    {
        var col = data.Schema[column];
        var sample = new List<string>();
        var distinct = new HashSet<string>(StringComparer.Ordinal);
        var rows = 0;

        using var cursor = data.GetRowCursor([col]);
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);
        ReadOnlyMemory<char> value = default;
        while (rows < ProbeRows && cursor.MoveNext())
        {
            getter(ref value);
            rows++;
            var text = value.ToString().Trim();
            if (text.Length == 0)
                continue;
            distinct.Add(text);
            if (sample.Count < TextLikeness.SampleSize)
                sample.Add(text);
        }

        return TextLikeness.LooksLikeText(sample, distinct.Count, rows);
    }
}
