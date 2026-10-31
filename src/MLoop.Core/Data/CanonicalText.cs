using System.Text;

namespace MLoop.Core.Data;

/// <summary>
/// Text that is the same under Unicode canonical equivalence: Hangul written as one syllable (NFC, <c>가</c>) or as its
/// jamo (NFD, <c>ᄀ</c> + <c>ᅡ</c>) is the same text to a reader and different bytes to a comparison.
/// </summary>
/// <remarks>
/// A file saved on macOS, or exported by a tool that decomposes, arrives in NFD; the model was trained on NFC, or the
/// other way round. Compared as bytes, a category <c>서울</c> is then a value training never saw and is replaced by the
/// most frequent one, and a column <c>지역</c> is "missing" from a file that visibly has it. Both are the same text, so
/// they are matched as text: a category through <see cref="Resolve"/>, a column name through <see cref="SameText"/>.
/// </remarks>
public static class CanonicalText
{
    /// <summary><paramref name="text"/> in the composed form (NFC); itself when it already is.</summary>
    public static string Nfc(string text) =>
        text.IsNormalized(NormalizationForm.FormC) ? text : text.Normalize(NormalizationForm.FormC);

    /// <summary>Whether the two differ as strings and are equal as text — written in different Unicode forms.</summary>
    public static bool SameText(string a, string b) =>
        !string.Equals(a, b, StringComparison.Ordinal) && string.Equals(Nfc(a), Nfc(b), StringComparison.Ordinal);

    /// <summary>The trained values by their composed form, each as the training data wrote it (the first, should two collide).</summary>
    public static Dictionary<string, string> LookupByNfc(IEnumerable<string> trained)
    {
        var lookup = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var value in trained)
            lookup.TryAdd(Nfc(value), value);
        return lookup;
    }

    /// <summary>
    /// The trained spelling of <paramref name="value"/>: itself when training saw it, the trained value that is the same
    /// text in another Unicode form, or null when it is a value training never saw.
    /// </summary>
    public static string? Resolve(ISet<string> trained, IReadOnlyDictionary<string, string> trainedByNfc, string value)
    {
        if (trained.Contains(value))
            return value;
        return trainedByNfc.TryGetValue(Nfc(value), out var match) ? match : null;
    }
}
