namespace MLoop.Core.Models;

/// <summary>
/// Which of a binary label's two original values <c>false</c> means, and which <c>true</c> means.
/// </summary>
/// <remarks>
/// <para>
/// ML.NET's binary classification takes a Boolean label, so a label written as <c>OK</c>/<c>NG</c>
/// is converted before training — and the conversion has to decide which value becomes the positive
/// class. That decision is the whole content of this type: it is made once here, and both the
/// conversion and everything that renders a prediction afterwards read it from the same place.
/// </para>
/// <para>
/// Before this existed, the rule lived only in the converter and nothing undid it. A model trained
/// on <c>OK</c>/<c>NG</c> predicted correctly and then reported <c>1</c>/<c>0</c> in CSV and
/// <c>True</c>/<c>False</c> in JSON — the class names the user wrote were absent from both answers,
/// and the two answers did not even agree with each other. The saved schema had the vocabulary all
/// along.
/// </para>
/// <para>
/// The ordering is case-insensitive, which is what the converter has always used; the schema's own
/// <c>CategoricalValues</c> list is sorted by a different comparer, so reading position 0 out of it
/// is not the same question and would disagree on a vocabulary like <c>a</c>/<c>B</c>. Comparing
/// against a recorded model's meaning is exactly why the rule cannot be restated in two places.
/// </para>
/// </remarks>
public static class BinaryLabelVocabulary
{
    /// <summary>
    /// The two values in the order the conversion assigns them: negative (<c>false</c>) first.
    /// Null when the values are not a binary vocabulary.
    /// </summary>
    public static (string Negative, string Positive)? Order(IEnumerable<string>? values)
    {
        if (values == null)
            return null;

        var distinct = values
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .Select(v => v.Trim())
            .Distinct(StringComparer.Ordinal)
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        return distinct.Length == 2 ? (distinct[0], distinct[1]) : null;
    }

    /// <summary>The same, read off a saved schema's label column.</summary>
    public static (string Negative, string Positive)? Of(InputSchemaInfo? schema)
    {
        var label = schema?.Columns.FirstOrDefault(c => c.Purpose == "Label");
        return Order(label?.CategoricalValues);
    }

    /// <summary>
    /// How a predicted Boolean should read. Falls back to <c>True</c>/<c>False</c> when the model
    /// carries no vocabulary — an older experiment, or a label that was already Boolean in the data
    /// — so that the two prediction paths still render a Boolean the same way as each other.
    /// </summary>
    public static string Render(bool predicted, (string Negative, string Positive)? vocabulary) =>
        vocabulary is { } v
            ? (predicted ? v.Positive : v.Negative)
            : (predicted ? "True" : "False");
}
