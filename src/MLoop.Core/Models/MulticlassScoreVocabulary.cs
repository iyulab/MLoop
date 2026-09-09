using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace MLoop.Core.Models;

/// <summary>
/// Which class each slot of a multiclass <c>Score</c> vector belongs to.
/// </summary>
/// <remarks>
/// <para>
/// A multiclass model scores every class at once and returns them as one vector. The vector alone
/// says nothing about which slot is which class — that lives in the scored schema, annotated by the
/// trainer from the label's key values. Reading it here is what lets a prediction report
/// <c>{"cat": 0.9, "dog": 0.1}</c> instead of <c>{"class_0": 0.9, "class_1": 0.1}</c>, where the
/// consumer has to guess the mapping and can only guess wrong.
/// </para>
/// <para>
/// Two annotations carry it, and both are read, because <b>which one exists depends on the label's
/// type</b>. A categorical label leaves <c>SlotNames</c> (text) on the Score column; a
/// numeric-looking label — class ids <c>0</c>/<c>1</c>/<c>2</c> — leaves <b>no <c>SlotNames</c> at
/// all</b>, only <c>TrainingLabelValues</c> typed <c>Single</c>. Measured on both shapes: reading
/// <c>SlotNames</c> alone would have named the classes for text labels and quietly kept positional
/// keys for numeric ones, which is the same document-disagrees-with-itself defect one task type
/// down. Where both exist they hold the same values in the same order, so preferring the
/// per-slot-specific one costs nothing.
/// </para>
/// <para>
/// This is a different question from <see cref="BinaryLabelVocabulary"/>'s, and deliberately reads a
/// different source. Binary classification has no Score vector — it returns one scalar probability,
/// and which of the two original values counts as positive is a decision the *training* conversion
/// made, so it is recovered from the saved schema. Multiclass makes no such decision: the slot order
/// is the trainer's, and only the trainer's own output schema records it. Deriving it from the saved
/// schema's value list instead would be a second implementation of an ordering ML.NET already owns —
/// and one that disagrees the moment the two sort differently.
/// </para>
/// <para>
/// The numeric form is rendered with the invariant culture — the same rendering the predicted label
/// itself gets, so a row's label and its probability keys are drawn from one alphabet.
/// </para>
/// </remarks>
public static class MulticlassScoreVocabulary
{
    /// <summary>
    /// The Score-column annotations that name the classes, in the order they are consulted.
    /// <c>SlotNames</c> answers "what is slot i" directly; <c>TrainingLabelValues</c> is the label
    /// vocabulary the trainer was fit on, in key order, and is the only one a numeric-looking label
    /// leaves behind.
    /// </summary>
    private static readonly string[] VocabularyAnnotations = { "SlotNames", "TrainingLabelValues" };

    /// <summary>
    /// The class names for a Score vector's slots, or null when the scored schema does not carry a
    /// usable one — in which case the caller keeps positional keys rather than inventing names.
    /// </summary>
    /// <param name="scoreColumn">The scored view's <c>Score</c> column, or null.</param>
    /// <param name="slotCount">How many slots the row actually produced.</param>
    /// <remarks>
    /// An annotation is skipped (and the next one tried) whenever it is missing, is neither of the
    /// two element types, disagrees with <paramref name="slotCount"/>, or does not name every slot
    /// distinctly. The last case matters most: these names become dictionary keys, so two equal names
    /// would collapse two classes into one entry and silently drop a probability.
    /// </remarks>
    public static IReadOnlyList<string>? Of(DataViewSchema.Column? scoreColumn, int slotCount)
    {
        if (scoreColumn is not { } column || slotCount <= 0)
            return null;

        foreach (var annotationName in VocabularyAnnotations)
        {
            var names = Read(column, annotationName);
            if (names is not null && IsUsable(names, slotCount))
                return names;
        }

        return null;
    }

    private static string[]? Read(DataViewSchema.Column column, string annotationName)
    {
        var annotation = column.Annotations.Schema.GetColumnOrNull(annotationName);
        if (annotation is null || annotation.Value.Type is not VectorDataViewType vector)
            return null;

        if (vector.ItemType == TextDataViewType.Instance)
        {
            VBuffer<ReadOnlyMemory<char>> buffer = default;
            column.Annotations.GetValue(annotationName, ref buffer);
            return buffer.DenseValues().Select(v => v.ToString()).ToArray();
        }

        if (vector.ItemType == NumberDataViewType.Single)
        {
            VBuffer<float> buffer = default;
            column.Annotations.GetValue(annotationName, ref buffer);
            return buffer.DenseValues().Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray();
        }

        return null;
    }

    private static bool IsUsable(string[] names, int slotCount) =>
        names.Length == slotCount
        && !names.Any(string.IsNullOrWhiteSpace)
        && names.Distinct(StringComparer.Ordinal).Count() == names.Length;

    /// <summary>
    /// The key for slot <paramref name="index"/>: its class name when one is known, and the
    /// positional <c>class_{i}</c> otherwise. The fallback is what older models and unannotated
    /// shapes keep producing, so a consumer that has been reading positional keys still gets them.
    /// </summary>
    public static string KeyFor(IReadOnlyList<string>? names, int index) =>
        names is not null && index >= 0 && index < names.Count
            ? names[index]
            : $"class_{index}";
}
