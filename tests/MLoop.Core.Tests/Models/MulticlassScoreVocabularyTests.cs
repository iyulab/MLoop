using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The Score vector's slot→class mapping, read straight off an annotated schema. These build the
/// annotation by hand rather than training a model, so each rejection case (wrong width, duplicate
/// name, unusable element type) can be stated exactly; the trained-model end of the same behavior —
/// including the fact that a numeric-looking label leaves no <c>SlotNames</c> at all — is pinned in
/// <c>PredictionServiceTests</c>.
/// </summary>
public class MulticlassScoreVocabularyTests
{
    private static DataViewSchema.Column ScoreColumn(int size, Action<DataViewSchema.Annotations.Builder>? annotate = null)
    {
        var annotations = new DataViewSchema.Annotations.Builder();
        annotate?.Invoke(annotations);

        var schema = new DataViewSchema.Builder();
        schema.AddColumn("Score", new VectorDataViewType(NumberDataViewType.Single, size), annotations.ToAnnotations());
        return schema.ToSchema()["Score"];
    }

    private static Action<DataViewSchema.Annotations.Builder> Text(string name, params string[] values) =>
        builder =>
        {
            var buffer = new VBuffer<ReadOnlyMemory<char>>(values.Length, values.Select(v => v.AsMemory()).ToArray());
            builder.Add<VBuffer<ReadOnlyMemory<char>>>(
                name,
                new VectorDataViewType(TextDataViewType.Instance, values.Length),
                (ref VBuffer<ReadOnlyMemory<char>> dst) => buffer.CopyTo(ref dst));
        };

    private static Action<DataViewSchema.Annotations.Builder> Numeric(string name, params float[] values) =>
        builder =>
        {
            var buffer = new VBuffer<float>(values.Length, values);
            builder.Add<VBuffer<float>>(
                name,
                new VectorDataViewType(NumberDataViewType.Single, values.Length),
                (ref VBuffer<float> dst) => buffer.CopyTo(ref dst));
        };

    [Fact]
    public void Of_TextSlotNames_ReturnsTheClassNames()
    {
        var column = ScoreColumn(3, Text("SlotNames", "cat", "dog", "bird"));

        Assert.Equal(new[] { "cat", "dog", "bird" }, MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void Of_NumericTrainingLabelValues_RendersInvariantly()
    {
        // The measured shape for a numeric-looking label ("0"/"1"/"2"): the trainer leaves NO SlotNames
        // on the Score column, only TrainingLabelValues typed Single. Reading SlotNames alone would
        // keep positional keys here while the predicted label reads "0" — the same document
        // disagreeing with itself that naming the keys exists to remove.
        var column = ScoreColumn(3, Numeric("TrainingLabelValues", 0f, 1f, 2f));

        Assert.Equal(new[] { "0", "1", "2" }, MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void Of_SlotNamesWinsOverTrainingLabelValues()
    {
        // Both exist for a categorical label and hold the same values; the per-slot-specific one is
        // the one that answers "what is slot i" directly, so it is consulted first.
        var column = ScoreColumn(3, builder =>
        {
            Text("SlotNames", "cat", "dog", "bird")(builder);
            Text("TrainingLabelValues", "x", "y", "z")(builder);
        });

        Assert.Equal(new[] { "cat", "dog", "bird" }, MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void Of_UnusableSlotNames_FallsThroughToTrainingLabelValues()
    {
        // A rejected first annotation must not mask a usable second one — otherwise a duplicate slot
        // name would drop the class vocabulary the model still carries elsewhere.
        var column = ScoreColumn(3, builder =>
        {
            Text("SlotNames", "cat", "cat", "bird")(builder);
            Text("TrainingLabelValues", "cat", "dog", "bird")(builder);
        });

        Assert.Equal(new[] { "cat", "dog", "bird" }, MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void Of_NoAnnotation_ReturnsNull()
    {
        Assert.Null(MulticlassScoreVocabulary.Of(ScoreColumn(3), 3));
    }

    [Fact]
    public void Of_NullColumn_ReturnsNull()
    {
        Assert.Null(MulticlassScoreVocabulary.Of(null, 3));
    }

    [Fact]
    public void Of_WidthDisagreesWithTheRow_ReturnsNull()
    {
        // Naming three slots when the row produced four would pair every name with a slot it may not
        // belong to — worse than saying nothing.
        var column = ScoreColumn(3, Text("SlotNames", "cat", "dog", "bird"));

        Assert.Null(MulticlassScoreVocabulary.Of(column, 4));
    }

    [Fact]
    public void Of_DuplicateNames_ReturnsNull()
    {
        // These names become dictionary keys: two equal names collapse two classes into one entry and
        // drop a probability without a word.
        var column = ScoreColumn(3, Text("SlotNames", "cat", "dog", "cat"));

        Assert.Null(MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void Of_BlankName_ReturnsNull()
    {
        var column = ScoreColumn(3, Text("SlotNames", "cat", "", "bird"));

        Assert.Null(MulticlassScoreVocabulary.Of(column, 3));
    }

    [Fact]
    public void KeyFor_WithoutNames_KeepsThePositionalKey()
    {
        Assert.Equal("class_0", MulticlassScoreVocabulary.KeyFor(null, 0));
        Assert.Equal("class_7", MulticlassScoreVocabulary.KeyFor(null, 7));
    }

    [Fact]
    public void KeyFor_IndexOutsideTheNames_FallsBackPositionally()
    {
        var names = new[] { "cat", "dog" };

        Assert.Equal("cat", MulticlassScoreVocabulary.KeyFor(names, 0));
        Assert.Equal("class_5", MulticlassScoreVocabulary.KeyFor(names, 5));
    }
}
