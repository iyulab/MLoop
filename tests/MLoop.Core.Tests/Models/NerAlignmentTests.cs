using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// ML.NET's NER transformer writes its answer one position to the right: the first entry is the start token's own
/// prediction, each word's slot holds the tag of the word before it, and the last word's tag is not written. A model
/// that tagged every word right scored 0.28 of its words by position (0.82 one position over) and the gate refused
/// it. These tests stand a transformer with exactly that output shape in for the real one (the real one needs a
/// libtorch runtime and ten minutes) and read it through <see cref="NerAlignment"/>.
/// </summary>
public class NerAlignmentTests
{
    private static readonly Dictionary<string, string> TagOf = new(StringComparer.Ordinal)
    {
        ["Alice"] = "B-PER",
        ["Paris"] = "B-LOC",
        ["Monday"] = "B-DATE",
    };

    [Fact]
    public void TheStandInTransformer_AnswersOnePositionToTheRight()
    {
        // The shape the real transformer was measured to have — if this stops holding the tests below say nothing.
        var answer = Shifted(new MLContext(1), "Alice flew to Paris").Single();

        Assert.Equal(["O", "B-PER", "O", "O"], answer);
    }

    [Fact]
    public void AnAlignedModel_TagsEveryWord_AtItsOwnPosition_TheLastOneIncluded()
    {
        var ml = new MLContext(1);
        var sentences = new[] { "Alice flew to Paris", "Monday", "Alice lives in Paris on Monday" };

        var answers = Aligned(ml, sentences);

        Assert.Equal(["B-PER", "O", "O", "B-LOC"], answers[0]);
        Assert.Equal(["B-DATE"], answers[1]);
        Assert.Equal(["B-PER", "O", "O", "B-LOC", "O", "B-DATE"], answers[2]);
        for (var i = 0; i < sentences.Length; i++)
            Assert.Equal(sentences[i].Split(' ').Length, answers[i].Length);
    }

    [Fact]
    public void AnAlignedModel_GivesTheSentenceBackAsItWas()
    {
        var ml = new MLContext(1);
        var model = NerAlignment.Apply(ml, Stand(ml), "Sentence");
        var scored = model.Transform(ml.Data.LoadFromEnumerable(new[]
        {
            new Input { Sentence = "Alice flew to Paris" }, new Input { Sentence = "ends with a dot ." }, new Input { Sentence = "" },
        }));

        Assert.Equal(["Alice flew to Paris", "ends with a dot .", ""],
            scored.GetColumn<ReadOnlyMemory<char>>("Sentence").Select(t => t.ToString()));
    }

    [Fact]
    public void AnEmptySentence_IsAnsweredWithNoTags_NotAnError()
    {
        Assert.Empty(Aligned(new MLContext(1), [""]).Single());
    }

    [Theory]
    [InlineData("Alice flew to Paris")]
    [InlineData("")]
    [InlineData("already ends with a dot .")]
    public void APaddedSentence_IsGivenBackAsItWas(string sentence)
    {
        Assert.Equal(sentence + " .", TextPadding.Padded(sentence));
        Assert.Equal(sentence, TextPadding.Unpadded(TextPadding.Padded(sentence)));
    }

    [Fact]
    public void Aligned_DropsTheStartTokensAnswer_OnlyTheFirst()
    {
        Assert.Equal(["B-PER", "O"], NerAlignment.Aligned(["O", "B-PER", "O"]));
        Assert.Empty(NerAlignment.Aligned(["O"]));
        Assert.Empty(NerAlignment.Aligned([]));
        Assert.Empty(NerAlignment.Aligned(null));
    }

    [Fact]
    public void AModelThatIsNotAnNerModel_IsLeftAsItIs()
    {
        var ml = new MLContext(1);
        var model = Stand(ml);

        Assert.False(NerAlignment.IsNer(model));
        Assert.Null(NerAlignment.TextColumnOf(model));
        Assert.Same(model, NerAlignment.Apply(ml, model));
    }

    [Fact]
    public void AnAlignedModel_IsNotSaved_TheModelFileIsTheTrainedModelAlone()
    {
        var ml = new MLContext(1);
        var model = NerAlignment.Apply(ml, Stand(ml), "Sentence");

        Assert.True(model.IsRowToRowMapper);
        Assert.Throws<NotSupportedException>(() => ((ICanSaveModel)model).Save(null!));
    }

    // What one sentence's answer is in the transformer's shape: the start token's slot, then each word's slot holds the
    // tag of the word before it (a word outside every entity is the outside tag).
    private static string[] Shift(string? sentence)
    {
        var words = (sentence ?? "").Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0)
            return [];
        var own = words.Select(w => TagOf.GetValueOrDefault(w, "O")).ToArray();
        return ["O", .. own.Take(words.Length - 1)];
    }

    private static ITransformer Stand(MLContext ml)
    {
        var input = SchemaDefinition.Create(typeof(Input));
        var output = SchemaDefinition.Create(typeof(Output));
        return ml.Transforms.CustomMapping<Input, Output>(
                (row, result) => result.PredictedLabel = Shift(row.Sentence), contractName: null, input, output)
            .Fit(ml.Data.LoadFromEnumerable(Array.Empty<Input>(), input));
    }

    private static List<string[]> Shifted(MLContext ml, params string[] sentences) =>
        ml.Data.CreateEnumerable<Output>(
                Stand(ml).Transform(ml.Data.LoadFromEnumerable(sentences.Select(s => new Input { Sentence = s }))), reuseRowObject: false)
            .Select(o => o.PredictedLabel).ToList();

    private static List<string[]> Aligned(MLContext ml, string[] sentences) =>
        ml.Data.CreateEnumerable<Output>(
                NerAlignment.Apply(ml, Stand(ml), "Sentence")
                    .Transform(ml.Data.LoadFromEnumerable(sentences.Select(s => new Input { Sentence = s }))), reuseRowObject: false)
            .Select(o => o.PredictedLabel).ToList();

    private sealed class Input
    {
        public string? Sentence { get; set; }
    }

    private sealed class Output
    {
        public string[] PredictedLabel { get; set; } = [];
    }
}
