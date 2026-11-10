using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// ML.NET's question-answering trainer takes the answer's last token as inclusive and its decoder cuts the span as if
/// that token were exclusive, so every answer came back without its last token: a model that found every span scored
/// <c>exact_match</c> 0 (33 of 60 answers empty, the other 27 a strict prefix of the right one). The model is taught the
/// span one token longer; these tests pin how far, and that the passage the model reads always has a token after its last.
/// </summary>
public class AnswerAlignmentTests
{
    [Theory]
    [InlineData("Bob works at Stark in Oslo. It opened.", "Oslo", "Oslo.")]          // punctuation is a token of its own
    [InlineData("Leo works at Tyrell in Oslo. It opened.", "Tyrell", "Tyrell i")]    // a space has no token: reach the next word's first character
    [InlineData("Leo works at Tyrell   in Oslo.", "Tyrell", "Tyrell   i")]           // however many spaces
    [InlineData("the capital is Oslo", "Oslo", "Oslo .")]                              // the end of the passage: the padding word is the next token
    public void TheTrainerIsToldTheAnswerReachesTheNextTokensFirstCharacter(string passage, string answer, string expected)
    {
        var padded = TextPadding.Padded(passage);
        var start = passage.IndexOf(answer, StringComparison.Ordinal);

        var told = AnswerAlignment.TrainingAnswer(padded, start, answer.Length);

        Assert.Equal(expected, told);
        Assert.StartsWith(answer, told, StringComparison.Ordinal);   // the answer is never changed, only how far it reaches
    }

    [Fact]
    public void AnAnswerThatEndsWhereThePassageEnds_IsStillGivenASuccessor()
    {
        var padded = TextPadding.Padded("see Oslo");

        Assert.Equal("Oslo .", AnswerAlignment.TrainingAnswer(padded, 4, 4));
    }

    [Fact]
    public void APassage_IsReadWithAWordAfterItsLast_AndGivenBackAsItWas()
    {
        var ml = new MLContext(1);
        var seen = new List<string?>();
        var input = SchemaDefinition.Create(typeof(Row));
        var output = SchemaDefinition.Create(typeof(Answer));
        var stand = ml.Transforms.CustomMapping<Row, Answer>(
                (row, result) => { seen.Add(row.Context); result.Text = "x"; }, contractName: null, input, output)
            .Fit(ml.Data.LoadFromEnumerable(Array.Empty<Row>(), input));

        var scored = AnswerAlignment.Apply(ml, stand, "Context")
            .Transform(ml.Data.LoadFromEnumerable(new[] { new Row { Context = "Oslo is a city" } }));

        // Rows are made when read: the model's answer is what makes it read the passage.
        Assert.Equal(["x"], scored.GetColumn<ReadOnlyMemory<char>>("Text").Select(t => t.ToString()));
        Assert.Equal(["Oslo is a city ."], seen);
        Assert.Equal(["Oslo is a city"], scored.GetColumn<ReadOnlyMemory<char>>("Context").Select(t => t.ToString()));
    }

    [Fact]
    public void AModelThatIsNotAQuestionAnsweringModel_IsLeftAsItIs()
    {
        var ml = new MLContext(1);
        var model = ml.Transforms.Conversion.MapValueToKey("Label").Fit(ml.Data.LoadFromEnumerable(new[] { new Labeled { Label = "a" } }));

        Assert.False(AnswerAlignment.IsQuestionAnswering(model));
        Assert.Null(AnswerAlignment.ContextColumnOf(model));
        Assert.Same(model, AnswerAlignment.Apply(ml, model));
    }

    [Fact]
    public void APaddedModel_IsNotSaved_TheModelFileIsTheTrainedModelAlone()
    {
        var ml = new MLContext(1);
        var model = AnswerAlignment.Apply(ml, ml.Transforms.Conversion.MapValueToKey("Label")
            .Fit(ml.Data.LoadFromEnumerable(new[] { new Labeled { Label = "a" } })), "Label");

        Assert.Throws<NotSupportedException>(() => ((ICanSaveModel)model).Save(null!));
    }

    private sealed class Row
    {
        public string? Context { get; set; }
    }

    private sealed class Answer
    {
        public string? Text { get; set; }
    }

    private sealed class Labeled
    {
        public string? Label { get; set; }
    }
}
