using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

public class HasTextLabelTests
{
    [Theory]
    [InlineData("ner")]
    [InlineData("question-answering")]
    [InlineData(" NER ")]
    public void A_tag_sequence_or_an_answer_is_a_text_label(string task) =>
        Assert.True(TrainingRunner.HasTextLabel(task));

    [Theory]
    [InlineData("regression")]
    [InlineData("text-classification")]
    [InlineData("sentence-similarity")]
    [InlineData(null)]
    public void A_number_or_a_class_is_not(string? task) =>
        Assert.False(TrainingRunner.HasTextLabel(task));
}
