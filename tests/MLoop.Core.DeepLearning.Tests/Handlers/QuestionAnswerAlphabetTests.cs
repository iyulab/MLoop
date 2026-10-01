using MLoop.Core.DeepLearning;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

public class QuestionAnswerAlphabetTests
{
    [Theory]
    [InlineData("Paris", true)]
    [InlineData("1,100 km", true)]
    [InlineData("café", true)]       // é is U+00E9: one byte, one character
    [InlineData("서울", false)]
    [InlineData("12월 31일", false)]  // the digits come out, the Hangul does not
    [InlineData("東京", false)]
    [InlineData("Москва", false)]
    public void An_answer_is_writable_only_when_every_character_is_one_byte(string answer, bool expected) =>
        Assert.Equal(expected, QuestionAnswerAlphabet.CanWrite(answer));

    [Fact]
    public void Data_whose_answers_are_all_writable_draws_no_warning() =>
        Assert.Null(QuestionAnswerAlphabet.Warning(writable: 20, total: 20));

    [Fact]
    public void The_warning_says_how_many_answers_and_where_exact_match_stops()
    {
        var warning = QuestionAnswerAlphabet.Warning(writable: 1, total: 4);

        Assert.NotNull(warning);
        Assert.Contains("3 of 4", warning);
        Assert.Contains("exact_match cannot exceed 25%", warning);
    }
}
