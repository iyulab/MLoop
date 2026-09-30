using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class AnswerOverlapTests
{
    private sealed class Scored
    {
        public string Label { get; set; } = "";

        [VectorType]
        public string[] Answer { get; set; } = [];
    }

    private static (double ExactMatch, double F1) Measure(params Scored[] rows)
    {
        var ml = new MLContext(seed: 0);
        return AnswerOverlap.Measure(ml.Data.LoadFromEnumerable(rows), "Label");
    }

    [Fact]
    public void An_answer_equal_after_normalization_is_an_exact_match()
    {
        Assert.True(AnswerOverlap.IsExactMatch("\"Seoul.\"", "seoul"));
        Assert.True(AnswerOverlap.IsExactMatch("《수도 동파》", "수도  동파"));
        Assert.False(AnswerOverlap.IsExactMatch("서울에서", "서울"));
    }

    [Fact]
    public void F1_counts_characters_so_an_answer_inside_a_word_is_partly_right()
    {
        // A word-level F1 scores this zero: the prediction and the truth share no whole word.
        var f1 = AnswerOverlap.CharacterF1("서울에서", "서울");

        Assert.Equal(2 * 0.5 * 1.0 / 1.5, f1, 6);
    }

    [Fact]
    public void F1_ignores_spaces_and_counts_repeated_characters_once_each()
    {
        Assert.Equal(1.0, AnswerOverlap.CharacterF1("수도 동파", "수도동파"), 6);
        // "aab" vs "ab": two characters in common (a, b), not three.
        Assert.Equal(2 * (2.0 / 3) * 1.0 / (2.0 / 3 + 1.0), AnswerOverlap.CharacterF1("aab", "ab"), 6);
    }

    [Fact]
    public void An_empty_answer_scores_zero()
    {
        Assert.Equal(0, AnswerOverlap.CharacterF1("", "서울"));
        Assert.Equal(0, AnswerOverlap.CharacterF1("...", "서울"));
    }

    [Fact]
    public void Measure_averages_over_rows_and_reads_only_the_best_answer()
    {
        var (em, f1) = Measure(
            new Scored { Label = "서울", Answer = ["서울", "부산"] },
            new Scored { Label = "부산", Answer = ["서울", "부산"] });

        Assert.Equal(0.5, em, 6);
        Assert.Equal(0.5, f1, 6);
    }

    [Fact]
    public void A_row_without_an_answer_scores_zero()
    {
        var (em, f1) = Measure(new Scored { Label = "서울", Answer = [] });

        Assert.Equal(0, em);
        Assert.Equal(0, f1);
    }
}
