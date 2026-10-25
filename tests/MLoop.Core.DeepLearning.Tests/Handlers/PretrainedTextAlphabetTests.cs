using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using MLoop.Core.DeepLearning;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

public class PretrainedTextAlphabetTests
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
        Assert.Equal(expected, PretrainedTextAlphabet.CanWrite(answer));

    [Fact]
    public void Data_whose_answers_are_all_writable_draws_no_warning() =>
        Assert.Null(PretrainedTextAlphabet.AnswerWarning(writable: 20, total: 20));

    [Fact]
    public void The_answer_warning_says_how_many_answers_and_where_exact_match_stops()
    {
        var warning = PretrainedTextAlphabet.AnswerWarning(writable: 1, total: 4);

        Assert.NotNull(warning);
        Assert.Contains("3 of 4", warning);
        Assert.Contains("exact_match cannot exceed 25%", warning);
    }

    /// <summary>
    /// The rule is checked against the tokenizer the text trainers use, character by character: what it keeps
    /// through an encode and decode is exactly what <see cref="PretrainedTextAlphabet.Keeps"/> says. The tokenizer's
    /// files ship inside Microsoft.ML.TorchSharp, so this reads nothing from the network.
    /// </summary>
    [Fact]
    public void The_rule_is_the_tokenizers()
    {
        var tokenizer = TrainersTokenizer();
        var disagreements = new List<string>();
        foreach (var c in Enumerable.Range(0x21, 0x3000 - 0x21).Where(c => c < 0xD800 || c >= 0xE000).Select(c => (char)c))
        {
            var text = $"a{c}a";
            var kept = tokenizer.Decode(tokenizer.EncodeToIds(text)) == text;
            if (kept != PretrainedTextAlphabet.Keeps(c))
                disagreements.Add($"U+{(int)c:X4} tokenizer keeps: {kept}");
        }

        Assert.True(disagreements.Count == 0, string.Join(Environment.NewLine, disagreements.Take(20)));
    }

    [Fact]
    public void The_tokenizer_reads_Korean_text_as_nothing()
    {
        var tokenizer = TrainersTokenizer();

        Assert.Equal(" ", tokenizer.Decode(tokenizer.EncodeToIds("음악을 들으면서")));
        Assert.Equal("2026 10 25", tokenizer.Decode(tokenizer.EncodeToIds("2026년 10월 25일")).Trim());
    }

    [Fact]
    public void The_dropped_share_counts_non_space_characters_over_every_row()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[] { new Row { Text = "ab 가나" }, new Row { Text = "  다라  " } });

        Assert.Equal(4.0 / 6, PretrainedTextAlphabet.DroppedShare(data, nameof(Row.Text)), precision: 10);
    }

    [Theory]
    [InlineData(0.0, false)]
    [InlineData(0.05, false)]   // English with curly quotes and dashes loses about this much — not the warning's case
    [InlineData(0.84, true)]
    public void Training_warns_only_when_much_of_the_text_is_unreadable(double share, bool warns)
    {
        var warning = PretrainedTextAlphabet.ReadingWarning("NER", "sentence", share);

        Assert.Equal(warns, warning is not null);
        if (warns)
        {
            Assert.Contains("84%", warning);
            Assert.Contains("'sentence'", warning);
        }
    }

    [Fact]
    public void A_classification_warning_names_the_task_that_reads_the_text()
    {
        var warning = PretrainedTextAlphabet.ReadingWarning("Text classification", "Text", 0.82, PretrainedTextAlphabet.ClassifyInstead);

        Assert.Contains("--task multiclass-classification", warning);
        Assert.Contains("encoder: multilingual-minilm", warning);
        Assert.DoesNotContain("--task", PretrainedTextAlphabet.ReadingWarning("NER", "Text", 0.82));
    }

    [Fact]
    public void A_sentence_similarity_warning_names_the_regression_over_two_encoded_columns()
    {
        var warning = PretrainedTextAlphabet.ReadingWarning("Sentence similarity", "s1", 0.85, PretrainedTextAlphabet.CompareInstead);

        Assert.Contains("--task regression", warning);
        Assert.Contains("encoder: multilingual-minilm", warning);
        Assert.Contains("cosine", warning);
    }

    private sealed class Row
    {
        public string Text { get; set; } = "";
    }

    // TorchSharp's own tokenizer instance — the one TextClassification, SentenceSimilarity, NER and QA build.
    private static Tokenizer TrainersTokenizer()
    {
        var extensions = typeof(Microsoft.ML.TorchSharp.NasBert.NerTrainer).Assembly
            .GetType("Microsoft.ML.TorchSharp.Extensions.TokenizerExtensions", throwOnError: true)!;
        var env = (IHostEnvironment)new MLContext(seed: 1);
        using var channel = env.Register(nameof(PretrainedTextAlphabetTests)).Start(nameof(PretrainedTextAlphabetTests));
        return (Tokenizer)extensions
            .GetMethod("GetInstance", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!
            .Invoke(null, [channel])!;
    }
}
