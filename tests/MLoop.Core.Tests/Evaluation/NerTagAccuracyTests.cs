using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class NerTagAccuracyTests
{
    private sealed class Scored
    {
        public string Label { get; set; } = "";

        [VectorType]
        public string[] PredictedLabel { get; set; } = [];
    }

    private static NerScores Measure(params Scored[] rows)
    {
        var ml = new MLContext(1);
        return NerTagAccuracy.Measure(ml, ml.Data.LoadFromEnumerable(rows), "Label");
    }

    [Fact]
    public void A_missing_tag_counts_as_O()
    {
        // The trainer answers a word outside every entity with a missing key ("").
        var scores = Measure(new Scored { Label = "O B-DATE O", PredictedLabel = ["", "B-DATE", ""] });

        Assert.Equal(1.0, scores.Micro);
        Assert.Equal(1.0, scores.EntityF1);
    }

    [Fact]
    public void Micro_counts_words_and_macro_averages_tags()
    {
        // Six O words right, the one B-DATE wrong: most words are right, but one tag of two never is.
        var scores = Measure(
            new Scored { Label = "O O O B-DATE", PredictedLabel = ["", "", "", ""] },
            new Scored { Label = "O O O", PredictedLabel = ["", "", ""] });

        Assert.Equal(6.0 / 7, scores.Micro, 6);
        Assert.Equal(0.5, scores.Macro, 6);
        Assert.Equal(0, scores.EntityRecall);
    }

    [Fact]
    public void A_shorter_prediction_leaves_the_rest_wrong()
    {
        var scores = Measure(new Scored { Label = "O B-PER I-PER", PredictedLabel = ["", "B-PER"] });

        Assert.Equal(2.0 / 3, scores.Micro, 6);
        Assert.Equal(0, scores.EntityF1);   // half a person is no person found
    }

    [Fact]
    public void Entities_count_only_whole_spans_of_the_right_type()
    {
        // Gold: PER(1–2), LOC(4). Found: PER(1–2) right, LOC(4) as ORG wrong, and a stray DATE(0).
        var scores = Measure(new Scored { Label = "O B-PER I-PER O B-LOC", PredictedLabel = ["B-DATE", "B-PER", "I-PER", "", "B-ORG"] });

        Assert.Equal(1.0 / 3, scores.EntityPrecision, 6);
        Assert.Equal(0.5, scores.EntityRecall, 6);
        Assert.Equal(0.4, scores.EntityF1, 6);
        Assert.Equal(["entity_f1", "entity_precision", "entity_recall", "macro_accuracy", "micro_accuracy"],
            scores.Metrics().Keys.Order());
    }

    [Theory]
    [InlineData("B-PER I-PER O B-PER", "0-1:PER 3-3:PER")]
    [InlineData("I-PER I-PER O", "0-1:PER")]                 // IOB1: an I with nothing before starts one
    [InlineData("B-PER B-PER", "0-0:PER 1-1:PER")]
    [InlineData("I-PER I-LOC", "0-0:PER 1-1:LOC")]
    [InlineData("S-PER B-LOC E-LOC O", "0-0:PER 1-2:LOC")]   // BIOES
    [InlineData("PER PER O LOC", "0-1:PER 3-3:LOC")]         // tags without a prefix
    [InlineData("O O", "")]
    public void Entities_read_the_IOB_family_leniently(string tags, string expected)
    {
        var entities = NerTagAccuracy.Entities(tags.Split(' '))
            .OrderBy(e => e.Start).Select(e => $"{e.Start}-{e.End}:{e.Type}");

        Assert.Equal(expected, string.Join(' ', entities));
    }

    [Fact]
    public void Every_recorded_score_is_a_known_unit_scaled_higher_is_better_metric()
    {
        foreach (var name in new NerScores(0, 0, 0, 0, 0).Metrics().Keys)
        {
            Assert.True(MetricDirection.IsKnown(name), name);
            Assert.False(MetricDirection.IsLowerBetter(name), name);
            Assert.True(MetricScale.IsUnitScaled(name), name);
        }
    }
}
