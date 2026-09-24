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

    private static (double Micro, double Macro) Measure(params Scored[] rows)
    {
        var ml = new MLContext(1);
        return NerTagAccuracy.Measure(ml, ml.Data.LoadFromEnumerable(rows), "Label");
    }

    [Fact]
    public void A_missing_tag_counts_as_O()
    {
        // The trainer answers a word outside every entity with a missing key ("").
        var (micro, _) = Measure(new Scored { Label = "O B-DATE O", PredictedLabel = ["", "B-DATE", ""] });

        Assert.Equal(1.0, micro);
    }

    [Fact]
    public void Micro_counts_words_and_macro_averages_tags()
    {
        // Six O words right, the one B-DATE wrong: most words are right, but one tag of two never is.
        var (micro, macro) = Measure(
            new Scored { Label = "O O O B-DATE", PredictedLabel = ["", "", "", ""] },
            new Scored { Label = "O O O", PredictedLabel = ["", "", ""] });

        Assert.Equal(6.0 / 7, micro, 6);
        Assert.Equal(0.5, macro, 6);
    }

    [Fact]
    public void A_shorter_prediction_leaves_the_rest_wrong()
    {
        var (micro, _) = Measure(new Scored { Label = "O B-PER I-PER", PredictedLabel = ["", "B-PER"] });

        Assert.Equal(2.0 / 3, micro, 6);
    }
}
