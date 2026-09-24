using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class TagSequenceTests
{
    [Fact]
    public void A_word_outside_every_entity_is_written_O()
    {
        // The trainer answers such a word with a missing key, which reads back as "".
        Assert.Equal("O B-DATE I-DATE O", TagSequence.Render(["", "B-DATE", "I-DATE", ""]));
    }

    [Fact]
    public void Tags_are_joined_the_way_the_label_column_is_read()
    {
        Assert.Equal("B-PER O", TagSequence.Render(["B-PER", "O"]));
    }

    private sealed class TagRow
    {
        [VectorType]
        public string[] PredictedLabel { get; set; } = [];
    }

    [Fact]
    public void A_scored_view_of_tag_vectors_is_not_taken_for_a_degenerate_model()
    {
        // Before NER had an extractor its vector fell to the score reader, found no score, and every
        // prediction was refused as degenerate.
        var view = new MLContext(1).Data.LoadFromEnumerable(new[]
        {
            new TagRow { PredictedLabel = ["", "B-DATE", ""] },
            new TagRow { PredictedLabel = ["", "", ""] },
        });

        PredictionService.RequireNonDegenerateScoredView(view, "ner");
    }
}
