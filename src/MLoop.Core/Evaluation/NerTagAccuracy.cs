using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Scores a NER model word by word. A sentence's label is a vector of tags, which the
/// multiclass evaluator — built for one class per row — cannot read. Training and
/// <c>mloop evaluate</c> both score with this, so the number a model was promoted on is the number
/// evaluation reports.
/// </summary>
public static class NerTagAccuracy
{
    /// <summary>
    /// Scores <paramref name="scored"/> — a model's output over data that still carries its label
    /// column, one space-separated tag per word — against the model's <c>PredictedLabel</c>.
    /// </summary>
    public static (double Micro, double Macro) Measure(MLContext mlContext, IDataView scored, string labelColumn)
    {
        const string expectedTags = "__NerExpectedTags";
        var withTags = mlContext.Transforms.Text.TokenizeIntoWords(expectedTags, labelColumn, [Prediction.TagSequence.Separator])
            .Fit(scored)
            .Transform(scored);
        return Measure(withTags, expectedTags, "PredictedLabel");
    }

    /// <summary>
    /// <c>micro</c>: the share of words tagged correctly. <c>macro</c>: the mean, over the tags
    /// that occur, of the share of each tag's words tagged correctly — so a model that tags
    /// everything <c>O</c> does not score well on text where most words are <c>O</c>.
    /// </summary>
    public static (double Micro, double Macro) Measure(IDataView scored, string expectedColumn, string predictedColumn)
    {
        var expectedCol = scored.Schema[expectedColumn];
        var predictedCol = scored.Schema[predictedColumn];
        using var cursor = scored.GetRowCursor([expectedCol, predictedCol]);
        var expectedGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(expectedCol);
        var predictedGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedCol);

        VBuffer<ReadOnlyMemory<char>> expected = default, predicted = default;
        var perTag = new Dictionary<string, (int Right, int Total)>(StringComparer.Ordinal);
        int right = 0, total = 0;
        while (cursor.MoveNext())
        {
            expectedGetter(ref expected);
            predictedGetter(ref predicted);
            var e = expected.DenseValues().Select(v => v.ToString()).ToArray();
            var p = predicted.DenseValues().Select(v => Prediction.TagSequence.Tag(v.ToString())).ToArray();
            for (var i = 0; i < e.Length; i++)
            {
                var hit = i < p.Length && p[i] == e[i];
                perTag.TryGetValue(e[i], out var t);
                perTag[e[i]] = (t.Right + (hit ? 1 : 0), t.Total + 1);
                right += hit ? 1 : 0;
                total++;
            }
        }

        if (total == 0)
            return (0, 0);
        return ((double)right / total, perTag.Values.Average(t => (double)t.Right / t.Total));
    }
}
