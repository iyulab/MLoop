using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>A NER model's scores: word by word, and entity by entity.</summary>
/// <param name="Micro">The share of words tagged correctly.</param>
/// <param name="Macro">The mean, over the tags that occur, of the share of each tag's words tagged correctly.</param>
/// <param name="EntityPrecision">Of the entities the model found, the share that are an entity of the data — same words, same type.</param>
/// <param name="EntityRecall">Of the data's entities, the share the model found exactly.</param>
/// <param name="EntityF1">The harmonic mean of the two — the score NER is usually reported by (CoNLL).</param>
public sealed record NerScores(double Micro, double Macro, double EntityPrecision, double EntityRecall, double EntityF1)
{
    /// <summary>The scores by the names training and <c>mloop evaluate</c> record them under.</summary>
    public Dictionary<string, double> Metrics() => new()
    {
        [NerTagAccuracy.MacroAccuracy] = Macro,
        [NerTagAccuracy.MicroAccuracy] = Micro,
        [NerTagAccuracy.EntityPrecisionName] = EntityPrecision,
        [NerTagAccuracy.EntityRecallName] = EntityRecall,
        [NerTagAccuracy.EntityF1Name] = EntityF1,
    };
}

/// <summary>
/// Scores a NER model word by word and entity by entity. A sentence's label is a vector of tags, which
/// the multiclass evaluator — built for one class per row — cannot read. Training and
/// <c>mloop evaluate</c> both score with this, so the number a model was promoted on is the number
/// evaluation reports.
/// </summary>
/// <remarks>
/// Word accuracy flatters a model that finds half of every entity: «B-PER I-PER» answered «B-PER O» is half the
/// words right and no person found. The entity scores count a hit only for the whole span with its type.
/// </remarks>
public static class NerTagAccuracy
{
    /// <summary>The tag average — what a model answering O everywhere cannot fake; the gate's 1/N floor applies to it.</summary>
    public const string MacroAccuracy = "macro_accuracy";

    /// <summary>The share of words tagged correctly.</summary>
    public const string MicroAccuracy = "micro_accuracy";

    /// <summary>Entity precision.</summary>
    public const string EntityPrecisionName = "entity_precision";

    /// <summary>Entity recall.</summary>
    public const string EntityRecallName = "entity_recall";

    /// <summary>Entity F1.</summary>
    public const string EntityF1Name = "entity_f1";

    /// <summary>
    /// Scores <paramref name="scored"/> — a model's output over data that still carries its label
    /// column, one space-separated tag per word — against the model's <c>PredictedLabel</c>.
    /// </summary>
    public static NerScores Measure(MLContext mlContext, IDataView scored, string labelColumn)
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
    /// everything <c>O</c> does not score well on text where most words are <c>O</c>. The entity scores
    /// compare whole spans (<see cref="Entities"/>).
    /// </summary>
    public static NerScores Measure(IDataView scored, string expectedColumn, string predictedColumn)
    {
        var expectedCol = scored.Schema[expectedColumn];
        var predictedCol = scored.Schema[predictedColumn];
        using var cursor = scored.GetRowCursor([expectedCol, predictedCol]);
        var expectedGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(expectedCol);
        var predictedGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedCol);

        VBuffer<ReadOnlyMemory<char>> expected = default, predicted = default;
        var perTag = new Dictionary<string, (int Right, int Total)>(StringComparer.Ordinal);
        int right = 0, total = 0, found = 0, present = 0, matched = 0;
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

            var gold = Entities(e);
            var guessed = Entities(p.Take(e.Length).ToArray());
            present += gold.Count;
            found += guessed.Count;
            matched += guessed.Count(gold.Contains);
        }

        if (total == 0)
            return new NerScores(0, 0, 0, 0, 0);
        var precision = found == 0 ? 0 : (double)matched / found;
        var recall = present == 0 ? 0 : (double)matched / present;
        var f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return new NerScores((double)right / total, perTag.Values.Average(t => (double)t.Right / t.Total), precision, recall, f1);
    }

    /// <summary>
    /// The entities a tag sequence marks, as (first word, last word, type). Read leniently, as the IOB family is
    /// written in practice: <c>B-X</c> starts an X; <c>I-X</c> (and <c>E-X</c>, which also ends it) continues an
    /// X or starts one when the word before is not an X; <c>S-X</c> is a one-word X; a tag without a prefix runs
    /// as one entity while it repeats; <c>O</c> or nothing is outside.
    /// </summary>
    public static HashSet<(int Start, int End, string Type)> Entities(IReadOnlyList<string> tags)
    {
        ArgumentNullException.ThrowIfNull(tags);
        var entities = new HashSet<(int, int, string)>();
        string? type = null;
        var start = 0;

        void Close(int end)
        {
            if (type is not null)
                entities.Add((start, end, type));
            type = null;
        }

        for (var i = 0; i < tags.Count; i++)
        {
            var tag = Prediction.TagSequence.Tag(tags[i]);
            if (tag == Prediction.TagSequence.Outside)
            {
                Close(i - 1);
                continue;
            }
            var (prefix, name) = tag.Length > 2 && tag[1] == '-' ? (tag[0], tag[2..]) : ('\0', tag);
            switch (prefix)
            {
                case 'B':
                    Close(i - 1);
                    (type, start) = (name, i);
                    break;
                case 'S':
                    Close(i - 1);
                    entities.Add((i, i, name));
                    break;
                case 'I' or 'E' or '\0':
                    if (type != name)
                    {
                        Close(i - 1);
                        (type, start) = (name, i);
                    }
                    if (prefix == 'E')
                        Close(i);
                    break;
                default:
                    Close(i - 1);
                    (type, start) = (tag, i);
                    break;
            }
        }
        Close(tags.Count - 1);
        return entities;
    }
}
