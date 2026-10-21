using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Evaluation;

/// <summary>
/// How a classifier's answers fell against the labels: <see cref="Counts"/>[actual][predicted], each
/// row counting by its weight. Every class-set measure is read from it — «of the rows whose label is
/// in A, the share answered with a class in B» (<see cref="ShareOf"/>) is a miss rate when A holds the
/// classes that must be caught and B the ones that let them through, a recall when B = A — so a
/// consumer states its own classes instead of the evaluator knowing any.
/// </summary>
/// <param name="Classes">The class names, in the model's order.</param>
/// <param name="Counts">Weighted row count per (actual, predicted) pair, indexed like <paramref name="Classes"/>.</param>
public sealed record ConfusionCounts(IReadOnlyList<string> Classes, IReadOnlyList<IReadOnlyList<double>> Counts)
{
    /// <summary>The weighted rows counted.</summary>
    public double Total => Counts.Sum(r => r.Sum());

    /// <summary>
    /// Of the rows whose label is one of <paramref name="actual"/>, the weighted share answered with
    /// one of <paramref name="predicted"/>; <c>null</c> when no row has such a label. Names match
    /// ignoring case; a name that is not a class matches nothing.
    /// </summary>
    public double? ShareOf(IEnumerable<string> actual, IEnumerable<string> predicted)
    {
        var a = Indices(actual);
        var p = Indices(predicted);
        double rows = 0, hits = 0;
        foreach (var i in a)
            for (var j = 0; j < Classes.Count; j++)
            {
                rows += Counts[i][j];
                if (p.Contains(j))
                    hits += Counts[i][j];
            }
        return rows > 0 ? hits / rows : null;
    }

    private HashSet<int> Indices(IEnumerable<string> names)
    {
        var wanted = names.Select(n => n.Trim()).ToHashSet(StringComparer.OrdinalIgnoreCase);
        return Enumerable.Range(0, Classes.Count).Where(i => wanted.Contains(Classes[i])).ToHashSet();
    }

    /// <summary>
    /// The matrix of <paramref name="scored"/> (a classifier's output over labeled rows), or <c>null</c>
    /// for a task that is not binary or multiclass classification, or when the columns it reads are
    /// missing. Rows without a label, or with a weight that is not positive, are not counted.
    /// </summary>
    /// <param name="scored">The model's output over the test rows.</param>
    /// <param name="task">The task the model was trained for.</param>
    /// <param name="labelColumn">The label column, as the model's pipeline left it (binary: boolean or number; multiclass: a key).</param>
    /// <param name="weights">Each row's weight, in row order; <c>null</c> counts every row once.</param>
    /// <param name="schema">The saved schema — a binary model's class names come from it (<see cref="BinaryLabelVocabulary"/>).</param>
    public static ConfusionCounts? Of(IDataView scored, string? task, string labelColumn, IReadOnlyList<double>? weights = null, InputSchemaInfo? schema = null)
    {
        ArgumentNullException.ThrowIfNull(scored);
        return TaskTypes.Canonical(task) switch
        {
            "binary-classification" => Binary(scored, labelColumn, weights, BinaryLabelVocabulary.Of(schema)),
            "multiclass-classification" => Multiclass(scored, labelColumn, weights),
            _ => null,
        };
    }

    private static ConfusionCounts? Binary(IDataView scored, string labelColumn, IReadOnlyList<double>? weights, (string Negative, string Positive)? vocabulary)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var predicted = scored.Schema.GetColumnOrNull("PredictedLabel");
        if (label is null || predicted is null || predicted.Value.Type != BooleanDataViewType.Instance
            || (label.Value.Type != BooleanDataViewType.Instance && label.Value.Type != NumberDataViewType.Single))
            return null;

        var counts = new double[2, 2];
        using (var cursor = scored.GetRowCursor([label.Value, predicted.Value]))
        {
            var getPredicted = cursor.GetGetter<bool>(predicted.Value);
            Func<bool?> truth;
            if (label.Value.Type == BooleanDataViewType.Instance)
            {
                var get = cursor.GetGetter<bool>(label.Value);
                truth = () => { bool v = default; get(ref v); return v; };
            }
            else
            {
                var get = cursor.GetGetter<float>(label.Value);
                truth = () => { float v = default; get(ref v); return float.IsNaN(v) ? null : v > 0; };
            }
            bool p = default;
            var i = 0;
            while (cursor.MoveNext())
            {
                getPredicted(ref p);
                var w = weights is null ? 1.0 : weights[i];
                i++;
                if (truth() is not { } y || !(w > 0))
                    continue;
                counts[y ? 1 : 0, p ? 1 : 0] += w;
            }
        }
        return new ConfusionCounts(
            [BinaryLabelVocabulary.Render(false, vocabulary), BinaryLabelVocabulary.Render(true, vocabulary)],
            [[counts[0, 0], counts[0, 1]], [counts[1, 0], counts[1, 1]]]);
    }

    private static ConfusionCounts? Multiclass(IDataView scored, string labelColumn, IReadOnlyList<double>? weights)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull("Score");
        if (label is null || score is null || label.Value.Type is not KeyDataViewType key
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType } vector
            || vector.Size != (int)key.Count)
            return null;

        // The label key's own values name the classes, and the trainer's score slots follow the key.
        var classes = Enumerable.Range(1, vector.Size).Select(k => $"class_{k - 1}").ToArray();
        if (label.Value.Annotations.Schema.GetColumnOrNull("KeyValues") is { Type: VectorDataViewType { ItemType: TextDataViewType } })
        {
            VBuffer<ReadOnlyMemory<char>> names = default;
            label.Value.GetKeyValues(ref names);
            var values = names.DenseValues().Select(v => v.ToString()).ToArray();
            if (values.Length == classes.Length)
                classes = values;
        }

        var counts = new double[classes.Length, classes.Length];
        using (var cursor = scored.GetRowCursor([label.Value, score.Value]))
        {
            var getLabel = cursor.GetGetter<uint>(label.Value);
            var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
            uint k = default;
            VBuffer<float> scores = default;
            var i = 0;
            while (cursor.MoveNext())
            {
                getLabel(ref k);
                getScore(ref scores);
                var w = weights is null ? 1.0 : weights[i];
                i++;
                if (k == 0 || !(w > 0))
                    continue;
                var values = scores.DenseValues().ToArray();
                var top = 0;
                for (var c = 1; c < values.Length; c++)
                    if (values[c] > values[top])
                        top = c;
                counts[k - 1, top] += w;
            }
        }
        return new ConfusionCounts(classes,
            Enumerable.Range(0, classes.Length)
                .Select(r => (IReadOnlyList<double>)Enumerable.Range(0, classes.Length).Select(c => counts[r, c]).ToArray())
                .ToArray());
    }
}
