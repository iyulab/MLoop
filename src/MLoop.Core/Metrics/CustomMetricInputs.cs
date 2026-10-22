using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Metrics;

/// <summary>
/// The predictions and labels a project's metric script receives (<c>MetricContext.Predictions</c> /
/// <c>Labels</c>), read from scored rows in the shapes that contract states: binary <c>bool[]</c>,
/// multiclass <c>int[]</c> class indices (the trainer's score slots), regression <c>float[]</c>. Rows
/// without a label are left out of both arrays, so index <c>i</c> of one is row <c>i</c> of the other.
/// </summary>
public static class CustomMetricInputs
{
    /// <summary>The two arrays, or <c>null</c> for a task the contract does not cover or scored rows it cannot read.</summary>
    public static (Array Predictions, Array Labels)? Of(IDataView scored, string? task, string labelColumn)
    {
        ArgumentNullException.ThrowIfNull(scored);
        return TaskTypes.Canonical(task) switch
        {
            "binary-classification" => Binary(scored, labelColumn),
            "multiclass-classification" => Multiclass(scored, labelColumn),
            "regression" => Regression(scored, labelColumn),
            _ => null,
        };
    }

    private static (Array, Array)? Binary(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var predicted = scored.Schema.GetColumnOrNull("PredictedLabel");
        if (label is null || predicted is null || predicted.Value.Type != BooleanDataViewType.Instance
            || (label.Value.Type != BooleanDataViewType.Instance && label.Value.Type != NumberDataViewType.Single))
            return null;

        var predictions = new List<bool>();
        var labels = new List<bool>();
        using var cursor = scored.GetRowCursor([label.Value, predicted.Value]);
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
        while (cursor.MoveNext())
        {
            if (truth() is not { } y)
                continue;
            getPredicted(ref p);
            predictions.Add(p);
            labels.Add(y);
        }
        return (predictions.ToArray(), labels.ToArray());
    }

    private static (Array, Array)? Multiclass(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull("Score");
        if (label is null || score is null || label.Value.Type is not KeyDataViewType key
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType } vector
            || vector.Size != (int)key.Count)
            return null;

        var predictions = new List<int>();
        var labels = new List<int>();
        using var cursor = scored.GetRowCursor([label.Value, score.Value]);
        var getLabel = cursor.GetGetter<uint>(label.Value);
        var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
        uint k = default;
        VBuffer<float> scores = default;
        while (cursor.MoveNext())
        {
            getLabel(ref k);
            if (k == 0)
                continue;
            getScore(ref scores);
            var values = scores.DenseValues().ToArray();
            var top = 0;
            for (var c = 1; c < values.Length; c++)
                if (values[c] > values[top])
                    top = c;
            predictions.Add(top);
            labels.Add((int)k - 1);
        }
        return (predictions.ToArray(), labels.ToArray());
    }

    private static (Array, Array)? Regression(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull("Score");
        if (label is null || score is null || label.Value.Type != NumberDataViewType.Single || score.Value.Type != NumberDataViewType.Single)
            return null;

        var predictions = new List<float>();
        var labels = new List<float>();
        using var cursor = scored.GetRowCursor([label.Value, score.Value]);
        var getLabel = cursor.GetGetter<float>(label.Value);
        var getScore = cursor.GetGetter<float>(score.Value);
        float y = default, s = default;
        while (cursor.MoveNext())
        {
            getLabel(ref y);
            if (float.IsNaN(y))
                continue;
            getScore(ref s);
            predictions.Add(s);
            labels.Add(y);
        }
        return (predictions.ToArray(), labels.ToArray());
    }
}
