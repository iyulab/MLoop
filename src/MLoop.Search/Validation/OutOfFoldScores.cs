using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Search.Validation;

/// <summary>
/// A binary pipeline's answers on rows it was not fitted on — every validated part of a <see cref="FoldPlan"/>, scored
/// by the model fitted without it: the raw <c>Score</c>, the <c>Probability</c> when the trainer gives one, the label
/// and the row's weight. What a calibration of the refit model is fitted on: the refit has seen every training row,
/// so its own scores on them are optimistic, and the test partition is the run's report and fits nothing.
/// </summary>
/// <remarks>
/// A holdout plan gives the holdout's rows only; folds give every row of the partition once.
/// </remarks>
public sealed class OutOfFoldScores
{
    private readonly List<float> _score = [];
    private readonly List<float>? _probability;
    private readonly List<bool> _label = [];
    private readonly List<float> _weight = [];

    private OutOfFoldScores(bool hasProbability) => _probability = hasProbability ? [] : null;

    /// <summary>The model's raw score of each row.</summary>
    public IReadOnlyList<float> Score => _score;

    /// <summary>The model's probability of each row, or <see langword="null"/> when its trainer gives none.</summary>
    public IReadOnlyList<float>? Probability => _probability;

    /// <summary>Whether each row is positive.</summary>
    public IReadOnlyList<bool> Label => _label;

    /// <summary>Each row's weight — 1 when the search is unweighted.</summary>
    public IReadOnlyList<float> Weight => _weight;

    /// <summary>How many rows.</summary>
    public int Count => _score.Count;

    /// <summary>
    /// Reads a scored validation part into <paramref name="into"/> (a new collection when <see langword="null"/>), or
    /// returns <see langword="null"/> when the part does not carry what a binary calibration reads — a boolean label
    /// and a single-valued <c>Score</c>: a multiclass or regression pipeline has nothing to add here.
    /// </summary>
    public static OutOfFoldScores? Read(IDataView scored, string label, string? weight, OutOfFoldScores? into = null)
    {
        ArgumentNullException.ThrowIfNull(scored);
        var schema = scored.Schema;
        if (schema.GetColumnOrNull(label) is not { Type: BooleanDataViewType } labelColumn
            || schema.GetColumnOrNull("Score") is not { Type: NumberDataViewType { RawType: var scoreType } } scoreColumn
            || scoreType != typeof(float))
            return null;
        var probabilityColumn = schema.GetColumnOrNull("Probability") is { Type: NumberDataViewType { RawType: var pType } } p
            && pType == typeof(float) ? p : (DataViewSchema.Column?)null;
        var weightColumn = weight is not null && schema.GetColumnOrNull(weight) is { Type: NumberDataViewType { RawType: var wType } } w
            && wType == typeof(float) ? w : (DataViewSchema.Column?)null;
        var rows = into ?? new OutOfFoldScores(probabilityColumn is not null);
        if ((rows._probability is not null) != (probabilityColumn is not null))
            return null;   // the folds of one pipeline give the same columns; this guards a caller that mixed two

        var columns = new List<DataViewSchema.Column> { labelColumn, scoreColumn };
        if (probabilityColumn is { } pc) columns.Add(pc);
        if (weightColumn is { } wc) columns.Add(wc);
        using var cursor = scored.GetRowCursor(columns);
        var getLabel = cursor.GetGetter<bool>(labelColumn);
        var getScore = cursor.GetGetter<float>(scoreColumn);
        var getProbability = probabilityColumn is { } pcol ? cursor.GetGetter<float>(pcol) : null;
        var getWeight = weightColumn is { } wcol ? cursor.GetGetter<float>(wcol) : null;
        bool y = default;
        float s = default, q = default, rowWeight = 1;
        while (cursor.MoveNext())
        {
            getLabel(ref y);
            getScore(ref s);
            rows._label.Add(y);
            rows._score.Add(s);
            if (getProbability is not null)
            {
                getProbability(ref q);
                rows._probability!.Add(q);
            }
            if (getWeight is not null)
                getWeight(ref rowWeight);
            rows._weight.Add(rowWeight);
        }
        return rows;
    }
}
