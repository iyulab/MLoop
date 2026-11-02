using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Search.Validation;

/// <summary>
/// A classifier's answers on rows it was not fitted on — every validated part of a <see cref="FoldPlan"/>, scored by
/// the model fitted without it — and each row's weight. Binary: the raw <c>Score</c>, the <c>Probability</c> when the
/// trainer gives one, and the label. Multiclass: each row's class probabilities (<c>Score</c>, one slot per class)
/// and the slot of its true class. What a calibration of the refit model is fitted on: the refit has seen every
/// training row, so its own scores on them are optimistic, and the test partition is the run's report and fits nothing.
/// </summary>
/// <remarks>
/// A holdout plan gives the holdout's rows only; folds give every row of the partition once.
/// </remarks>
public sealed class OutOfFoldScores
{
    private readonly List<float> _score = [];
    private readonly List<float>? _probability;
    private readonly List<bool> _label = [];
    private readonly List<float[]> _classProbabilities = [];
    private readonly List<int> _class = [];
    private readonly List<float> _target = [];
    private readonly List<float> _weight = [];

    private OutOfFoldScores(bool multiclass, bool hasProbability, bool regression = false)
    {
        IsMulticlass = multiclass;
        IsRegression = regression;
        _probability = hasProbability ? [] : null;
    }

    /// <summary>Whether the rows are a multiclass model's (<see cref="ClassProbabilities"/>) rather than a binary one's.</summary>
    public bool IsMulticlass { get; }

    /// <summary>Whether the rows are a regression's (<see cref="Score"/> the predicted value, <see cref="Target"/> the true one).</summary>
    public bool IsRegression { get; }

    /// <summary>Binary: the model's raw score of each row. Regression: its predicted value.</summary>
    public IReadOnlyList<float> Score => _score;

    /// <summary>Regression: the true value of each row.</summary>
    public IReadOnlyList<float> Target => _target;

    /// <summary>Binary: the model's probability of each row, or <see langword="null"/> when its trainer gives none.</summary>
    public IReadOnlyList<float>? Probability => _probability;

    /// <summary>Binary: whether each row is positive.</summary>
    public IReadOnlyList<bool> Label => _label;

    /// <summary>Multiclass: each row's class probabilities, slot i the class whose label key is i + 1.</summary>
    public IReadOnlyList<float[]> ClassProbabilities => _classProbabilities;

    /// <summary>Multiclass: the slot of each row's true class.</summary>
    public IReadOnlyList<int> Class => _class;

    /// <summary>Each row's weight — 1 when the search is unweighted.</summary>
    public IReadOnlyList<float> Weight => _weight;

    /// <summary>How many rows.</summary>
    public int Count => _weight.Count;

    /// <summary>
    /// Reads a scored validation part into <paramref name="into"/> (a new collection when <see langword="null"/>), or
    /// returns <see langword="null"/> when the part does not carry a classifier's answers — a boolean label and a
    /// single-valued <c>Score</c>, or a key label and a <c>Score</c> vector of its width: a regression has nothing to add.
    /// A multiclass row whose label is missing (key 0) is left out.
    /// </summary>
    /// <param name="scored">A validation part, scored by the model fitted without it.</param>
    /// <param name="label">The label column.</param>
    /// <param name="weight">The column that weights rows, or <see langword="null"/>.</param>
    /// <param name="into">The collection the parts of one trial are read into, or <see langword="null"/> to start one.</param>
    /// <param name="regression">
    /// Also read a regression — a numeric label and a single-valued <c>Score</c>. Off by default: a classifier's calibration is
    /// what these answers were kept for, and a ranking's score has the same shape, so only a caller that knows its task is a
    /// regression asks (an ensemble's weights are fitted on them).
    /// </param>
    public static OutOfFoldScores? Read(IDataView scored, string label, string? weight, OutOfFoldScores? into = null, bool regression = false)
    {
        ArgumentNullException.ThrowIfNull(scored);
        var schema = scored.Schema;
        if (schema.GetColumnOrNull(label) is not { } labelColumn || schema.GetColumnOrNull("Score") is not { } scoreColumn)
            return null;
        var weightColumn = weight is not null && schema.GetColumnOrNull(weight) is { Type: NumberDataViewType { RawType: var wType } } w
            && wType == typeof(float) ? w : (DataViewSchema.Column?)null;

        if (labelColumn.Type is BooleanDataViewType && scoreColumn.Type == NumberDataViewType.Single)
        {
            var probabilityColumn = schema.GetColumnOrNull("Probability") is { } p && p.Type == NumberDataViewType.Single
                ? p : (DataViewSchema.Column?)null;
            var rows = into ?? new OutOfFoldScores(multiclass: false, hasProbability: probabilityColumn is not null);
            if (rows.IsMulticlass || rows.IsRegression || (rows._probability is not null) != (probabilityColumn is not null))
                return null;   // the folds of one pipeline give the same columns; this guards a caller that mixed two
            ReadBinary(scored, labelColumn, scoreColumn, probabilityColumn, weightColumn, rows);
            return rows;
        }

        if (labelColumn.Type is KeyDataViewType key
            && scoreColumn.Type is VectorDataViewType { ItemType: var item, Size: var width } && item == NumberDataViewType.Single
            && width > 0 && (ulong)width == key.Count)
        {
            var rows = into ?? new OutOfFoldScores(multiclass: true, hasProbability: false);
            if (!rows.IsMulticlass)
                return null;
            ReadMulticlass(scored, labelColumn, scoreColumn, weightColumn, rows);
            return rows;
        }

        if (regression && labelColumn.Type == NumberDataViewType.Single && scoreColumn.Type == NumberDataViewType.Single)
        {
            var rows = into ?? new OutOfFoldScores(multiclass: false, hasProbability: false, regression: true);
            if (!rows.IsRegression)
                return null;
            ReadRegression(scored, labelColumn, scoreColumn, weightColumn, rows);
            return rows;
        }
        return null;
    }

    /// <summary>
    /// The rows of a blend: <paramref name="reference"/>'s labels (or targets) and weights, with the blend's own answer for
    /// every row — a probability (its <see cref="Score"/> the log-odds of it, zero where a binary evaluator splits) or
    /// a regression's value. What a calibration of the blend is fitted on, in the shape the single trial's rows have.
    /// </summary>
    /// <param name="reference">Any member's rows — the truth and the weights are the same for every member.</param>
    /// <param name="answer">The blend's answer for each row, in the rows' order.</param>
    /// <exception cref="ArgumentException">A multiclass reference, or an answer that is not one per row.</exception>
    public static OutOfFoldScores Blended(OutOfFoldScores reference, IReadOnlyList<float> answer)
    {
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(answer);
        if (reference.IsMulticlass)
            throw new ArgumentException("A multiclass model's rows are not blended.", nameof(reference));
        if (answer.Count != reference.Count)
            throw new ArgumentException("The blend must answer every row once.", nameof(answer));
        var rows = new OutOfFoldScores(multiclass: false, hasProbability: !reference.IsRegression, regression: reference.IsRegression);
        rows._weight.AddRange(reference._weight);
        if (reference.IsRegression)
        {
            rows._target.AddRange(reference._target);
            rows._score.AddRange(answer);
        }
        else
        {
            rows._label.AddRange(reference._label);
            rows._probability!.AddRange(answer);
            rows._score.AddRange(answer.Select(p => MathF.Log(Math.Max(p, 1e-6f) / Math.Max(1 - p, 1e-6f))));
        }
        return rows;
    }

    private static void ReadBinary(IDataView scored, DataViewSchema.Column labelColumn, DataViewSchema.Column scoreColumn,
        DataViewSchema.Column? probabilityColumn, DataViewSchema.Column? weightColumn, OutOfFoldScores rows)
    {
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
            getWeight?.Invoke(ref rowWeight);
            rows._weight.Add(rowWeight);
        }
    }

    private static void ReadMulticlass(IDataView scored, DataViewSchema.Column labelColumn, DataViewSchema.Column scoreColumn,
        DataViewSchema.Column? weightColumn, OutOfFoldScores rows)
    {
        var columns = new List<DataViewSchema.Column> { labelColumn, scoreColumn };
        if (weightColumn is { } wc) columns.Add(wc);
        using var cursor = scored.GetRowCursor(columns);
        var getLabel = cursor.GetGetter<uint>(labelColumn);
        var getScore = cursor.GetGetter<VBuffer<float>>(scoreColumn);
        var getWeight = weightColumn is { } wcol ? cursor.GetGetter<float>(wcol) : null;
        uint key = default;
        VBuffer<float> scores = default;
        float rowWeight = 1;
        while (cursor.MoveNext())
        {
            getLabel(ref key);
            getWeight?.Invoke(ref rowWeight);
            if (key == 0)
                continue;   // a missing label
            getScore(ref scores);
            rows._classProbabilities.Add(scores.DenseValues().ToArray());
            rows._class.Add((int)key - 1);
            rows._weight.Add(rowWeight);
        }
    }

    private static void ReadRegression(IDataView scored, DataViewSchema.Column labelColumn, DataViewSchema.Column scoreColumn,
        DataViewSchema.Column? weightColumn, OutOfFoldScores rows)
    {
        var columns = new List<DataViewSchema.Column> { labelColumn, scoreColumn };
        if (weightColumn is { } wc) columns.Add(wc);
        using var cursor = scored.GetRowCursor(columns);
        var getLabel = cursor.GetGetter<float>(labelColumn);
        var getScore = cursor.GetGetter<float>(scoreColumn);
        var getWeight = weightColumn is { } wcol ? cursor.GetGetter<float>(wcol) : null;
        float target = default, score = default, rowWeight = 1;
        while (cursor.MoveNext())
        {
            getLabel(ref target);
            getWeight?.Invoke(ref rowWeight);
            if (float.IsNaN(target))
                continue;   // a missing label
            getScore(ref score);
            rows._target.Add(target);
            rows._score.Add(score);
            rows._weight.Add(rowWeight);
        }
    }
}
