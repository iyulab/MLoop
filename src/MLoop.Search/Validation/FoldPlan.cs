using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Search.Validation;

/// <summary>
/// Which rows of a training partition validate which fold: <see cref="Folds"/> folds averaged when the
/// partition is small, one holdout fold when it is large. A classification label is spread
/// over the folds in proportion — every fold sees every class it can — and the assignment depends on the
/// seed alone.
/// </summary>
public sealed class FoldPlan
{
    /// <summary>Rows from which a single holdout replaces the averaged folds.</summary>
    public const int HoldoutThreshold = 15_000;

    /// <summary>Folds averaged below <see cref="HoldoutThreshold"/>.</summary>
    public const int DefaultFolds = 5;

    /// <summary>The holdout's share at and above the threshold: one in ten rows.</summary>
    public const int HoldoutParts = 10;

    private FoldPlan(int[] foldOf, int parts, IReadOnlyList<int> validated, int[]? holdoutSliceOf)
    {
        FoldOf = foldOf;
        Parts = parts;
        Validated = validated;
        HoldoutSliceOf = holdoutSliceOf;
    }

    /// <summary>
    /// For a holdout, the slice (0 to <see cref="Validator.HoldoutSlices"/> − 1) of each holdout row, the rows taken
    /// in their order in the partition — what the holdout's standard error is measured over. Dealt with the holdout
    /// itself, so a slice is a stratified random share of it whatever order the file is in. Null for k-fold.
    /// </summary>
    public IReadOnlyList<int>? HoldoutSliceOf { get; }

    /// <summary>The part each row (by position) belongs to.</summary>
    public IReadOnlyList<int> FoldOf { get; }

    /// <summary>How many parts the rows are split into.</summary>
    public int Parts { get; }

    /// <summary>The parts that are validated on — every part for k-fold, the first for a holdout.</summary>
    public IReadOnlyList<int> Validated { get; }

    /// <summary>How many folds are averaged.</summary>
    public int Folds => Validated.Count;

    /// <summary>Whether this is a single holdout rather than averaged folds.</summary>
    public bool IsHoldout => Parts != Validated.Count;

    /// <summary>
    /// The plan for <paramref name="data"/>: stratified by <paramref name="labelColumn"/> when
    /// <paramref name="stratify"/> (classification), shuffled otherwise (regression).
    /// </summary>
    public static FoldPlan For(IDataView data, string labelColumn, bool stratify, int seed, int folds = DefaultFolds)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (folds < 2)
            throw new ArgumentOutOfRangeException(nameof(folds), "At least two folds.");
        var keys = LabelKeys(data, labelColumn);
        var holdout = keys.Length >= HoldoutThreshold;
        var parts = holdout ? HoldoutParts : folds;
        var (foldOf, dealt) = Deal(keys, stratify, parts, seed);
        return new FoldPlan(foldOf, parts, holdout ? [0] : [.. Enumerable.Range(0, parts)], holdout ? Slices(foldOf, dealt) : null);
    }

    // The holdout's rows in the order they were dealt — shuffled within each class, the classes one after another —
    // dealt round the slices as the rows were dealt round the parts, then read back in the partition's order. Cut
    // in the partition's order instead, a file sorted by its label gave slices of one class: the AUC of a slice was
    // undefined and the others' spread measured the sort (accuracy's standard error 0.02 on 17,000 rows, where the
    // binomial one is 0.0005).
    private static int[] Slices(int[] foldOf, int[] dealt)
    {
        var sliceOfRow = new Dictionary<int, int>();
        var next = 0;
        foreach (var row in dealt)
            if (foldOf[row] == 0)
                sliceOfRow[row] = next++ % Validator.HoldoutSlices;
        return [.. Enumerable.Range(0, foldOf.Length).Where(row => foldOf[row] == 0).Select(row => sliceOfRow[row])];
    }

    /// <summary>
    /// The assignment itself: each class's rows shuffled by <paramref name="seed"/> and dealt round
    /// the parts, the deal carried on from one class to the next so the parts stay even.
    /// </summary>
    internal static int[] Assign(IReadOnlyList<string> keys, bool stratify, int parts, int seed)
        => Deal(keys, stratify, parts, seed).FoldOf;

    // The assignment and the order the rows were dealt in.
    private static (int[] FoldOf, int[] Dealt) Deal(IReadOnlyList<string> keys, bool stratify, int parts, int seed)
    {
        var random = new Random(seed);
        var foldOf = new int[keys.Count];
        var order = new List<int>(keys.Count);
        var groups = stratify
            ? keys.Select((k, i) => (k, i)).GroupBy(x => x.k, StringComparer.Ordinal).OrderBy(g => g.Key, StringComparer.Ordinal)
                .Select(g => g.Select(x => x.i).ToArray())
            : [Enumerable.Range(0, keys.Count).ToArray()];
        var dealt = 0;
        foreach (var rows in groups)
        {
            for (var i = rows.Length - 1; i > 0; i--)
            {
                var j = random.Next(i + 1);
                (rows[i], rows[j]) = (rows[j], rows[i]);
            }
            foreach (var row in rows)
            {
                foldOf[row] = dealt++ % parts;
                order.Add(row);
            }
        }
        return (foldOf, [.. order]);
    }

    private static string[] LabelKeys(IDataView data, string labelColumn)
    {
        var column = data.Schema.GetColumnOrNull(labelColumn)
            ?? throw new ArgumentException($"The data has no label column '{labelColumn}'.", nameof(labelColumn));
        return column.Type switch
        {
            BooleanDataViewType => [.. data.GetColumn<bool>(column).Select(v => v ? "1" : "0")],
            TextDataViewType => [.. data.GetColumn<ReadOnlyMemory<char>>(column).Select(v => v.ToString())],
            KeyDataViewType => [.. data.GetColumn<uint>(column).Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture))],
            NumberDataViewType n when n == NumberDataViewType.Single
                => [.. data.GetColumn<float>(column).Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture))],
            NumberDataViewType n when n == NumberDataViewType.Double
                => [.. data.GetColumn<double>(column).Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture))],
            _ => throw new NotSupportedException($"Label column '{labelColumn}' of type {column.Type} cannot be split into folds."),
        };
    }
}
