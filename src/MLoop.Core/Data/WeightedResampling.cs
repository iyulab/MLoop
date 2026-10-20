using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// Weighted training by resampling: rows are drawn with replacement in proportion to a weight
/// column, as many as there were, so a learner that takes no weights fits the weighted
/// distribution. This is how MLoop trains on a weight column at all — AutoML accepts
/// <c>ColumnInformation.ExampleWeightColumnName</c> and does not pass it to the trainers it builds
/// (measured on AutoML 0.23: a fixture where the weight decides the answer gave the same model with
/// and without it).
/// </summary>
/// <remarks>
/// The weight is the relative frequency a row should have in the sample — for a row drawn with
/// probability π, that is 1/π. Each partition is resampled on its own, after the split, so the same
/// source row never sits in both the data a model is fitted on and the data it is chosen on.
/// </remarks>
public static class WeightedResampling
{
    /// <summary>The tasks weighted training applies to. The rest do not fit on rows of a table.</summary>
    public static bool Supports(string task) => task.ToLowerInvariant() is
        "binary-classification" or "multiclass-classification" or "regression";

    /// <summary>
    /// Reads the weights of <paramref name="data"/>, or throws with what is wrong with them: missing
    /// column, not numeric, negative, not finite, or all zero.
    /// </summary>
    public static double[] ReadWeights(IDataView data, string weightColumn)
    {
        var column = data.Schema.GetColumnOrNull(weightColumn)
            ?? throw new InvalidOperationException($"Weight column '{weightColumn}' is not in the data.");

        var weights = new List<double>();
        using (var cursor = data.GetRowCursor(new[] { column }))
        {
            ValueGetter<double> read = column.Type switch
            {
                NumberDataViewType t when t.RawType == typeof(float) => Wrap<float>(cursor, column, v => v),
                NumberDataViewType t when t.RawType == typeof(double) => Wrap<double>(cursor, column, v => v),
                NumberDataViewType t when t.RawType == typeof(int) => Wrap<int>(cursor, column, v => v),
                NumberDataViewType t when t.RawType == typeof(long) => Wrap<long>(cursor, column, v => v),
                _ => throw new InvalidOperationException(
                    $"Weight column '{weightColumn}' must be numeric; it was read as {column.Type}."),
            };
            var value = 0.0;
            while (cursor.MoveNext())
            {
                read(ref value);
                if (!double.IsFinite(value) || value < 0)
                    throw new InvalidOperationException(
                        $"Weight column '{weightColumn}' has {(double.IsFinite(value) ? "a negative" : "a missing or non-finite")} value in row {weights.Count + 1}; weights must be zero or positive numbers.");
                weights.Add(value);
            }
        }

        if (weights.Count > 0 && weights.All(w => w == 0))
            throw new InvalidOperationException($"Every value of weight column '{weightColumn}' is 0 — there is nothing to train on.");
        return weights.ToArray();
    }

    /// <summary>
    /// Rows drawn with replacement in proportion to <paramref name="weights"/>, as many as there are
    /// rows, ascending (a row drawn k times appears k times). Deterministic for a seed.
    /// </summary>
    public static int[] DrawRows(IReadOnlyList<double> weights, int seed)
    {
        var n = weights.Count;
        if (n == 0) return [];
        var cumulative = new double[n];
        var total = 0.0;
        for (var i = 0; i < n; i++)
            cumulative[i] = total += weights[i];

        var random = new Random(seed);
        var rows = new int[n];
        for (var k = 0; k < n; k++)
        {
            var u = random.NextDouble() * total;
            var i = Array.BinarySearch(cumulative, u);
            i = i < 0 ? ~i : i + 1;          // first cumulative strictly above u
            rows[k] = Math.Min(i, n - 1);     // a zero-weight row never wins: its cumulative equals its predecessor's
        }
        Array.Sort(rows);
        return rows;
    }

    /// <summary>
    /// <paramref name="data"/> resampled in proportion to its weight column, then shuffled so a row's
    /// copies are not adjacent.
    /// </summary>
    public static IDataView Resample(MLContext ml, IDataView data, string weightColumn, int seed)
    {
        var rows = DrawRows(ReadWeights(data, weightColumn), seed);
        return ml.Data.ShuffleRows(new RowSelection(data, rows), seed: seed);
    }

    private static ValueGetter<double> Wrap<T>(DataViewRowCursor cursor, DataViewSchema.Column column, Func<T, double> convert)
    {
        var getter = cursor.GetGetter<T>(column);
        T raw = default!;
        return (ref double value) => { getter(ref raw); value = convert(raw); };
    }

    /// <summary>
    /// A view of <c>source</c>'s rows at the given ascending positions, repeats allowed — read with
    /// one forward pass of the source, so it needs nothing but a cursor.
    /// </summary>
    private sealed class RowSelection(IDataView source, int[] rows) : IDataView
    {
        public bool CanShuffle => false;
        public DataViewSchema Schema => source.Schema;
        public long? GetRowCount() => rows.Length;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
            => new Cursor(source.GetRowCursor(columnsNeeded), rows);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class Cursor(DataViewRowCursor inner, int[] rows) : DataViewRowCursor
        {
            private long _position = -1;
            private long _innerPosition = -1;

            public override DataViewSchema Schema => inner.Schema;
            public override long Position => _position;
            public override long Batch => 0;

            public override bool MoveNext()
            {
                if (_position + 1 >= rows.Length)
                {
                    _position = rows.Length;
                    return false;
                }
                _position++;
                var target = rows[_position];
                while (_innerPosition < target)
                {
                    if (!inner.MoveNext())
                        throw new InvalidOperationException("The source view has fewer rows than were drawn from it.");
                    _innerPosition++;
                }
                return true;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => inner.GetGetter<TValue>(column);
            public override bool IsColumnActive(DataViewSchema.Column column) => inner.IsColumnActive(column);

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

            protected override void Dispose(bool disposing)
            {
                if (disposing) inner.Dispose();
                base.Dispose(disposing);
            }
        }
    }
}
