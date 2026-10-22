using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// A weight column: the tasks it applies to, and its values, checked. The search gives the weights to the
/// trainers and scores its folds by them; the evaluators weigh the test rows by them
/// (<see cref="Evaluation.WeightedMetrics"/>).
/// </summary>
/// <remarks>
/// The weight is the relative frequency a row should have — for a row drawn with probability π, that is 1/π.
/// </remarks>
public static class RowWeights
{
    /// <summary>The tasks weighted training applies to. The rest do not fit on rows of a table.</summary>
    public static bool Supports(string task) => task.ToLowerInvariant() is
        "binary-classification" or "multiclass-classification" or "regression";

    /// <summary>
    /// Reads the weights of <paramref name="data"/>, or throws with what is wrong with them: missing
    /// column, not numeric, negative, not finite, or all zero.
    /// </summary>
    public static double[] Read(IDataView data, string weightColumn)
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
                // A loader that does not treat the column as a feature may read it as text; the numbers are still numbers.
                TextDataViewType => Wrap<ReadOnlyMemory<char>>(cursor, column, v =>
                    double.TryParse(v.Span, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d)
                        ? d : double.NaN),
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

    private static ValueGetter<double> Wrap<T>(DataViewRowCursor cursor, DataViewSchema.Column column, Func<T, double> convert)
    {
        var getter = cursor.GetGetter<T>(column);
        T raw = default!;
        return (ref double value) => { getter(ref raw); value = convert(raw); };
    }
}
