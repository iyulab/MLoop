using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// How a missing number in a table is read: an empty (or whitespace-only) numeric field is
/// <see cref="float.NaN"/>, never zero. Every <see cref="TextLoader.Options"/> MLoop builds sets
/// <see cref="TextLoader.Options.MissingRealsAsNaNs"/> — ML.NET's default reads the field as 0, a real value
/// inside most columns' range, so before this a sensor that dropped out trained and predicted as a reading
/// of zero and the featurizer's missing-value replacement never had anything to replace.
/// <c>MissingNumberLoadingContractTests</c> keeps every loader in line.
/// </summary>
/// <remarks>
/// What a model does with a missing number is the model's: the trainer search replaces it by the column's
/// mean and marks it (<c>MLoop.Search.Candidates.Featurizer</c>), the clustering, anomaly-detection and
/// ranking pipelines replace it by the mean, and a time series refuses a gap (<see cref="RefuseGapsInSeries"/>) —
/// dropping the row would close up the time between its neighbours, and no fill is right for every series.
/// </remarks>
public static class MissingNumbers
{
    /// <summary>The number of missing values in a numeric column (each slot of a vector counts).</summary>
    public static long CountIn(IDataView data, string column)
    {
        ArgumentNullException.ThrowIfNull(data);
        var col = data.Schema[column];
        long count = 0;
        using var cursor = data.GetRowCursor([col]);
        if (col.Type is VectorDataViewType { ItemType: var item } && item == NumberDataViewType.Single)
        {
            var get = cursor.GetGetter<VBuffer<float>>(col);
            VBuffer<float> value = default;
            while (cursor.MoveNext())
            {
                get(ref value);
                foreach (var x in value.GetValues())
                    if (float.IsNaN(x)) count++;
            }
        }
        else if (col.Type == NumberDataViewType.Single)
        {
            var get = cursor.GetGetter<float>(col);
            float value = default;
            while (cursor.MoveNext())
            {
                get(ref value);
                if (float.IsNaN(value)) count++;
            }
        }
        else if (col.Type == NumberDataViewType.Double)
        {
            var get = cursor.GetGetter<double>(col);
            double value = default;
            while (cursor.MoveNext())
            {
                get(ref value);
                if (double.IsNaN(value)) count++;
            }
        }
        return count;
    }

    /// <summary>
    /// The 1-based positions (data rows, the header not counted) of the first <paramref name="limit"/> missing values of a
    /// single-number column — where in the table a gap is, so a refusal can name it.
    /// </summary>
    public static IReadOnlyList<long> FirstGapRows(IDataView data, string column, int limit)
    {
        ArgumentNullException.ThrowIfNull(data);
        var col = data.Schema[column];
        var rows = new List<long>();
        using var cursor = data.GetRowCursor([col]);
        if (col.Type == NumberDataViewType.Single)
        {
            var get = cursor.GetGetter<float>(col);
            float value = default;
            while (rows.Count < limit && cursor.MoveNext())
            {
                get(ref value);
                if (float.IsNaN(value)) rows.Add(cursor.Position + 1);
            }
        }
        else if (col.Type == NumberDataViewType.Double)
        {
            var get = cursor.GetGetter<double>(col);
            double value = default;
            while (rows.Count < limit && cursor.MoveNext())
            {
                get(ref value);
                if (double.IsNaN(value)) rows.Add(cursor.Position + 1);
            }
        }
        return rows;
    }

    /// <summary>
    /// Throws when the series in <paramref name="column"/> has a gap — a time-series model reads the values in
    /// order, and a missing one would reach it as NaN.
    /// </summary>
    /// <exception cref="SeriesGapException">The column holds a missing value.</exception>
    public static void RefuseGapsInSeries(IDataView data, string column)
    {
        var gaps = CountIn(data, column);
        if (gaps > 0)
        {
            var rows = FirstGapRows(data, column, 5);
            var where = rows.Count == 0 ? "" : gaps == 1
                ? $" (data row {rows[0]})"
                : $" (the first at data row{(rows.Count == 1 ? "" : "s")} {string.Join(", ", rows)})";
            throw new SeriesGapException(
                $"The series '{column}' has {gaps} missing value{(gaps == 1 ? "" : "s")}{where}. A time-series model reads every point in order, " +
                "and dropping a row would close up the time around it. Fill the gaps first — for example " +
                $"`mloop prep plan --set fill-missing:forward --columns {column}` — or fill them in the file.");
        }
    }
}
