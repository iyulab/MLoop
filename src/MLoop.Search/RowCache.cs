using Microsoft.ML;

namespace MLoop.Search;

/// <summary>
/// Rows kept in memory, filled before anything reads them.
/// </summary>
/// <remarks>
/// <see cref="DataOperationsCatalog.Cache"/> fills its copy in the background while the first cursor reads it, and a
/// reader that asks for a vector column in that window can see a row that is not written yet: the read fails with
/// <c>If specified, must be non-negative (Parameter 'valuesCount')</c>. Measured with ML.NET alone (a 4,000,000-row
/// text file, four numeric columns read as one vector, on arm64): the read failed in more than a quarter of the runs
/// and in none once a cursor had gone over every column without reading a value first. The pass costs nothing
/// extra — the copy has to be filled anyway.
/// </remarks>
public static class RowCache
{
    /// <summary>The rows of <paramref name="data"/> in memory, completely filled.</summary>
    public static IDataView Of(MLContext mlContext, IDataView data)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(data);

        var cached = mlContext.Data.Cache(data);
        using var cursor = cached.GetRowCursor(cached.Schema);
        while (cursor.MoveNext())
        {
            // a row at a time, no value read: waiting for the fill is all this pass does
        }
        return cached;
    }
}
