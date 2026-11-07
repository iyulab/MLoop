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
    /// <summary>
    /// The rows of <paramref name="data"/> in memory when <paramref name="rows"/> of them fit the search's cache limit
    /// (<see cref="TrialRunner.CacheCellLimit"/> cells, a text cell weighing <see cref="TrialRunner.TextCellWeight"/>), else the view itself. For scored rows
    /// that several steps read: a scored view is lazy, so each step that reads it scores the rows again — measured, four reads of 200,000
    /// rows through a model of 4,607 trees cost 43 s.
    /// </summary>
    public static IDataView WhenItFits(MLContext mlContext, IDataView data, long rows)
    {
        ArgumentNullException.ThrowIfNull(data);
        return rows * CellsPerRow(data.Schema) <= TrialRunner.CacheCellLimit ? Of(mlContext, data) : data;
    }

    // A vector counts its length (100 where the length is not known), a text cell its weight, any other cell one.
    internal static long CellsPerRow(DataViewSchema schema)
        => schema.Sum(column => column.Type switch
        {
            Microsoft.ML.Data.VectorDataViewType vector => vector.Size > 0 ? vector.Size : 100,
            Microsoft.ML.Data.TextDataViewType => (long)TrialRunner.TextCellWeight,
            _ => 1L,
        });

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
