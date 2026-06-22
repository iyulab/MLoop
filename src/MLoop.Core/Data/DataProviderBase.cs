using Microsoft.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// Shared <see cref="IDataProvider"/> plumbing for every loader. Concrete loaders implement
/// only <see cref="LoadData"/> — the format-specific part — while label validation, schema
/// extraction, and train/test splitting are format-independent and live here once.
/// </summary>
/// <remarks>
/// All loaders produce a column-oriented <see cref="IDataView"/>, so once the data is loaded
/// the remaining operations only depend on the schema, not on the on-disk format (CSV directory,
/// image folder, COCO JSON, or YOLO labels). Centralizing them removes the duplication that grew
/// as the loader count rose to five.
/// </remarks>
public abstract class DataProviderBase : IDataProvider
{
    /// <summary>ML.NET context shared with the trainer pipeline.</summary>
    protected readonly MLContext _mlContext;

    /// <summary>Diagnostic sink; defaults to <see cref="Console.WriteLine(string)"/>.</summary>
    protected readonly Action<string> _log;

    protected DataProviderBase(MLContext mlContext, Action<string>? log = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _log = log ?? Console.WriteLine;
    }

    /// <inheritdoc />
    public abstract IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null);

    /// <inheritdoc />
    public virtual bool ValidateLabelColumn(IDataView data, string labelColumn) =>
        data.Schema.Any(col => col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase));

    /// <inheritdoc />
    public virtual DataSchema GetSchema(IDataView data)
    {
        var columns = data.Schema
            .Select(column => new ColumnInfo
            {
                Name = column.Name,
                Type = DataTypeHelper.GetFriendlyTypeName(column.Type),
                IsLabel = false
            })
            .ToList();

        return new DataSchema
        {
            Columns = columns,
            RowCount = GetRowCount(data)
        };
    }

    /// <inheritdoc />
    public virtual (IDataView trainSet, IDataView testSet) SplitData(IDataView data, double testFraction = 0.2)
    {
        // testFraction == 0 → use all data for both sets (AutoML relies on cross-validation).
        if (testFraction <= 0)
            return (data, data);

        if (testFraction >= 1)
            throw new ArgumentException(
                "testFraction must be between 0 and 1 (exclusive).", nameof(testFraction));

        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42); // Fixed seed for reproducibility
        return (split.TrainSet, split.TestSet);
    }

    /// <summary>
    /// Returns the row count, using the cheap <see cref="IDataView.GetRowCount"/> when the view
    /// can report it and falling back to a full cursor scan otherwise (e.g. lazily-loaded CSV).
    /// </summary>
    protected static int GetRowCount(IDataView data)
    {
        long? count = data.GetRowCount();
        if (count.HasValue)
            return (int)count.Value;

        int rowCount = 0;
        using var cursor = data.GetRowCursor(data.Schema);
        while (cursor.MoveNext())
            rowCount++;
        return rowCount;
    }
}
