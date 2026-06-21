using Microsoft.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// Object-detection data loader that auto-detects the on-disk annotation format and delegates
/// to the matching parser:
/// <list type="bullet">
///   <item><b>YOLO</b> (<c>images/</c> + <c>labels/*.txt</c>) — the layout most KAMP image
///     detection datasets ship in — via <see cref="YoloDataLoader"/>.</item>
///   <item><b>COCO</b> (an <c>annotations.json</c> / Roboflow <c>_annotations.coco.json</c>) via
///     <see cref="CocoDataLoader"/>.</item>
/// </list>
/// Both parsers emit the identical <see cref="IDataView"/> schema
/// (<c>ImagePath</c>, <c>Label</c> vector, <c>BoundingBoxes</c> vector), so split/schema/validate
/// are format-independent and implemented once here.
/// </summary>
public sealed class ObjectDetectionDataLoader : IDataProvider
{
    private readonly MLContext _mlContext;
    private readonly Action<string> _log;

    public ObjectDetectionDataLoader(MLContext mlContext, Action<string>? log = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _log = log ?? Console.WriteLine;
    }

    public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null)
    {
        // YOLO's images/ + labels/*.txt layout is the most distinctive signal; prefer it,
        // otherwise fall back to the COCO parser (which gives a clear error if no JSON exists).
        IDataProvider inner = YoloDataLoader.IsYoloDirectory(filePath)
            ? new YoloDataLoader(_mlContext, _log)
            : new CocoDataLoader(_mlContext, _log);

        _log($"[Info] Object-detection format: {(inner is YoloDataLoader ? "YOLO" : "COCO")}.");
        return inner.LoadData(filePath, labelColumn, taskType, preserveColumns);
    }

    public bool ValidateLabelColumn(IDataView data, string labelColumn) =>
        data.Schema.Any(c => c.Name == labelColumn);

    public DataSchema GetSchema(IDataView data)
    {
        var columns = data.Schema
            .Select(column => new ColumnInfo
            {
                Name = column.Name,
                Type = DataTypeHelper.GetFriendlyTypeName(column.Type),
                IsLabel = false
            })
            .ToList();

        return new DataSchema { Columns = columns, RowCount = (int)(data.GetRowCount() ?? 0) };
    }

    public (IDataView trainSet, IDataView testSet) SplitData(IDataView data, double testFraction = 0.2)
    {
        if (testFraction <= 0)
            return (data, data);
        if (testFraction >= 1)
            throw new ArgumentException("testFraction must be between 0 and 1 (exclusive).", nameof(testFraction));

        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42);
        return (split.TrainSet, split.TestSet);
    }
}
