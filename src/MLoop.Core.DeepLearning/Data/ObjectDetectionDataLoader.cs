using Microsoft.ML;
using MLoop.Core.Contracts;

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
public sealed class ObjectDetectionDataLoader : DataProviderBase
{
    public ObjectDetectionDataLoader(MLContext mlContext, Action<string>? log = null)
        : base(mlContext, log)
    {
    }

    public override IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null)
    {
        // YOLO's images/ + labels/*.txt layout is the most distinctive signal; prefer it,
        // otherwise fall back to the COCO parser (which gives a clear error if no JSON exists).
        IDataProvider inner = YoloDataLoader.IsYoloDirectory(filePath)
            ? new YoloDataLoader(_mlContext, _log)
            : new CocoDataLoader(_mlContext, _log);

        _log($"[Info] Object-detection format: {(inner is YoloDataLoader ? "YOLO" : "COCO")}.");
        return inner.LoadData(filePath, labelColumn, taskType, preserveColumns);
    }
}
