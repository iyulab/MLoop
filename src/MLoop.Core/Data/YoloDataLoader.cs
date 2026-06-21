using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// Loads an object-detection dataset in <b>YOLO</b> format — the layout most KAMP image
/// detection datasets ship in:
/// <code>
/// datasets/yolo/
/// ├── images/   13575.jpg 13576.jpg ...
/// ├── labels/   13575.txt 13576.txt ...   (one .txt per image, matched by file stem)
/// └── classes.txt   (optional; one class name per line, indexed by id)
/// </code>
/// Each label line is <c>class_id x_center y_center width height</c> with all box values
/// <b>normalized</b> to <c>[0,1]</c> relative to the image. This loader reads each image's
/// pixel dimensions (via <see cref="MLImage"/>) and converts the boxes to the absolute
/// <c>x0 y0 x1 y1</c> order the ML.NET <c>ObjectDetection</c> trainer expects — producing the
/// exact same <see cref="IDataView"/> schema as <see cref="CocoDataLoader"/>
/// (<c>ImagePath</c>, <c>Label</c> vector, <c>BoundingBoxes</c> vector).
/// </summary>
public sealed class YoloDataLoader : IDataProvider
{
    /// <summary>Name of the column holding the absolute image file path.</summary>
    public const string ImagePathColumn = CocoDataLoader.ImagePathColumn;

    /// <summary>Default label column name (vector of class names).</summary>
    public const string DefaultLabelColumn = CocoDataLoader.DefaultLabelColumn;

    /// <summary>Name of the column holding the flattened bounding boxes (x0 y0 x1 y1 per object).</summary>
    public const string BoundingBoxColumn = CocoDataLoader.BoundingBoxColumn;

    private static readonly HashSet<string> SupportedImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp"
    };

    private static readonly string[] ConventionalClassFiles = { "classes.txt", "obj.names", "names.txt" };

    private readonly MLContext _mlContext;
    private readonly Action<string> _log;

    public YoloDataLoader(MLContext mlContext, Action<string>? log = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _log = log ?? Console.WriteLine;
    }

    /// <summary>
    /// Detects a YOLO layout under <paramref name="path"/> (an <c>images/</c> + <c>labels/</c>
    /// pair, or images and .txt labels in a single directory).
    /// </summary>
    public static bool IsYoloDirectory(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            return false;

        var (imageDir, labelDir) = ResolveImageAndLabelDirs(path);
        if (imageDir == null || labelDir == null)
            return false;

        // At least one label .txt must exist for a YOLO dataset.
        return Directory.EnumerateFiles(labelDir, "*.txt", SearchOption.TopDirectoryOnly).Any();
    }

    public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("YOLO dataset path must be provided.", nameof(filePath));

        if (!Directory.Exists(filePath))
            throw new DirectoryNotFoundException(
                $"YOLO dataset directory not found: {filePath}\n" +
                "Object detection (YOLO) expects an images/ + labels/ directory pair.");

        var (imageDir, labelDir) = ResolveImageAndLabelDirs(filePath);
        if (imageDir == null || labelDir == null)
            throw new InvalidOperationException(
                $"No YOLO layout found under '{filePath}'. Expected an 'images' and a 'labels' " +
                "subdirectory (or images and .txt label files in the same directory).");

        var classNames = LoadClassNames(filePath);
        var samples = BuildSamples(imageDir, labelDir, classNames);

        var dataView = _mlContext.Data.LoadFromEnumerable(samples);

        var label = string.IsNullOrWhiteSpace(labelColumn) ? DefaultLabelColumn : labelColumn;
        if (!string.Equals(label, DefaultLabelColumn, StringComparison.Ordinal))
        {
            dataView = _mlContext.Transforms
                .CopyColumns(label, DefaultLabelColumn)
                .Fit(dataView)
                .Transform(dataView);
        }

        return dataView;
    }

    /// <summary>
    /// Resolves the image and label directories. Prefers the conventional <c>images/</c> +
    /// <c>labels/</c> subdirectory pair; falls back to a flat directory holding both.
    /// </summary>
    private static (string? imageDir, string? labelDir) ResolveImageAndLabelDirs(string root)
    {
        var imagesSub = Path.Combine(root, "images");
        var labelsSub = Path.Combine(root, "labels");
        if (Directory.Exists(imagesSub) && Directory.Exists(labelsSub))
            return (imagesSub, labelsSub);

        // Flat layout: images and .txt labels in the same directory.
        var hasImages = Directory.EnumerateFiles(root)
            .Any(f => SupportedImageExtensions.Contains(Path.GetExtension(f)));
        var hasLabels = Directory.EnumerateFiles(root, "*.txt", SearchOption.TopDirectoryOnly).Any();
        if (hasImages && hasLabels)
            return (root, root);

        return (null, null);
    }

    /// <summary>Reads optional class names (id → name); empty when no class file is present.</summary>
    private Dictionary<int, string> LoadClassNames(string root)
    {
        foreach (var name in ConventionalClassFiles)
        {
            var path = Path.Combine(root, name);
            if (!File.Exists(path)) continue;

            var names = File.ReadAllLines(path)
                .Select(l => l.Trim())
                .Where(l => l.Length > 0)
                .ToList();
            var map = new Dictionary<int, string>();
            for (var i = 0; i < names.Count; i++)
                map[i] = names[i];
            if (map.Count > 0)
            {
                _log($"[Info] Loaded {map.Count} class name(s) from {name}.");
                return map;
            }
        }
        return new Dictionary<int, string>();
    }

    private List<CocoSampleRow> BuildSamples(string imageDir, string labelDir, Dictionary<int, string> classNames)
    {
        var images = Directory.EnumerateFiles(imageDir)
            .Where(f => SupportedImageExtensions.Contains(Path.GetExtension(f)))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToList();

        if (images.Count == 0)
            throw new InvalidOperationException($"No supported image files found under '{imageDir}'.");

        var samples = new List<CocoSampleRow>();
        var classCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var missingLabels = 0;
        var unreadableImages = new List<string>();
        var duplicatesCollapsed = 0;

        foreach (var imagePath in images)
        {
            var stem = Path.GetFileNameWithoutExtension(imagePath);
            var labelPath = Path.Combine(labelDir, stem + ".txt");
            if (!File.Exists(labelPath))
            {
                missingLabels++;
                continue;
            }

            int width, height;
            try
            {
                using var img = MLImage.CreateFromFile(imagePath);
                width = img.Width;
                height = img.Height;
            }
            catch (Exception ex)
            {
                unreadableImages.Add($"{Path.GetFileName(imagePath)} ({ex.GetType().Name})");
                continue;
            }

            var labels = new List<string>();
            var boxes = new List<float>();
            var seen = new HashSet<string>();

            foreach (var line in File.ReadLines(labelPath))
            {
                var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 5) continue;
                if (!int.TryParse(parts[0], out var classId)) continue;
                if (!TryParseBox(parts, out var cx, out var cy, out var w, out var h)) continue;

                // YOLO boxes are normalized; convert to absolute x0 y0 x1 y1.
                float x0 = (cx - w / 2f) * width;
                float y0 = (cy - h / 2f) * height;
                float x1 = (cx + w / 2f) * width;
                float y1 = (cy + h / 2f) * height;

                // KAMP YOLO exports often repeat identical boxes; collapse exact duplicates.
                var key = $"{classId}|{x0:F2}|{y0:F2}|{x1:F2}|{y1:F2}";
                if (!seen.Add(key))
                {
                    duplicatesCollapsed++;
                    continue;
                }

                var name = classNames.TryGetValue(classId, out var resolved) ? resolved : $"class_{classId}";
                labels.Add(name);
                boxes.Add(x0); boxes.Add(y0); boxes.Add(x1); boxes.Add(y1);
                classCounts[name] = classCounts.GetValueOrDefault(name) + 1;
            }

            if (labels.Count == 0)
                continue;

            samples.Add(new CocoSampleRow
            {
                ImagePath = Path.GetFullPath(imagePath),
                Label = labels.ToArray(),
                BoundingBoxes = boxes.ToArray()
            });
        }

        ReportValidation(labelDir, samples.Count, classCounts, missingLabels, unreadableImages, duplicatesCollapsed);
        return samples;
    }

    private static bool TryParseBox(string[] parts, out float cx, out float cy, out float w, out float h)
    {
        cx = cy = w = h = 0;
        return float.TryParse(parts[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out cx)
            && float.TryParse(parts[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out cy)
            && float.TryParse(parts[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out w)
            && float.TryParse(parts[4], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out h);
    }

    private void ReportValidation(
        string labelDir,
        int imageCount,
        Dictionary<string, int> classCounts,
        int missingLabels,
        List<string> unreadableImages,
        int duplicatesCollapsed)
    {
        if (missingLabels > 0)
            _log($"[Warning] {missingLabels} image(s) had no matching label .txt and were skipped.");

        if (unreadableImages.Count > 0)
            _log($"[Warning] {unreadableImages.Count} image(s) could not be read and were skipped: " +
                 $"{string.Join(", ", unreadableImages.Take(5))}.");

        if (duplicatesCollapsed > 0)
            _log($"[Info] Collapsed {duplicatesCollapsed} duplicate bounding box annotation(s).");

        if (imageCount == 0)
            throw new InvalidOperationException(
                $"No usable annotated images found under '{labelDir}'. Every image was missing a label, " +
                "unreadable, or had no valid boxes.");

        var totalObjects = classCounts.Values.Sum();
        _log($"[Info] Loaded {totalObjects} object(s) across {imageCount} image(s) and {classCounts.Count} class(es): " +
             $"{string.Join(", ", classCounts.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}.");
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

    /// <summary>Row shape produced by the loader (identical to the COCO loader's output schema).</summary>
    private sealed class CocoSampleRow
    {
        public string ImagePath { get; set; } = string.Empty;
        public string[] Label { get; set; } = Array.Empty<string>();
        public float[] BoundingBoxes { get; set; } = Array.Empty<float>();
    }
}
