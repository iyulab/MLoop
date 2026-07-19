using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;

namespace MLoop.Core.Data;

/// <summary>
/// Loads an object-detection dataset from a directory holding a COCO-format
/// annotations file alongside the referenced image files:
/// <code>
/// datasets/coco/
/// ├── annotations.json     (or _annotations.coco.json)
/// ├── img001.jpg
/// └── img002.jpg
/// </code>
/// Produces an <see cref="IDataView"/> with three columns:
/// <list type="bullet">
///   <item><c>ImagePath</c> — absolute file path (string), one row per image.</item>
///   <item><c>Label</c> — variable-length vector of class names (string[]), one entry per object.</item>
///   <item><c>BoundingBoxes</c> — variable-length vector of float, four values per object in
///     <c>x0 y0 x1 y1</c> order (absolute pixel coordinates).</item>
/// </list>
/// COCO stores each box as <c>[x, y, width, height]</c>; this loader converts it to the
/// <c>x0 y0 x1 y1</c> order the ML.NET <c>ObjectDetection</c> trainer expects. The trainer
/// pipeline turns <c>ImagePath</c> into an <c>MLImage</c> via <c>LoadImages</c> and maps the
/// <c>Label</c> vector to keys before fitting.
/// </summary>
public sealed class CocoDataLoader : DataProviderBase
{
    /// <summary>Name of the column holding the absolute image file path.</summary>
    public const string ImagePathColumn = "ImagePath";

    /// <summary>Default label column name when none is supplied. Holds a vector of class names.</summary>
    public const string DefaultLabelColumn = "Label";

    /// <summary>Name of the column holding the flattened bounding boxes (x0 y0 x1 y1 per object).</summary>
    public const string BoundingBoxColumn = "BoundingBoxes";

    /// <summary>Annotation file names searched for, in priority order.</summary>
    private static readonly string[] ConventionalAnnotationNames =
    {
        "annotations.json", "_annotations.coco.json", "annotations.coco.json"
    };

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public CocoDataLoader(MLContext mlContext, Action<string>? log = null)
        : base(mlContext, log)
    {
    }

    /// <summary>
    /// Resolves the COCO annotations file under <paramref name="filePath"/> (a directory or a
    /// direct path to the JSON file), parses it, and builds an <see cref="IDataView"/> of
    /// <c>(ImagePath, Label, BoundingBoxes)</c> rows — one row per image that has annotations.
    /// </summary>
    public override IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("Object-detection dataset path must be provided.", nameof(filePath));

        var (annotationFile, imageRoot) = ResolveAnnotationFile(filePath);
        var samples = ParseAndBuildSamples(annotationFile, imageRoot);

        var dataView = _mlContext.Data.LoadFromEnumerable(samples);

        // Honor a custom label column name while keeping the loader's internal schema stable.
        // LoadFromEnumerable always names columns after the CocoSample properties.
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
    /// Resolves the annotations JSON file and the directory image paths are relative to.
    /// Accepts either a directory (searched by convention) or a direct path to the JSON file.
    /// </summary>
    private static (string annotationFile, string imageRoot) ResolveAnnotationFile(string path)
    {
        if (File.Exists(path) && path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            var dir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? Directory.GetCurrentDirectory();
            return (Path.GetFullPath(path), dir);
        }

        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException(
                $"Object-detection dataset not found: {path}\n" +
                "Object detection expects a directory with a COCO annotations file " +
                "(annotations.json) plus the referenced images, or a direct path to the JSON file.");

        foreach (var name in ConventionalAnnotationNames)
        {
            var candidate = Path.Combine(path, name);
            if (File.Exists(candidate))
                return (Path.GetFullPath(candidate), path);
        }

        // Fall back to a single *.json file in the directory; ambiguity is an error.
        var jsonFiles = Directory.GetFiles(path, "*.json", SearchOption.TopDirectoryOnly);
        if (jsonFiles.Length == 1)
            return (Path.GetFullPath(jsonFiles[0]), path);

        if (jsonFiles.Length == 0)
            throw new FileNotFoundException(
                $"No COCO annotations file found in '{path}'. " +
                $"Expected one of: {string.Join(", ", ConventionalAnnotationNames)} (or a single *.json file).");

        throw new InvalidOperationException(
            $"Multiple JSON files found in '{path}'; cannot pick the COCO annotations file automatically. " +
            $"Rename the annotations file to 'annotations.json' or pass it directly.");
    }

    /// <summary>
    /// Parses the COCO JSON and projects it into <see cref="CocoSample"/> rows, validating the
    /// layout and warning about degenerate cases (missing images, sparse classes).
    /// </summary>
    private List<CocoSample> ParseAndBuildSamples(string annotationFile, string imageRoot)
    {
        CocoDocument? doc;
        try
        {
            using var stream = File.OpenRead(annotationFile);
            doc = JsonSerializer.Deserialize<CocoDocument>(stream, JsonOptions);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                $"Failed to parse COCO annotations file '{annotationFile}': {ex.Message}", ex);
        }

        if (doc?.Images == null || doc.Images.Count == 0)
            throw new InvalidOperationException(
                $"COCO annotations file '{annotationFile}' contains no images.");

        if (doc.Annotations == null || doc.Annotations.Count == 0)
            throw new InvalidOperationException(
                $"COCO annotations file '{annotationFile}' contains no annotations. " +
                "Object detection requires bounding-box annotations.");

        var categoryNames = (doc.Categories ?? new List<CocoCategory>())
            .Where(c => c.Id.HasValue)
            .GroupBy(c => c.Id!.Value)
            .ToDictionary(g => g.Key, g => g.First().Name ?? $"class_{g.Key}");

        var imagesById = doc.Images
            .Where(i => i.Id.HasValue && !string.IsNullOrWhiteSpace(i.FileName))
            .GroupBy(i => i.Id!.Value)
            .ToDictionary(g => g.Key, g => g.First());

        var annotationsByImage = doc.Annotations
            .Where(a => a.ImageId.HasValue && a.Bbox is { Count: 4 })
            .GroupBy(a => a.ImageId!.Value);

        var samples = new List<CocoSample>();
        var missingImageFiles = new List<string>();
        var unknownCategories = new HashSet<long>();
        var classCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var skippedImageRefs = 0;

        foreach (var group in annotationsByImage)
        {
            if (!imagesById.TryGetValue(group.Key, out var image))
            {
                skippedImageRefs++;
                continue;
            }

            var imagePath = Path.GetFullPath(Path.Combine(imageRoot, image.FileName!));
            if (!File.Exists(imagePath))
            {
                missingImageFiles.Add(image.FileName!);
                continue;
            }

            var labels = new List<string>();
            var boxes = new List<float>();

            foreach (var ann in group)
            {
                var bbox = ann.Bbox!;
                // COCO bbox is [x, y, width, height]; convert to x0 y0 x1 y1 (absolute pixels).
                float x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3];
                boxes.Add(x);
                boxes.Add(y);
                boxes.Add(x + w);
                boxes.Add(y + h);

                string name;
                if (ann.CategoryId.HasValue && categoryNames.TryGetValue(ann.CategoryId.Value, out var resolved))
                {
                    name = resolved;
                }
                else
                {
                    if (ann.CategoryId.HasValue) unknownCategories.Add(ann.CategoryId.Value);
                    name = ann.CategoryId.HasValue ? $"class_{ann.CategoryId.Value}" : "unknown";
                }

                labels.Add(name);
                classCounts[name] = classCounts.GetValueOrDefault(name) + 1;
            }

            samples.Add(new CocoSample
            {
                ImagePath = imagePath,
                Label = labels.ToArray(),
                BoundingBoxes = boxes.ToArray()
            });
        }

        ReportValidation(annotationFile, samples.Count, classCounts, missingImageFiles, unknownCategories, skippedImageRefs);

        return samples;
    }

    private void ReportValidation(
        string annotationFile,
        int imageCount,
        Dictionary<string, int> classCounts,
        List<string> missingImageFiles,
        HashSet<long> unknownCategories,
        int skippedImageRefs)
    {
        if (missingImageFiles.Count > 0)
        {
            var preview = string.Join(", ", missingImageFiles.Take(5));
            var suffix = missingImageFiles.Count > 5 ? $" (+{missingImageFiles.Count - 5} more)" : "";
            _log($"[Warning] {missingImageFiles.Count} annotated image file(s) were not found on disk and were skipped: {preview}{suffix}.");
        }

        if (skippedImageRefs > 0)
            _log($"[Warning] {skippedImageRefs} annotation group(s) referenced an image id absent from the 'images' list and were skipped.");

        if (unknownCategories.Count > 0)
            _log($"[Warning] {unknownCategories.Count} category id(s) were not declared in 'categories' and were named generically (class_<id>): " +
                 $"{string.Join(", ", unknownCategories.OrderBy(x => x).Take(10))}.");

        if (imageCount == 0)
            throw new InvalidOperationException(
                $"No usable annotated images found for '{annotationFile}'. " +
                "Every annotated image was missing on disk or had no valid bounding boxes.");

        var totalObjects = classCounts.Values.Sum();
        _log($"[Info] Loaded {totalObjects} object(s) across {imageCount} image(s) and {classCounts.Count} class(es): " +
             $"{string.Join(", ", classCounts.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}.");

        if (classCounts.Count < 1)
            _log("[Warning] No object classes were found. Object detection needs at least one labeled class.");
    }

    /// <summary>Row shape produced by the loader and consumed by the trainer pipeline.</summary>
    private sealed class CocoSample
    {
        public string ImagePath { get; set; } = string.Empty;
        public string[] Label { get; set; } = Array.Empty<string>();
        public float[] BoundingBoxes { get; set; } = Array.Empty<float>();
    }

    // --- COCO JSON DTOs (only the fields MLoop consumes) ---

    private sealed class CocoDocument
    {
        [JsonPropertyName("images")] public List<CocoImage>? Images { get; set; }
        [JsonPropertyName("annotations")] public List<CocoAnnotation>? Annotations { get; set; }
        [JsonPropertyName("categories")] public List<CocoCategory>? Categories { get; set; }
    }

    private sealed class CocoImage
    {
        [JsonPropertyName("id")] public long? Id { get; set; }
        [JsonPropertyName("file_name")] public string? FileName { get; set; }
    }

    private sealed class CocoAnnotation
    {
        [JsonPropertyName("image_id")] public long? ImageId { get; set; }
        [JsonPropertyName("category_id")] public long? CategoryId { get; set; }
        [JsonPropertyName("bbox")] public List<float>? Bbox { get; set; }
    }

    private sealed class CocoCategory
    {
        [JsonPropertyName("id")] public long? Id { get; set; }
        [JsonPropertyName("name")] public string? Name { get; set; }
    }
}
