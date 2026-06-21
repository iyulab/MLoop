using Microsoft.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// Loads an image-classification dataset from a directory laid out by the
/// MLoop convention <c>folder name = class label</c>:
/// <code>
/// datasets/images/
/// ├── OK/   img001.jpg ...
/// └── NG/   img003.jpg ...
/// </code>
/// Produces an <see cref="IDataView"/> with an <c>ImagePath</c> column (absolute
/// file path, string) and a label column (folder name, string). The trainer
/// pipeline turns <c>ImagePath</c> into raw image bytes via
/// <c>LoadRawImageBytes</c> before fitting the ML.NET ImageClassification trainer.
/// </summary>
public sealed class ImageDirectoryLoader : IDataProvider
{
    /// <summary>Name of the column holding the absolute image file path.</summary>
    public const string ImagePathColumn = "ImagePath";

    /// <summary>Default label column name when none is supplied.</summary>
    public const string DefaultLabelColumn = "Label";

    /// <summary>Below this per-class image count, training is unreliable; warn the user.</summary>
    private const int MinRecommendedImagesPerClass = 5;

    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif"
    };

    private readonly MLContext _mlContext;
    private readonly Action<string> _log;

    public ImageDirectoryLoader(MLContext mlContext, Action<string>? log = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _log = log ?? Console.WriteLine;
    }

    /// <summary>
    /// Scans <paramref name="filePath"/> (a directory) for class subfolders and
    /// builds an <see cref="IDataView"/> of <c>(ImagePath, Label)</c> rows.
    /// </summary>
    public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("Image directory path must be provided.", nameof(filePath));

        if (!Directory.Exists(filePath))
            throw new DirectoryNotFoundException(
                $"Image directory not found: {filePath}\n" +
                "Image classification expects a directory whose subfolders are class labels " +
                "(e.g. datasets/images/OK, datasets/images/NG).");

        var samples = ScanDirectory(filePath);

        var dataView = _mlContext.Data.LoadFromEnumerable(samples);

        // Honor a custom label column name while keeping the loader's internal
        // schema stable. LoadFromEnumerable always names columns after the
        // ImageSample properties ("ImagePath", "Label").
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
    /// Enumerates class subfolders and their images, validating the layout and
    /// emitting warnings for degenerate cases (single class, sparse classes,
    /// class imbalance).
    /// </summary>
    private List<ImageSample> ScanDirectory(string directory)
    {
        var classDirs = Directory.GetDirectories(directory)
            .OrderBy(d => d, StringComparer.Ordinal)
            .ToList();

        if (classDirs.Count == 0)
            throw new InvalidOperationException(
                $"No class subfolders found in '{directory}'. " +
                "Image classification requires one subfolder per class label " +
                "(folder name = label), each containing image files.");

        var samples = new List<ImageSample>();
        var classCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var emptyClasses = new List<string>();

        foreach (var classDir in classDirs)
        {
            var label = Path.GetFileName(classDir);
            var images = Directory.EnumerateFiles(classDir)
                .Where(f => SupportedExtensions.Contains(Path.GetExtension(f)))
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToList();

            if (images.Count == 0)
            {
                emptyClasses.Add(label);
                continue;
            }

            foreach (var image in images)
            {
                samples.Add(new ImageSample
                {
                    ImagePath = Path.GetFullPath(image),
                    Label = label
                });
            }

            classCounts[label] = images.Count;
        }

        ReportValidation(directory, classCounts, emptyClasses, samples.Count);

        return samples;
    }

    private void ReportValidation(
        string directory,
        Dictionary<string, int> classCounts,
        List<string> emptyClasses,
        int totalImages)
    {
        if (emptyClasses.Count > 0)
            _log($"[Warning] {emptyClasses.Count} class folder(s) contain no supported images and were skipped: " +
                 $"{string.Join(", ", emptyClasses)}. Supported extensions: {string.Join(", ", SupportedExtensions.OrderBy(e => e))}.");

        if (totalImages == 0)
            throw new InvalidOperationException(
                $"No supported image files found under '{directory}'. " +
                $"Supported extensions: {string.Join(", ", SupportedExtensions.OrderBy(e => e))}.");

        if (classCounts.Count < 2)
            _log($"[Warning] Only {classCounts.Count} class with images was found. " +
                 "Image classification needs at least two classes to learn a meaningful boundary.");

        var sparseClasses = classCounts
            .Where(kv => kv.Value < MinRecommendedImagesPerClass)
            .Select(kv => $"{kv.Key} ({kv.Value})")
            .ToList();
        if (sparseClasses.Count > 0)
            _log($"[Warning] Classes with fewer than {MinRecommendedImagesPerClass} images may train poorly: " +
                 $"{string.Join(", ", sparseClasses)}.");

        if (classCounts.Count >= 2)
        {
            var max = classCounts.Values.Max();
            var min = classCounts.Values.Min();
            if (min > 0 && max >= min * 10)
                _log($"[Warning] Severe class imbalance detected (largest {max} vs smallest {min} images). " +
                     "Consider balancing the dataset for better accuracy.");
        }

        _log($"[Info] Loaded {totalImages} images across {classCounts.Count} class(es): " +
             $"{string.Join(", ", classCounts.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}.");
    }

    public bool ValidateLabelColumn(IDataView data, string labelColumn)
    {
        return data.Schema.Any(c => c.Name == labelColumn);
    }

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

        return new DataSchema
        {
            Columns = columns,
            RowCount = (int)(data.GetRowCount() ?? 0)
        };
    }

    public (IDataView trainSet, IDataView testSet) SplitData(IDataView data, double testFraction = 0.2)
    {
        if (testFraction <= 0)
            return (data, data);

        if (testFraction >= 1)
            throw new ArgumentException(
                "testFraction must be between 0 and 1 (exclusive).", nameof(testFraction));

        var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42);
        return (split.TrainSet, split.TestSet);
    }

    /// <summary>Row shape produced by the loader and consumed by the trainer pipeline.</summary>
    private sealed class ImageSample
    {
        public string ImagePath { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
    }
}
