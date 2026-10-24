using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Diagnostics;
using MLoop.Core.Training;

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
public sealed class ImageDirectoryLoader : DataProviderBase
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

    public ImageDirectoryLoader(MLContext mlContext, Action<string>? log = null)
        : base(mlContext, log)
    {
    }

    /// <summary>
    /// Scans <paramref name="filePath"/> (a directory) for class subfolders and
    /// builds an <see cref="IDataView"/> of <c>(ImagePath, Label)</c> rows.
    /// </summary>
    public override IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null,
        DataKind? labelKind = null)
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
    /// Sets aside about <paramref name="fraction"/> of each class of a partition this loader produced (stratified,
    /// <see cref="ClassBalance.Seed"/>) — the slice a trainer validates on while it trains. Taken before balancing,
    /// so no copy of an image is on both sides. A class too small to spare an image (fewer than two, or a share
    /// that rounds to none) gives none; <c>Validation</c> is null when no class gave any.
    /// </summary>
    public static (IDataView Train, IDataView? Validation) HoldOut(
        MLContext mlContext, IDataView data, string? labelColumn, double fraction)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var label = LabelOf(labelColumn);
        var (paths, labels, byClass) = Read(data, label);

        var random = new Random(ClassBalance.Seed);
        var heldOut = new HashSet<int>();
        foreach (var members in byClass.Values)
        {
            var take = members.Count < 2 ? 0 : (int)Math.Floor(members.Count * fraction);
            foreach (var i in members.OrderBy(_ => random.Next()).Take(take))
                heldOut.Add(i);
        }
        if (heldOut.Count == 0)
            return (data, null);

        var train = Enumerable.Range(0, paths.Count).Where(i => !heldOut.Contains(i))
            .Select(i => new ImageSample { ImagePath = paths[i], Label = labels[i] });
        var validation = heldOut.Order().Select(i => new ImageSample { ImagePath = paths[i], Label = labels[i] });
        return (View(mlContext, train, label), View(mlContext, validation, label));
    }

    /// <summary>
    /// Oversamples the classes of a training partition this loader produced — <c>--balance</c> for image
    /// classification, by <see cref="ClassBalance"/>. Called on the training partition after the split (and after
    /// <see cref="HoldOut"/>), so a copied image never lands among the held-out images. Returns the partition
    /// unchanged when nothing applied.
    /// </summary>
    public static (IDataView Data, ClassBalance.Outcome<string> Outcome) Balance(
        MLContext mlContext, IDataView trainSet, string? labelColumn, string balance)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var label = LabelOf(labelColumn);
        var (paths, labels, byClass) = Read(trainSet, label);

        var outcome = ClassBalance.Oversample(
            byClass.Select(c => KeyValuePair.Create(c.Key, (IReadOnlyList<string>)c.Value.Select(i => paths[i]).ToList())).ToList(),
            balance);
        if (!outcome.Applied)
            return (trainSet, outcome);

        var labelOf = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < paths.Count; i++)
            labelOf[paths[i]] = labels[i];
        var samples = paths.Select((path, i) => new ImageSample { ImagePath = path, Label = labels[i] })
            .Concat(outcome.Copies.Select(path => new ImageSample { ImagePath = path, Label = labelOf[path] }));
        return (View(mlContext, samples, label), outcome);
    }

    private static string LabelOf(string? labelColumn) =>
        string.IsNullOrWhiteSpace(labelColumn) ? DefaultLabelColumn : labelColumn;

    /// <summary>The partition's paths and labels, and each class's row indices in reading order.</summary>
    private static (List<string> Paths, List<string> Labels, Dictionary<string, List<int>> ByClass) Read(IDataView data, string label)
    {
        ArgumentNullException.ThrowIfNull(data);
        var paths = data.GetColumn<string>(ImagePathColumn).ToList();
        var labels = data.GetColumn<string>(label).ToList();
        var byClass = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (var i = 0; i < paths.Count; i++)
        {
            if (!byClass.TryGetValue(labels[i], out var members))
                byClass[labels[i]] = members = [];
            members.Add(i);
        }
        return (paths, labels, byClass);
    }

    /// <summary>The loader's own schema over <paramref name="samples"/>, a custom label column included.</summary>
    private static IDataView View(MLContext mlContext, IEnumerable<ImageSample> samples, string label)
    {
        IDataView view = mlContext.Data.LoadFromEnumerable(samples.ToList());
        return string.Equals(label, DefaultLabelColumn, StringComparison.Ordinal)
            ? view
            : mlContext.Transforms.CopyColumns(label, DefaultLabelColumn).Fit(view).Transform(view);
    }

    /// <summary>
    /// Counts the class labels in an image-classification dataset directory: the number of
    /// immediate subfolders that contain at least one supported image. Mirrors the loader's
    /// own scan semantics (so the quality gate's 1/N threshold matches what training sees) and
    /// is the single source of truth for "what counts as a class". Returns 0 for a missing
    /// directory or one with no qualifying class folders.
    /// </summary>
    public static int CountClasses(string directory) => CountImagesPerClass(directory).Count;

    /// <summary>
    /// Images per class label in an image-classification dataset directory, ordered by label
    /// (ordinal) — the same scan the loader trains on, for callers that describe a dataset
    /// without loading it. Folders without a supported image are not classes and are left out.
    /// Empty for a missing directory or one with no qualifying class folders.
    /// </summary>
    public static IReadOnlyDictionary<string, int> CountImagesPerClass(string directory)
    {
        var counts = new SortedDictionary<string, int>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return counts;

        foreach (var classDir in Directory.EnumerateDirectories(directory))
        {
            var images = Directory.EnumerateFiles(classDir).Count(IsSupportedImage);
            if (images > 0)
                counts[Path.GetFileName(classDir)] = images;
        }
        return counts;
    }

    /// <summary>Whether <paramref name="path"/> has an extension the loader reads as an image.</summary>
    public static bool IsSupportedImage(string path) => SupportedExtensions.Contains(Path.GetExtension(path));

    /// <summary>The image extensions the loader reads, for messages that name them.</summary>
    public static string SupportedExtensionList => string.Join(", ", SupportedExtensions.OrderBy(e => e, StringComparer.Ordinal));

    /// <summary>
    /// Every supported image under <paramref name="directory"/>, at any depth, ordered by full path
    /// (ordinal) — the images a folder of unlabelled pictures holds, and equally a labelled
    /// class-folder layout's. Empty for a missing directory.
    /// </summary>
    public static IReadOnlyList<string> ListImages(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return [];

        return Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(IsSupportedImage)
            .Select(Path.GetFullPath)
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>
    /// Writes <paramref name="images"/> as the table a trained image-classification model scores —
    /// one <see cref="ImagePathColumn"/> per row — to a temporary <c>.csv</c> and returns its path.
    /// </summary>
    public static string WriteImageTable(IEnumerable<string> images)
    {
        var path = MLoop.Core.Data.ScratchFiles.New("mloop-images-", ".csv");
        using var writer = new StreamWriter(path, append: false, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        writer.WriteLine(ImagePathColumn);
        foreach (var image in images)
            writer.WriteLine(MLoop.Core.Prediction.CsvFieldParser.FormatLine([image])); // a path may hold a comma
        return path;
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
                .Where(IsSupportedImage)
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
            _log(CoreMessage.Warning($"{emptyClasses.Count} class folder(s) contain no supported images and were skipped: " +
                 $"{string.Join(", ", emptyClasses)}. Supported extensions: {string.Join(", ", SupportedExtensions.OrderBy(e => e))}."));

        if (totalImages == 0)
            throw new InvalidOperationException(
                $"No supported image files found under '{directory}'. " +
                $"Supported extensions: {string.Join(", ", SupportedExtensions.OrderBy(e => e))}.");

        if (classCounts.Count < 2)
            _log(CoreMessage.Warning($"Only {classCounts.Count} class with images was found. " +
                 "Image classification needs at least two classes to learn a meaningful boundary."));

        var sparseClasses = classCounts
            .Where(kv => kv.Value < MinRecommendedImagesPerClass)
            .Select(kv => $"{kv.Key} ({kv.Value})")
            .ToList();
        if (sparseClasses.Count > 0)
            _log(CoreMessage.Warning($"Classes with fewer than {MinRecommendedImagesPerClass} images may train poorly: " +
                 $"{string.Join(", ", sparseClasses)}."));

        if (classCounts.Count >= 2)
        {
            var max = classCounts.Values.Max();
            var min = classCounts.Values.Min();
            if (min > 0 && max >= min * 10)
                _log(CoreMessage.Warning($"Severe class imbalance detected (largest {max} vs smallest {min} images). " +
                     "--balance <ratio> (e.g. --balance 3) oversamples the smaller classes in the training split."));
        }

        _log($"[Info] Loaded {totalImages} images across {classCounts.Count} class(es): " +
             $"{string.Join(", ", classCounts.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}.");
    }

    /// <summary>Row shape produced by the loader and consumed by the trainer pipeline.</summary>
    private sealed class ImageSample
    {
        public string ImagePath { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
    }
}
