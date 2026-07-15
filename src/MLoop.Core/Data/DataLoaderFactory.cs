using Microsoft.ML;
using MLoop.Core.Contracts;

namespace MLoop.Core.Data;

/// <summary>
/// Selects the appropriate <see cref="IDataProvider"/> for a given task type.
/// Tabular tasks load CSV files; image classification loads a directory laid out
/// by the <c>folder name = label</c> convention; object detection loads a directory
/// holding a COCO-format annotations file plus the referenced images.
/// </summary>
public static class DataLoaderFactory
{
    /// <summary>
    /// Returns the data loader for <paramref name="taskType"/>. Unknown or tabular
    /// tasks fall back to <see cref="CsvDataLoader"/>.
    /// </summary>
    public static IDataProvider Create(string? taskType, MLContext mlContext, Action<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        if (IsDirectoryBased(taskType) && !MLoop.Core.AutoML.DeepLearningRegistry.IsRegistered)
            throw new NotSupportedException(
                $"Task '{taskType}' requires the MLoop.Core.DeepLearning package (data loader not registered).");

        var dl = MLoop.Core.AutoML.DeepLearningRegistry.Current?.CreateDataLoader(
            taskType ?? string.Empty, mlContext, log);
        if (dl is not null) return dl;

        // Invariant: a directory-based task must never fall back to CsvDataLoader.
        // If we reach here for such a task, a DL module IS registered but its
        // CreateDataLoader returned null (e.g. a task added to IsDirectoryBased
        // without a corresponding case in the module's loader switch). Silently
        // returning CsvDataLoader would misinterpret a directory/COCO layout as CSV.
        if (IsDirectoryBased(taskType))
            throw new NotSupportedException(
                $"Task '{taskType}' is directory-based but no data loader was provided by " +
                "the registered MLoop.Core.DeepLearning module (CreateDataLoader returned null).");

        return new CsvDataLoader(mlContext, log);
    }

    /// <summary>
    /// True when <paramref name="taskType"/> consumes a directory (image classification)
    /// or a COCO annotations file (object detection) rather than a CSV file. Callers use
    /// this to bypass CSV-specific preprocessing.
    /// </summary>
    public static bool IsDirectoryBased(string? taskType) =>
        string.Equals(taskType, "image-classification", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(taskType, "object-detection", StringComparison.OrdinalIgnoreCase);
}
