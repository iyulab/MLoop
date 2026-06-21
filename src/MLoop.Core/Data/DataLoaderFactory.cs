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

        return taskType?.ToLowerInvariant() switch
        {
            "image-classification" => new ImageDirectoryLoader(mlContext, log),
            "object-detection" => new CocoDataLoader(mlContext, log),
            _ => new CsvDataLoader(mlContext, log)
        };
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
