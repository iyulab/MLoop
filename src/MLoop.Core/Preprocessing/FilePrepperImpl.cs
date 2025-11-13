using MLoop.Extensibility.Preprocessing;
using FilePrepper.Pipeline;

namespace MLoop.Core.Preprocessing;

/// <summary>
/// FilePrepper library integration for MLoop preprocessing.
/// Wraps FilePrepper Pipeline API for use in preprocessing scripts.
/// </summary>
/// <remarks>
/// This is a minimal wrapper. Preprocessing scripts should use
/// FilePrepper.Pipeline.DataPipeline directly for full functionality.
/// </remarks>
public class FilePrepperImpl : IFilePrepper
{
    public async Task<string> NormalizeEncodingAsync(
        string inputPath,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        // Read and write with UTF-8 encoding (FilePrepper default)
        var pipeline = await DataPipeline.FromCsvAsync(inputPath);
        await pipeline.ToCsvAsync(outputPath);

        return outputPath;
    }

    public async Task<(string Path, int DuplicatesRemoved)> RemoveDuplicatesAsync(
        string inputPath,
        string outputPath,
        string[]? keyColumns = null,
        CancellationToken cancellationToken = default)
    {
        // TODO: FilePrepper has DropDuplicatesTask (Task API) but not in Pipeline API
        // See: FilePrepper_Gaps_Response.md - Gap #1 resolution
        // Current: Using FilterRows workaround for Pipeline API consistency
        // Future: Request DropDuplicates() addition to DataPipeline fluent API
        // Issue: https://github.com/yourusername/FilePrepper/issues/XXX

        var pipeline = await DataPipeline.FromCsvAsync(inputPath);
        var beforeCount = pipeline.RowCount;

        // Use FilterRows to implement basic duplicate removal
        // This is temporary until FilePrepper adds DropDuplicates
        var seen = new HashSet<string>();

        if (keyColumns != null && keyColumns.Length > 0)
        {
            // Key-based deduplication
            pipeline = pipeline.FilterRows(row =>
            {
                var key = string.Join("|", keyColumns.Select(col => row.GetValueOrDefault(col, "")));
                return seen.Add(key);
            });
        }
        else
        {
            // Full-row deduplication
            pipeline = pipeline.FilterRows(row =>
            {
                var key = string.Join("|", row.OrderBy(kvp => kvp.Key).Select(kvp => $"{kvp.Key}={kvp.Value}"));
                return seen.Add(key);
            });
        }

        await pipeline.ToCsvAsync(outputPath);

        // Read back to get final count
        var afterPipeline = await DataPipeline.FromCsvAsync(outputPath);
        var afterCount = afterPipeline.RowCount;

        var duplicatesRemoved = beforeCount - afterCount;

        return (outputPath, duplicatesRemoved);
    }

    public async Task<Dictionary<string, string>> InferAndConvertTypesAsync(
        string inputPath,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        // FilePrepper provides ToDataFrame() for row access
        // See: FilePrepper_Gaps_Response.md - Gap #2 resolution
        var typeMap = new Dictionary<string, string>();
        var pipeline = await DataPipeline.FromCsvAsync(inputPath);

        // Use ToDataFrame() to access rows (FilePrepper built-in API)
        var dataFrame = pipeline.ToDataFrame();
        var firstRow = dataFrame.Rows.FirstOrDefault();

        if (firstRow != null)
        {
            foreach (var kvp in firstRow)
            {
                // Try to infer type from first value
                var value = kvp.Value;

                if (int.TryParse(value, out _))
                    typeMap[kvp.Key] = "int";
                else if (double.TryParse(value, out _))
                    typeMap[kvp.Key] = "double";
                else if (DateTime.TryParse(value, out _))
                    typeMap[kvp.Key] = "datetime";
                else if (bool.TryParse(value, out _))
                    typeMap[kvp.Key] = "bool";
                else
                    typeMap[kvp.Key] = "string";
            }
        }

        // Write output (just copy for now, actual type conversion happens in DataPipeline)
        await pipeline.ToCsvAsync(outputPath);

        return typeMap;
    }
}
