using MLoop.Core.Data;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Several data files merged into one training file — what <c>mloop train --data a.csv b.csv</c> and
/// <c>POST /train</c> with <c>additionalDataFiles</c> both train on, by one rule: each file is read as a table
/// (JSON, Parquet and Excel included), the files must share their columns, and a file without the weight
/// column is unweighted — its rows count 1. That is how the original training data merges with a weighted
/// retraining set from feedback.
/// </summary>
public static class TrainingDataFiles
{
    /// <summary>The merge, or why it could not be made.</summary>
    /// <param name="Path">The merged file; <c>null</c> when it was refused.</param>
    /// <param name="Error">What went wrong, naming the files as the caller gave them.</param>
    /// <param name="Guidance">What to do about it.</param>
    /// <param name="RowsPerFile">Rows each file contributed, by the name the caller gave it.</param>
    public sealed record Merge(
        string? Path,
        string? Error,
        string? Guidance,
        IReadOnlyList<(string File, int Rows)> RowsPerFile)
    {
        /// <summary>Rows in the merged file.</summary>
        public int TotalRows => RowsPerFile.Sum(f => f.Rows);
    }

    /// <summary>
    /// Merges <paramref name="files"/> (absolute paths, in order) into <paramref name="outputPath"/>.
    /// </summary>
    /// <param name="weightColumn">The model's weight column, if it trains weighted.</param>
    /// <param name="jsonRecordPath">Where the rows sit in a JSON file (<c>--records</c>).</param>
    public static async Task<Merge> MergeAsync(
        IReadOnlyList<string> files, string? weightColumn, string outputPath, string? jsonRecordPath = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        var merger = new CsvMerger(new CsvHelperImpl());

        // Each file is read as a table before the merge reads it as CSV — a JSON, Parquet or Excel file
        // otherwise had its raw text taken for a header. Reports keep naming the files the caller gave.
        var tables = new List<string>();
        var givenName = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in files)
        {
            var table = await TabularDataFile.AsCsvAsync(file, jsonRecordPath);
            tables.Add(table);
            givenName[System.IO.Path.GetFileName(table)] = System.IO.Path.GetFileName(file);
        }
        string Given(string name) => givenName.GetValueOrDefault(name, name);

        IReadOnlyDictionary<string, string>? columnDefaults = string.IsNullOrEmpty(weightColumn)
            ? null
            : new Dictionary<string, string> { [weightColumn] = "1" };

        var validation = await merger.ValidateSchemaCompatibilityAsync(tables, columnDefaults);
        if (!validation.IsCompatible)
        {
            // validation.Message only counts the mismatch; the columns that actually differ per file
            // are in MismatchedColumns — say them instead of making the caller re-diff the files.
            var perFile = string.Join("; ", validation.MismatchedColumns
                .Select(kv => $"{Given(kv.Key)}: {string.Join(", ", kv.Value)}"));
            return new Merge(null,
                $"Schema mismatch between files: {validation.Message}",
                $"Columns unique to each file — {perFile}. Keep only the {validation.CommonColumns.Count} shared columns across files, or merge files with matching schemas.",
                []);
        }

        var result = await merger.MergeAsync(tables, outputPath, columnDefaults);
        if (!result.Success)
            return new Merge(null, $"Failed to merge files: {result.Error}",
                "Check that each file is a valid, non-empty CSV with a header row.", []);

        return new Merge(result.OutputPath, null, null,
            result.RowsPerFile.Select(kv => (Given(kv.Key), kv.Value)).ToList());
    }
}
