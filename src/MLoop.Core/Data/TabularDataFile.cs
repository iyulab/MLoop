using System.Collections.Concurrent;
using FilePrepper.Pipeline;
using FilePrepper.Utils;

namespace MLoop.Core.Data;

/// <summary>
/// Turns a data file in any format FilePrepper reads into the comma-separated text every reader in
/// MLoop expects — decided once, at the point a command settles which file it was given.
/// </summary>
/// <remarks>
/// <para>MLoop reads data in many places, all of them as CSV. Rather than teach each of them other
/// formats, a command passes its data path through <see cref="AsCsvAsync"/> as soon as the path is
/// known, and everything after sees CSV. Delimited text (<c>.csv</c>, and any extension FilePrepper
/// does not claim) is returned unchanged; Excel, Parquet, JSON and TSV are written to a temporary
/// <c>.csv</c> — the extension matters, because readers downstream route on it.</para>
/// <para>How nested Parquet data becomes columns (a struct's leaves as dotted names, a list as JSON
/// text) is FilePrepper's decision, not restated here.</para>
/// </remarks>
public static class TabularDataFile
{
    // One process converts one file once, however many of its steps ask.
    private static readonly ConcurrentDictionary<(string Path, DateTime Written, long Length), string> Converted = new();

    /// <summary>Whether <paramref name="path"/> is in a format that needs converting before MLoop reads it.</summary>
    public static bool NeedsConversion(string path) =>
        DataFileFormats.IsReadable(path) && DataFileFormats.FromPath(path) != DataFileFormat.Csv;

    /// <summary>
    /// <paramref name="path"/> itself when it is delimited text, otherwise a temporary CSV holding the
    /// same table.
    /// </summary>
    public static async Task<string> AsCsvAsync(string path)
    {
        if (!NeedsConversion(path) || !File.Exists(path))
            return path;

        var info = new FileInfo(path);
        var key = (info.FullName, info.LastWriteTimeUtc, info.Length);
        if (Converted.TryGetValue(key, out var existing) && File.Exists(existing))
            return existing;

        var csv = Path.Combine(Path.GetTempPath(), $"mloop_table_{Guid.NewGuid():N}.csv");
        var pipeline = await DataPipeline.FromFileAsync(path).ConfigureAwait(false);
        await pipeline.ToCsvAsync(csv).ConfigureAwait(false);

        Converted[key] = csv;
        return csv;
    }
}
