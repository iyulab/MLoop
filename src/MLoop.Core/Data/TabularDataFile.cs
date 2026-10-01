using System.Collections.Concurrent;
using FilePrepper.Pipeline;
using FilePrepper.Utils;

namespace MLoop.Core.Data;

/// <summary>
/// Turns a data file in any format FilePrepper reads — or a folder of such files — into the
/// comma-separated text every reader in MLoop expects, decided once, at the point a command settles
/// which data it was given.
/// </summary>
/// <remarks>
/// <para>MLoop reads data in many places, all of them as CSV. Rather than teach each of them other
/// formats, a command passes its data path through <see cref="AsCsvAsync"/> as soon as the path is
/// known, and everything after sees CSV. Delimited text (<c>.csv</c>, and any extension FilePrepper
/// does not claim) is returned unchanged; Excel, Parquet, JSON, SVMlight and TSV are written to a temporary
/// <c>.csv</c> — the extension matters, because readers downstream route on it.</para>
/// <para>A folder is a table when it holds data files FilePrepper reads: their rows, in file-name
/// order (<see cref="DataPipeline.FromDirectoryAsync"/>). That is the shape of exports that write a
/// file per record, which could not otherwise be passed at all — listing hundreds of files on a
/// command line runs into the operating system's length limit. A folder laid out for an image or
/// object-detection task is that task's own input and never reaches here.</para>
/// <para>How nested Parquet data becomes columns (a struct's leaves as dotted names, a list as JSON
/// text) is FilePrepper's decision, not restated here.</para>
/// </remarks>
public static class TabularDataFile
{
    // One process converts one input once, however many of its steps ask.
    private static readonly ConcurrentDictionary<(string Path, DateTime Written, long Length), string> Converted = new();

    /// <summary>Whether <paramref name="path"/> is a folder holding data files to read as one table.</summary>
    public static bool IsTableFolder(string path) =>
        Directory.Exists(path) && DataFileFormats.FilesIn(path).Count > 0;

    /// <summary>Whether <paramref name="path"/> is data a command can read: a file, or a table folder.</summary>
    public static bool Exists(string path) => File.Exists(path) || IsTableFolder(path);

    /// <summary>
    /// The size to report for <paramref name="path"/>: a file's length, or the total of a table
    /// folder's data files — what the user named, not the converted copy.
    /// </summary>
    public static long SizeOf(string path) =>
        Directory.Exists(path)
            ? DataFileFormats.FilesIn(path).Sum(f => new FileInfo(f).Length)
            : new FileInfo(path).Length;

    /// <summary>When <paramref name="path"/> last changed: a file's write time, or a table folder's newest data file's.</summary>
    public static DateTime LastWriteTimeOf(string path) =>
        Directory.Exists(path)
            ? DataFileFormats.FilesIn(path).Select(File.GetLastWriteTime).DefaultIfEmpty(Directory.GetLastWriteTime(path)).Max()
            : File.GetLastWriteTime(path);

    /// <summary>Whether <paramref name="path"/> is in a format that needs converting before MLoop reads it.</summary>
    public static bool NeedsConversion(string path) =>
        IsTableFolder(path)
        || (DataFileFormats.IsReadable(path) && DataFileFormats.FromPath(path) != DataFileFormat.Csv);

    /// <summary>
    /// <paramref name="path"/> itself when it is delimited text, otherwise a temporary CSV holding the
    /// same table. <paramref name="jsonRecordPath"/> names, for a JSON file whose rows sit in a nested
    /// array, the dotted path to that array; it is refused for any other format rather than ignored.
    /// For a folder it applies to each of its JSON files.
    /// </summary>
    public static async Task<string> AsCsvAsync(string path, string? jsonRecordPath = null)
    {
        var isFolder = IsTableFolder(path);
        if (!string.IsNullOrWhiteSpace(jsonRecordPath) && !IsJson(path, isFolder))
            throw new ArgumentException(
                $"--records reads the rows of a JSON file; '{Path.GetFileName(path)}' is not one.", nameof(jsonRecordPath));

        if (!NeedsConversion(path) || !Exists(path))
            return path;

        var key = isFolder
            ? (Path.GetFullPath(path) + "#" + jsonRecordPath, LastWriteTimeOf(path).ToUniversalTime(), (long)DataFileFormats.FilesIn(path).Count)
            : (new FileInfo(path).FullName + "#" + jsonRecordPath, File.GetLastWriteTimeUtc(path), new FileInfo(path).Length);
        if (Converted.TryGetValue(key, out var existing) && File.Exists(existing))
            return existing;

        var csv = Path.Combine(Path.GetTempPath(), $"mloop_table_{Guid.NewGuid():N}.csv");
        DataPipeline pipeline;
        try
        {
            pipeline = isFolder
                ? await DataPipeline.FromDirectoryAsync(path, jsonRecordPath).ConfigureAwait(false)
                : await DataPipeline.FromFileAsync(path, jsonRecordPath).ConfigureAwait(false);
        }
        catch (JsonShapeException ex) when (jsonRecordPath is null && ex.RecordPaths.Count > 0)
        {
            // The reader names the record paths that would read the file; here the way to give one is an
            // option. A document with no rows to point at gets no such advice.
            throw new InvalidDataException($"{ex.Message} Pass the path with --records.", ex);
        }
        await pipeline.ToCsvAsync(csv).ConfigureAwait(false);

        Converted[key] = csv;
        return csv;
    }

    private static bool IsJson(string path, bool isFolder) =>
        isFolder
            ? DataFileFormats.FilesIn(path).All(f => DataFileFormats.FromPath(f) == DataFileFormat.Json)
            : DataFileFormats.IsReadable(path) && DataFileFormats.FromPath(path) == DataFileFormat.Json;
}
