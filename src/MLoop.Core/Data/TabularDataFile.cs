using System.Collections.Concurrent;
using System.Text;
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
/// does not claim) is returned as it is — unless its Hangul is decomposed (see <see cref="CanonicalText"/>), when it is a
/// temporary copy with the text composed; Excel, Parquet, JSON, SVMlight and TSV are written to a temporary
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
    private static readonly ConcurrentDictionary<(string Path, DateTime Written, long Length), string> Composed = new();

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

        if (!Exists(path))
            return path;
        if (!NeedsConversion(path))
            return ComposedCopy(path);

        var key = isFolder
            ? (Path.GetFullPath(path) + "#" + jsonRecordPath, LastWriteTimeOf(path).ToUniversalTime(), (long)DataFileFormats.FilesIn(path).Count)
            : (new FileInfo(path).FullName + "#" + jsonRecordPath, File.GetLastWriteTimeUtc(path), new FileInfo(path).Length);
        if (Converted.TryGetValue(key, out var existing) && File.Exists(existing))
            return existing;

        var csv = ScratchFiles.New("mloop_table_", ".csv");
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

    /// <summary>
    /// <paramref name="path"/> when its text is composed already — the case for nearly every file, and for a file that is not
    /// UTF-8, whose own encoding the readers detect and which is never rewritten here — else a temporary copy with the text
    /// composed (NFC), byte for byte the same apart from that: line endings, a byte-order mark and the quoting stay.
    /// </summary>
    private static string ComposedCopy(string path)
    {
        var info = new FileInfo(path);
        var key = (info.FullName, info.LastWriteTimeUtc, info.Length);
        if (Composed.TryGetValue(key, out var known) && (known == path || File.Exists(known)))
            return known;

        var result = ComposeText(path);
        Composed[key] = result;
        return result;
    }

    private static string ComposeText(string path)
    {
        var strict = new UTF8Encoding(false, throwOnInvalidBytes: true);
        try
        {
            if (WholeLines(path, strict).All(chunk => chunk.IsNormalized(NormalizationForm.FormC)))
                return path;

            var copy = ScratchFiles.New("mloop_composed_", ".csv");
            using var writer = new StreamWriter(copy, false, new UTF8Encoding(HasUtf8Bom(path)));
            foreach (var chunk in WholeLines(path, strict))
                writer.Write(CanonicalText.Nfc(chunk));
            return copy;
        }
        catch (DecoderFallbackException)
        {
            return path; // not UTF-8: the readers' own encoding detection decides
        }
    }

    /// <summary>
    /// The file's text in pieces that end at a line break (the last at the end of the file): a character never combines
    /// across a line break, so each piece composes on its own, and no more than a buffer of it is held at once.
    /// </summary>
    private static IEnumerable<string> WholeLines(string path, UTF8Encoding strict)
    {
        using var reader = new StreamReader(path, strict, detectEncodingFromByteOrderMarks: true);
        var buffer = new char[1 << 16];
        var pending = new StringBuilder();
        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            pending.Append(buffer, 0, read);
            var text = pending.ToString();
            var cut = text.LastIndexOf('\n') + 1;
            if (cut == 0)
                continue;
            pending.Clear().Append(text, cut, text.Length - cut);
            yield return text[..cut];
        }
        if (pending.Length > 0)
            yield return pending.ToString();
    }

    private static bool HasUtf8Bom(string path)
    {
        using var stream = File.OpenRead(path);
        Span<byte> head = stackalloc byte[3];
        return stream.Read(head) == 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF;
    }

    private static bool IsJson(string path, bool isFolder) =>
        isFolder
            ? DataFileFormats.FilesIn(path).All(f => DataFileFormats.FromPath(f) == DataFileFormat.Json)
            : DataFileFormats.IsReadable(path) && DataFileFormats.FromPath(path) == DataFileFormat.Json;
}
