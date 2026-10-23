using System.Collections.Concurrent;

namespace MLoop.Core.Data;

/// <summary>
/// The copies of a data file MLoop writes to the temp directory while preparing it — re-encoded, flattened, columns
/// removed, converted from another format. Each is made here and deleted here: when the step after it has written its
/// own copy (<see cref="Release"/>), or when the process ends.
/// </summary>
/// <remarks>
/// The last copy cannot go sooner: ML.NET's <c>TextLoader</c> reads its file lazily, every time the view is
/// cursored, so the file must outlive the <c>IDataView</c> — and a command runs in a process of its own, which ends
/// when the command does. Before this every step wrote a full copy of the data set and nothing deleted it: hundreds of
/// megabytes of <c>mloop_*</c> files accumulated in the temp directory of a machine that trained repeatedly.
/// </remarks>
public static class ScratchFiles
{
    private static readonly ConcurrentDictionary<string, byte> Owned = new(StringComparer.Ordinal);

    static ScratchFiles() => AppDomain.CurrentDomain.ProcessExit += (_, _) => ReleaseAll();

    /// <summary>A new path in the temp directory, owned here: <paramref name="prefix"/>, a unique part, <paramref name="extension"/>.</summary>
    public static string New(string prefix, string extension)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{prefix}{Guid.NewGuid():N}{extension}");
        Owned[path] = 0;
        return path;
    }

    /// <summary>Whether <paramref name="path"/> is a copy made here and not yet deleted.</summary>
    public static bool Owns(string path) => Owned.ContainsKey(path);

    /// <summary>
    /// Deletes <paramref name="path"/> when it is a copy made here; anything else — the user's own file — is left
    /// alone. Deleting is best-effort: a file another process still holds stays for the exit sweep.
    /// </summary>
    public static void Release(string path)
    {
        if (!Owned.TryRemove(path, out _))
            return;
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            Owned[path] = 0;
        }
        catch (UnauthorizedAccessException)
        {
            Owned[path] = 0;
        }
    }

    /// <summary>
    /// The next step of a chain of copies: <paramref name="next"/> replaces <paramref name="current"/>, which is
    /// released when the step made a new one — unless it is the chain's input, <paramref name="input"/>. The input is
    /// never the chain's to delete, even when it is a copy made here: a converted table is shared by every load that
    /// reads the same file (<see cref="TabularDataFile"/>).
    /// </summary>
    public static string Advance(string current, string next, string input)
    {
        if (!string.Equals(current, next, StringComparison.Ordinal) && !string.Equals(current, input, StringComparison.Ordinal))
            Release(current);
        return next;
    }

    private static void ReleaseAll()
    {
        foreach (var path in Owned.Keys.ToArray())
            Release(path);
    }
}
