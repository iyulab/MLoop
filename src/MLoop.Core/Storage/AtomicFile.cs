namespace MLoop.Core.Storage;

/// <summary>
/// Replaces a file as a whole: the content is written to a temporary file beside it and renamed over it, so a process
/// reading the file meanwhile — <c>mloop serve</c> answering while a model is promoted, calibrated or given a threshold —
/// finds the old content or the new, never a missing file or a half-written one.
/// </summary>
/// <remarks>
/// Writing in place (<c>File.WriteAllText</c> truncates, <c>File.Copy(overwrite)</c> removes then writes) left the file
/// absent or short for the length of the write: under load a prediction answered 404 or 500 at that moment (found by the
/// lifecycle check's promotion-under-load step). One authority, because the first fix was in one writer and the others kept
/// the defect: every writer of a file another process reads goes through here.
/// </remarks>
public static class AtomicFile
{
    /// <summary>Writes <paramref name="text"/> to <paramref name="path"/> (UTF-8, no byte-order mark, as <see cref="File.WriteAllText(string, string)"/>).</summary>
    public static void WriteAllText(string path, string text) => Replace(path, temporary => File.WriteAllText(temporary, text));

    /// <summary>Lets <paramref name="fill"/> write the new content to the temporary path it is given, then makes it <paramref name="path"/>.</summary>
    public static void Replace(string path, Action<string> fill)
    {
        ArgumentNullException.ThrowIfNull(fill);
        var temporary = TemporaryBeside(path);
        try
        {
            fill(temporary);
            MoveOver(temporary, path);
        }
        catch
        {
            Discard(temporary);
            throw;
        }
    }

    /// <summary>The asynchronous <see cref="Replace"/>.</summary>
    public static async Task ReplaceAsync(string path, Func<string, Task> fill, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(fill);
        var temporary = TemporaryBeside(path);
        try
        {
            await fill(temporary).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            MoveOver(temporary, path);
        }
        catch
        {
            Discard(temporary);
            throw;
        }
    }

    // On Windows a rename over a file that a reader has open is refused (the reader did not share it for deletion) until the
    // reader closes it — a request reading the model, a moment. Waiting it out keeps the replacement from failing under the very
    // load it exists for. Elsewhere a rename over an open file just works, and a refusal is a real error.
    private static void MoveOver(string temporary, string path)
    {
        for (var attempt = 0; ; attempt++)
        {
            try
            {
                File.Move(temporary, path, overwrite: true);
                return;
            }
            catch (Exception e) when (OperatingSystem.IsWindows() && attempt < WindowsRetries && e is IOException or UnauthorizedAccessException)
            {
                Thread.Sleep(10 + attempt % 5);
            }
        }
    }

    private const int WindowsRetries = 200;

    // A hidden name in the same directory: a rename inside one volume is atomic, and no listing by extension picks it up.
    private static string TemporaryBeside(string path) =>
        Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");

    private static void Discard(string temporary)
    {
        try { File.Delete(temporary); } catch (IOException) { /* the cleanup of a failed write is not its error */ }
    }
}
