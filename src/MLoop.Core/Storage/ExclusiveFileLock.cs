namespace MLoop.Core.Storage;

/// <summary>
/// An advisory lock between processes: the holder has the lock file open with no sharing, so a second process (or thread)
/// that tries to open it is refused by the operating system and waits. It is for a read-modify-write of one state file —
/// the experiment index — where "each command runs independently and exits" means two <c>mloop train</c> processes can be
/// at the same step of the same file at the same time.
/// </summary>
/// <remarks>
/// <para>The lock file is left in place after it is released. Deleting it on release would let a waiter that had already
/// opened the old file hold a lock on a file nobody else can find while a third process creates a new one — two holders.
/// It lives under <c>.mloop/temp/locks/</c>: the <c>.gitignore</c> that <c>mloop init</c> writes has ignored <c>.mloop/temp/</c> from the start, so a lock file never reaches a commit of a project made by any version.</para>
/// <para>A process that dies releases its lock with it: the operating system closes the handle, so there is no stale lock
/// to clean up.</para>
/// </remarks>
public sealed class ExclusiveFileLock : IDisposable
{
    /// <summary>How long a caller waits for another holder before it gives up and says who it was waiting for.</summary>
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(30);

    private FileStream? _handle;

    private ExclusiveFileLock(FileStream handle) => _handle = handle;

    /// <summary>Where the lock for <paramref name="name"/> is kept in the project: <c>.mloop/temp/locks/&lt;name&gt;.lock</c>.</summary>
    public static string PathFor(string projectRoot, string name) =>
        Path.Combine(projectRoot, ".mloop", "temp", "locks", name + ".lock");

    /// <summary>
    /// Waits until <paramref name="lockPath"/> can be held exclusively, retrying with a short growing pause.
    /// </summary>
    /// <exception cref="IOException">Nobody let go within <paramref name="timeout"/>; the message names the file.</exception>
    public static async Task<ExclusiveFileLock> AcquireAsync(
        string lockPath, TimeSpan? timeout = null, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(lockPath))!);
        var give = DateTime.UtcNow + (timeout ?? DefaultTimeout);
        var pause = 5;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var handle = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, bufferSize: 1);
                return new ExclusiveFileLock(handle);
            }
            catch (IOException) when (DateTime.UtcNow < give)
            {
                // Held by another process or thread (a sharing violation, or flock refused): wait and ask again.
            }
            catch (IOException ex)
            {
                throw new IOException(
                    $"Another mloop process has held '{lockPath}' for {(timeout ?? DefaultTimeout).TotalSeconds:0} seconds. " +
                    "Wait for it to finish, or stop it if it is stuck.", ex);
            }
            await Task.Delay(pause, cancellationToken).ConfigureAwait(false);
            pause = Math.Min(pause * 2, 200);
        }
    }

    public void Dispose()
    {
        _handle?.Dispose();
        _handle = null;
    }
}
