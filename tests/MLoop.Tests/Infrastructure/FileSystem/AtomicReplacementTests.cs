using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// Overwriting a file through <see cref="FileSystemManager"/> replaces it as a whole: a process reading it meanwhile —
/// <c>mloop serve</c> answering while a model is promoted — finds the old content or the new, never a missing file or a
/// half-written one.
/// </summary>
public sealed class AtomicReplacementTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-atomic-" + Guid.NewGuid().ToString("N"));

    public AtomicReplacementTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private const int Size = 2 * 1024 * 1024;

    private static byte[] Payload(byte value) => Enumerable.Repeat(value, Size).ToArray();

    /// <summary>Overwrites <paramref name="target"/> many times with <paramref name="overwrite"/> while a reader loops over it; what the reader saw that no reader should.</summary>
    private static async Task<(int Missing, int Torn, int Reads)> ReadWhileOverwriting(string target, Func<byte, Task> overwrite)
    {
        await overwrite(1);
        int missing = 0, torn = 0, reads = 0;
        using var stop = new CancellationTokenSource();
        var reader = Task.Run(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                try
                {
                    var bytes = File.ReadAllBytes(target);
                    reads++;
                    if (bytes.Length != Size || bytes.Any(b => b != bytes[0]))
                        torn++;
                }
                catch (FileNotFoundException) { missing++; }
                catch (DirectoryNotFoundException) { missing++; }
                catch (IOException) { /* Windows: the instant of a replacement refuses a new reader — not an absence, and not a torn read */ }
                Thread.Yield();
            }
        });
        for (var i = 0; i < 60; i++)
            await overwrite((byte)(2 + i % 2));
        await stop.CancelAsync();
        await reader;
        return (missing, torn, reads);
    }

    [Fact]
    public async Task OverwritingACopy_NeverLeavesTheFileMissingOrHalfWritten()
    {
        var fileSystem = new FileSystemManager();
        var target = Path.Combine(_directory, "production", "model.zip");
        var source = Path.Combine(_directory, "source.bin");

        var (missing, torn, reads) = await ReadWhileOverwriting(target, async value =>
        {
            await File.WriteAllBytesAsync(source, Payload(value));
            await fileSystem.CopyFileAsync(source, target, overwrite: true);
        });

        Assert.True(reads > 0, "the reader never read — the test saw nothing");
        Assert.Equal(0, missing);
        Assert.Equal(0, torn);
    }

    [Fact]
    public async Task OverwritingText_NeverLeavesTheFileMissingOrHalfWritten()
    {
        var fileSystem = new FileSystemManager();
        var target = Path.Combine(_directory, "production", "metadata.txt");

        var (missing, torn, reads) = await ReadWhileOverwriting(target,
            value => fileSystem.WriteTextAsync(target, new string((char)('a' + value), Size)));

        Assert.True(reads > 0, "the reader never read — the test saw nothing");
        Assert.Equal(0, missing);
        Assert.Equal(0, torn);
    }

    [Fact]
    public async Task AReplacement_LeavesNoTemporaryFileBehind()
    {
        var fileSystem = new FileSystemManager();
        var target = Path.Combine(_directory, "a", "x.json");

        await fileSystem.WriteJsonAsync(target, new { A = 1 });
        await fileSystem.WriteJsonAsync(target, new { A = 2 });

        Assert.Equal(["x.json"], Directory.GetFileSystemEntries(Path.GetDirectoryName(target)!).Select(Path.GetFileName));
        Assert.Contains("2", await File.ReadAllTextAsync(target));
    }

    [Fact]
    public async Task ACopyWithoutOverwrite_StillRefusesAnExistingFile()
    {
        var fileSystem = new FileSystemManager();
        var source = Path.Combine(_directory, "s.txt");
        var target = Path.Combine(_directory, "t.txt");
        await File.WriteAllTextAsync(source, "s");
        await File.WriteAllTextAsync(target, "t");

        await Assert.ThrowsAsync<IOException>(() => fileSystem.CopyFileAsync(source, target, overwrite: false));
        Assert.Equal("t", await File.ReadAllTextAsync(target));
    }
}
