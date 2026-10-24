using System.Runtime.CompilerServices;
using MLoop.Core.Storage;

namespace MLoop.DataStore.Services;

/// <summary>
/// Reading the store's JSON-lines day files one line at a time. A day file of a busy model is tens of megabytes
/// (80,000 predictions a day is 20 MB), and reading it whole to keep a few entries — or to find one id — held the
/// file and every parsed entry at once.
/// </summary>
internal static class JsonLines
{
    /// <summary>
    /// The non-blank lines of <paramref name="path"/>, in file order. A file that cannot be opened or read — locked,
    /// removed by a prune while it was being listed — ends the sequence instead of throwing, as the store's readers
    /// have always skipped such a file.
    /// </summary>
    public static async IAsyncEnumerable<string> ReadAsync(
        string path, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        StreamReader reader;
        try
        {
            reader = new StreamReader(path);
        }
        catch (IOException)
        {
            yield break;
        }
        catch (UnauthorizedAccessException)
        {
            yield break;
        }

        using (reader)
        {
            while (true)
            {
                string? line;
                try
                {
                    line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (IOException)
                {
                    line = null;
                }

                if (line is null)
                    yield break;
                if (!string.IsNullOrWhiteSpace(line))
                    yield return line;
            }
        }
    }

    /// <summary>
    /// A model's day files, newest day first. A file whose name is not a day sorts by name among them and is never
    /// passed over by <see cref="NewestEntries{T}.HoldsAllNewerThan"/>.
    /// </summary>
    public static IEnumerable<string> DayFilesNewestFirst(string directory, DateTimeOffset from, DateTimeOffset to) =>
        Directory.GetFiles(directory, "*.jsonl")
            .Where(f => TimestampKeys.DayFileCovers(f, from, to))
            .OrderByDescending(f => Path.GetFileName(f), StringComparer.Ordinal);

    /// <summary>The first instant after the day a day file stands for, or null when its name is not a day.</summary>
    public static DateTimeOffset? EndOfDay(string dayFile) =>
        TimestampKeys.TryParseDay(Path.GetFileNameWithoutExtension(dayFile), out var day)
            ? new DateTimeOffset(day.AddDays(1).ToDateTime(TimeOnly.MinValue), TimeSpan.Zero)
            : null;
}

/// <summary>
/// The newest <c>limit</c> entries of everything added, by time — held in a heap of that size, so reading a large
/// store for its latest few entries costs memory for the few.
/// </summary>
internal sealed class NewestEntries<T>(int limit, Func<T, DateTimeOffset> timeOf)
{
    // Min-heap on time: the root is the oldest entry kept, the one a newer arrival displaces. The sequence number
    // breaks ties so that, among entries of one time, the earlier-read one is kept — as the List + stable sort did.
    private readonly PriorityQueue<T, (DateTimeOffset Time, long Order)> _heap = new();
    private long _added;

    public bool IsFull => limit <= 0 || _heap.Count >= limit;

    public void Add(T entry)
    {
        if (limit <= 0)
            return;

        var priority = (timeOf(entry), -_added++);
        if (_heap.Count < limit)
            _heap.Enqueue(entry, priority);
        else if (_heap.TryPeek(out _, out var oldest) && priority.CompareTo(oldest) > 0)
            _heap.DequeueEnqueue(entry, priority);
    }

    /// <summary>
    /// Whether every entry kept is at or after <paramref name="time"/> and no more are wanted — then nothing older
    /// than <paramref name="time"/> can enter, and a day file that ends there need not be read.
    /// </summary>
    public bool HoldsAllNewerThan(DateTimeOffset time) =>
        IsFull && (limit <= 0 || (_heap.TryPeek(out _, out var oldest) && oldest.Time >= time));

    /// <summary>The entries kept, newest first.</summary>
    public List<T> NewestFirst()
    {
        var items = _heap.UnorderedItems.ToList();
        items.Sort((a, b) => b.Priority.CompareTo(a.Priority));
        return [.. items.Select(i => i.Element)];
    }
}
