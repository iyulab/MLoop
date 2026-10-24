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
    /// passed over by <see cref="TopEntries{T, TKey}.HoldsOnlyAtLeast"/>.
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
/// The <c>limit</c> entries with the largest key of everything added — held in a heap of that size, so reading a
/// large store for its few newest (or least confident) entries costs memory for the few. Among entries of one key,
/// the earlier-added one is kept and listed first, as a stable sort of the whole would.
/// </summary>
internal sealed class TopEntries<T, TKey>(int limit, Func<T, TKey> keyOf)
    where TKey : IComparable<TKey>
{
    // Min-heap: the root is the smallest entry kept, the one a larger arrival displaces. The negated sequence
    // number ranks an earlier arrival above a later one of the same key.
    private readonly PriorityQueue<T, Rank> _heap = new();
    private long _added;

    public bool IsFull => limit <= 0 || _heap.Count >= limit;

    public void Add(T entry)
    {
        if (limit <= 0)
            return;

        var rank = new Rank(keyOf(entry), -_added++);
        if (_heap.Count < limit)
            _heap.Enqueue(entry, rank);
        else if (_heap.TryPeek(out _, out var smallest) && rank.CompareTo(smallest) > 0)
            _heap.DequeueEnqueue(entry, rank);
    }

    /// <summary>
    /// Whether no more entries are wanted and every one kept has a key of at least <paramref name="key"/> — then
    /// nothing below it can enter, and a source whose keys all fall below it need not be read.
    /// </summary>
    public bool HoldsOnlyAtLeast(TKey key) =>
        IsFull && (limit <= 0 || (_heap.TryPeek(out _, out var smallest) && smallest.Key.CompareTo(key) >= 0));

    /// <summary>The entries kept, largest key first.</summary>
    public List<T> LargestFirst()
    {
        var items = _heap.UnorderedItems.ToList();
        items.Sort((a, b) => b.Priority.CompareTo(a.Priority));
        return [.. items.Select(i => i.Element)];
    }

    private readonly record struct Rank(TKey Key, long Order) : IComparable<Rank>
    {
        public int CompareTo(Rank other)
        {
            var byKey = Key.CompareTo(other.Key);
            return byKey != 0 ? byKey : Order.CompareTo(other.Order);
        }
    }
}

/// <summary>
/// A uniform sample without replacement of <c>size</c> entries from a stream of unknown length (reservoir sampling,
/// Algorithm R) — what ordering the whole by a random key and taking the first <c>size</c> gave, without holding the
/// whole.
/// </summary>
internal sealed class Reservoir<T>(int size, Random random)
{
    private readonly List<T> _kept = [];
    private long _seen;

    public long Seen => _seen;

    public void Add(T entry)
    {
        _seen++;
        if (size <= 0)
            return;
        if (_kept.Count < size)
            _kept.Add(entry);
        else if (random.NextInt64(_seen) is var slot && slot < size)
            _kept[(int)slot] = entry;
    }

    public List<T> Kept => _kept;
}
