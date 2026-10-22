using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// The bytes of a file <see cref="ColumnLayoutInference"/> reads its layout from: the first 4 MB of a
/// file up to 8 MB, and for a larger one its first megabyte plus whole-line chunks from across the rest
/// (about 4.4 MB in all).
/// </summary>
/// <remarks>
/// <para>
/// Types are read from the sample's first 10,000 lines, so the chunks matter only when those lines do
/// not fit in the first megabyte — lines longer than about 100 bytes — and then only the earliest of
/// them, since chunks are kept in file order. With shorter lines a large file is typed from its head
/// alone.
/// </para>
/// <para>
/// The chunk positions are drawn from a fixed seed: one file always gives one sample, and so one
/// layout. (ML.NET AutoML draws them from an unseeded generator, so its layout of a large file with
/// long lines can change from one call to the next.)
/// </para>
/// </remarks>
internal sealed class ColumnLayoutSample : IMultiStreamSource
{
    private const int BufferBytes = 4 << 20;
    private const int FirstChunkBytes = 1 << 20;
    private const int LinesPerChunk = 20;
    private const double OversamplingRate = 1.1;
    private const int ChunkSeed = 1;

    private readonly byte[] _buffer;

    private ColumnLayoutSample(byte[] buffer) => _buffer = buffer;

    public int Count => 1;

    public string? GetPathOrNull(int index) => null;

    public Stream Open(int index) => new MemoryStream(_buffer, writable: false);

    public TextReader OpenTextReader(int index) => new StreamReader(Open(index));

    public static ColumnLayoutSample Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var size = stream.Length;
        if (size <= 2L * BufferBytes)
            return FromHead(stream);

        var first = new byte[FirstChunkBytes];
        stream.ReadExactly(first);
        if (!SamplesSafely(first))
        {
            // From the start of the file — the megabyte just read is not to be skipped.
            stream.Seek(0, SeekOrigin.Begin);
            return FromHead(stream);
        }

        var lines = first.Count(b => b == '\n');
        if (lines == 0)
            throw new ArgumentException("Couldn't identify line breaks. Provided file is not text?");

        var firstNewline = Array.IndexOf(first, (byte)'\n');
        var averageLine = Math.Max(2.0 * (first.Length - firstNewline) / (lines * 2 - 1), 3);
        var usefulChunk = (int)(averageLine * LinesPerChunk);
        var chunkSize = (int)(usefulChunk + averageLine);
        var chunkCount = (int)Math.Ceiling((BufferBytes * OversamplingRate - FirstChunkBytes) / usefulChunk);
        chunkCount = Math.Min(chunkCount, (int)Math.Floor((double)(size - first.Length) / chunkSize));

        var remaining = size - first.Length - (long)chunkSize * chunkCount;
        var random = new Random(ChunkSeed);
        var starts = Enumerable.Range(0, chunkCount)
            .Select(_ => random.NextDouble() * remaining)
            .OrderBy(x => x)
            .Select((spot, i) => (long)(spot + first.Length + (long)i * chunkSize))
            .ToArray();

        var chunks = new List<byte[]> { first };
        foreach (var start in starts)
        {
            stream.Seek(start, SeekOrigin.Begin);
            var chunk = new byte[chunkSize];
            var read = stream.ReadAtLeast(chunk, chunkSize, throwOnEndOfStream: false);
            Array.Resize(ref chunk, read);
            chunks.Add(chunk);
        }
        return new ColumnLayoutSample(Stitch(wholeFile: false, chunks));
    }

    private static ColumnLayoutSample FromHead(Stream stream)
    {
        var buffer = new byte[BufferBytes];
        var read = stream.ReadAtLeast(buffer, buffer.Length, throwOnEndOfStream: false);
        Array.Resize(ref buffer, read);
        return new ColumnLayoutSample(Stitch(wholeFile: read == stream.Length, [buffer]));
    }

    // Each chunk but the first starts after its first line break; each ends at its last one, except
    // the last chunk of a file read whole.
    private static byte[] Stitch(bool wholeFile, List<byte[]> chunks)
    {
        using var result = new MemoryStream();
        for (var i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];
            var from = i == 0 ? 0 : Array.IndexOf(chunk, (byte)'\n') + 1;
            var to = wholeFile && i == chunks.Count - 1 ? chunk.Length : Array.LastIndexOf(chunk, (byte)'\n') + 1;
            if (to == 0)
                continue;
            result.Write(chunk, from, to - from);
        }

        return result.Length > 0
            ? result.ToArray()
            : throw new ArgumentException("File is not text, or couldn't detect line breaks");
    }

    // Chunks can be cut at any byte, which only a byte-oriented encoding survives: UTF-16 and UTF-32
    // (by their marks) and text with NUL bytes are read from the head instead.
    private static bool SamplesSafely(byte[] buffer)
    {
        if (buffer is [0x00, 0x00, 0xFE, 0xFF, ..] or [0xFF, 0xFE, 0x00, 0x00, ..] or [0xFE, 0xFF, ..] or [0xFF, 0xFE, ..])
            return false;
        if (buffer is [0xEF, 0xBB, 0xBF, ..] or [0x2B, 0x2F, 0x76, ..])
            return true;

        var limit = Math.Min(1000, buffer.Length);
        var i = 0;
        var utf8 = false;
        while (i < limit - 4)
        {
            var b = buffer[i];
            if (b <= 0x7F)
            {
                i += 1;
                continue;
            }
            if (b is >= 0xC2 and <= 0xDF && IsContinuation(buffer[i + 1]))
            {
                i += 2;
                utf8 = true;
                continue;
            }
            if (b is >= 0xE0 and <= 0xF0 && IsContinuation(buffer[i + 1]) && IsContinuation(buffer[i + 2]))
            {
                i += 3;
                utf8 = true;
                continue;
            }
            if (b is >= 0xF0 and <= 0xF4 && IsContinuation(buffer[i + 1]) && IsContinuation(buffer[i + 2]) && IsContinuation(buffer[i + 3]))
            {
                i += 4;
                utf8 = true;
                continue;
            }
            utf8 = false;
            break;
        }

        return utf8 || !buffer.Take(limit).Any(b => b == 0);

        static bool IsContinuation(byte b) => b is >= 0x80 and < 0xC0;
    }
}
