using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace MLoop.Core.ImageAnomaly;

/// <summary>
/// The patch vectors of normal images an image-anomaly model remembers: <see cref="Count"/> unit vectors of
/// <see cref="Dimension"/> numbers, row by row. A new image is as strange as its strangest patch is far from every
/// patch here (<see cref="PatchBankScorer"/>).
/// </summary>
/// <remarks>
/// File: the magic <c>MLPB</c>, a format number (1), the dimension and the count (little-endian int32 each), then
/// <c>count × dimension</c> little-endian float32. The vectors are the encoder's (L2-normalised), so the file is
/// only meaningful beside the identity of the encoder that made them (<see cref="ImageAnomalyModel"/>).
/// </remarks>
public sealed class PatchBank
{
    private const int FormatNumber = 1;
    private const int HeaderBytes = 16;

    private PatchBank(int dimension, int count, float[] data)
    {
        Dimension = dimension;
        Count = count;
        Data = data;
    }

    /// <summary>The length of each vector.</summary>
    public int Dimension { get; }

    /// <summary>How many vectors.</summary>
    public int Count { get; }

    /// <summary>The vectors, row by row: vector <c>i</c> is <c>Data[i * Dimension .. (i + 1) * Dimension]</c>.</summary>
    public float[] Data { get; }

    /// <summary>Vector <paramref name="index"/> as a span over the bank's own storage.</summary>
    public ReadOnlySpan<float> this[int index] => Data.AsSpan(index * Dimension, Dimension);

    /// <summary>A bank of <paramref name="vectors"/>, each of <paramref name="dimension"/> numbers.</summary>
    public static PatchBank Of(IReadOnlyList<float[]> vectors, int dimension)
    {
        ArgumentNullException.ThrowIfNull(vectors);
        if (dimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), "A vector has at least one number.");
        var data = new float[checked(vectors.Count * dimension)];
        for (var i = 0; i < vectors.Count; i++)
        {
            if (vectors[i].Length != dimension)
                throw new ArgumentException($"Vector {i} has {vectors[i].Length} numbers, not {dimension}.", nameof(vectors));
            vectors[i].CopyTo(data, i * dimension);
        }
        return new PatchBank(dimension, vectors.Count, data);
    }

    /// <summary>A bank over vectors already laid out row by row.</summary>
    public static PatchBank FromRows(float[] data, int dimension)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (dimension <= 0 || data.Length % dimension != 0)
            throw new ArgumentException($"{data.Length} numbers are not whole vectors of {dimension}.", nameof(data));
        return new PatchBank(dimension, data.Length / dimension, data);
    }

    /// <summary>Writes the bank to <paramref name="path"/>.</summary>
    public void Write(string path)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 16);
        Span<byte> header = stackalloc byte[HeaderBytes];
        "MLPB"u8.CopyTo(header);
        BinaryPrimitives.WriteInt32LittleEndian(header[4..], FormatNumber);
        BinaryPrimitives.WriteInt32LittleEndian(header[8..], Dimension);
        BinaryPrimitives.WriteInt32LittleEndian(header[12..], Count);
        stream.Write(header);
        // Little-endian on disk; every platform this runs on is little-endian, which the check below enforces.
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("A patch bank is stored little-endian and read as it is.");
        stream.Write(MemoryMarshal.AsBytes(Data.AsSpan()));
    }

    /// <summary>Reads the bank at <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a patch bank, or is cut short.</exception>
    public static PatchBank Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        Span<byte> header = stackalloc byte[HeaderBytes];
        if (stream.Read(header) != HeaderBytes || !header[..4].SequenceEqual("MLPB"u8))
            throw new InvalidDataException($"'{path}' is not a patch bank.");
        var format = BinaryPrimitives.ReadInt32LittleEndian(header[4..]);
        if (format != FormatNumber)
            throw new InvalidDataException($"'{path}' is patch bank format {format}; this MLoop reads format {FormatNumber}.");
        var dimension = BinaryPrimitives.ReadInt32LittleEndian(header[8..]);
        var count = BinaryPrimitives.ReadInt32LittleEndian(header[12..]);
        if (dimension <= 0 || count < 0)
            throw new InvalidDataException($"'{path}' has an impossible shape ({count} × {dimension}).");
        var expected = (long)count * dimension * sizeof(float);
        if (stream.Length - HeaderBytes != expected)
            throw new InvalidDataException($"'{path}' holds {stream.Length - HeaderBytes} bytes of vectors, not the {expected} its header says.");
        var data = new float[checked(count * dimension)];
        stream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
        return new PatchBank(dimension, count, data);
    }
}
