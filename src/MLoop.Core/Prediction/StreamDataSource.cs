using Microsoft.ML.Data;

namespace MLoop.Core.Prediction;

public sealed class StreamDataSource : IMultiStreamSource
{
    private readonly byte[] _data;

    public StreamDataSource(MemoryStream stream)
    {
        _data = stream.ToArray();
    }

    public int Count => 1;
    public string? GetPathOrNull(int index) => null;

    public Stream Open(int index)
    {
        if (index != 0)
            throw new ArgumentOutOfRangeException(nameof(index));
        return new MemoryStream(_data, writable: false);
    }

    public TextReader OpenTextReader(int index)
    {
        return new StreamReader(Open(index));
    }
}
