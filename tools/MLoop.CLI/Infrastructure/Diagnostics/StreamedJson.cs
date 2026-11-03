using System.Text;
using System.Text.Json;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// A JSON document written to a text writer as it is serialized, never as one string.
/// </summary>
/// <remarks>
/// <c>JsonSerializer.Serialize</c> to a string fails with "Arithmetic operation resulted in an overflow" once the
/// document passes the largest string .NET can make (about a billion characters) - a file of a few hundred thousand
/// rows answered with a probability for each of a dozen classes reaches it, and the command that was asked for the
/// answers said nothing of why it had none.
/// </remarks>
public static class StreamedJson
{
    /// <summary>Writes <paramref name="value"/> as one line of JSON to <paramref name="writer"/>.</summary>
    public static void WriteLine<T>(TextWriter writer, T value, JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(options);

        using (var stream = new TextWriterStream(writer))
            JsonSerializer.Serialize(stream, value, options);
        writer.WriteLine();
    }

    /// <summary>The bytes the serializer produces, decoded as UTF-8 into a text writer (a character split across two writes is kept whole).</summary>
    private sealed class TextWriterStream(TextWriter writer) : Stream
    {
        private readonly Decoder _decoder = new UTF8Encoding(false).GetDecoder();
        private char[] _chars = new char[4096];

        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            var needed = _decoder.GetCharCount(buffer, flush: false);
            if (_chars.Length < needed)
                _chars = new char[needed];
            var written = _decoder.GetChars(buffer, _chars, flush: false);
            writer.Write(_chars, 0, written);
        }

        public override void Flush() => writer.Flush();
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }
}
