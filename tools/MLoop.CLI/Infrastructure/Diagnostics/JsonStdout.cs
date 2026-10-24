using System.Text;
using System.Text.Json;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// Writes a <c>--json</c> document to stdout as it is serialized, instead of building it as one string first.
/// A large document — <c>detect</c> over two million points is 0.5 GB of JSON — otherwise sat in memory twice
/// (the UTF-16 string and the serializer's buffer) before the first byte left. Goes through <see cref="Console.Out"/>,
/// so a <see cref="JsonOutputScope"/> still sees that stdout received its document.
/// </summary>
internal static class JsonStdout
{
    public static void Write<T>(T value, JsonSerializerOptions options)
    {
        var stdout = Console.Out;
        using (var stream = new TextWriterStream(stdout))
            JsonSerializer.Serialize(stream, value, options);
        stdout.WriteLine();
        stdout.Flush();
    }

    /// <summary>A write-only stream that decodes the serializer's UTF-8 into a <see cref="TextWriter"/>.</summary>
    private sealed class TextWriterStream(TextWriter writer) : Stream
    {
        private readonly Decoder _decoder = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetDecoder();
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
            if (needed > _chars.Length)
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
