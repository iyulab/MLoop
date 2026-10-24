using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Diagnostics;

/// <summary>
/// A streamed <c>--json</c> document is the document <c>JsonSerializer.Serialize</c> would have built as a string,
/// and a <see cref="JsonOutputScope"/> sees it as the command's own (adds no document of its own on close).
/// </summary>
[Collection("FileSystem")]
public class JsonStdoutTests
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    [Fact]
    public void Write_UnderAJsonScope_IsTheSerializedDocument_AndTheScopeAddsNothing()
    {
        // Large enough to cross the serializer's flush threshold several times, with multi-byte characters
        // that a chunk boundary can split.
        var payload = new { Name = "측정 — ünïcode", Points = Enumerable.Range(0, 50_000).Select(i => new { Index = i, Label = $"점{i}" }) };
        var expected = JsonSerializer.Serialize(payload, Options) + Environment.NewLine;

        var original = Console.Out;
        var buffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            using (new JsonOutputScope())
                JsonStdout.Write(payload, Options);
        }
        finally
        {
            Console.SetOut(original);
        }

        Assert.Equal(expected, buffer.ToString());
    }
}
