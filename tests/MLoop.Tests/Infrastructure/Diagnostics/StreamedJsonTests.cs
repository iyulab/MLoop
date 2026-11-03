using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Infrastructure.Diagnostics;

/// <summary>
/// A document too large for one string is written as it is serialized; what comes out is the document a string
/// would have held, byte for byte - including a character that is two or four bytes and falls across the
/// serializer's buffer boundary.
/// </summary>
public class StreamedJsonTests
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    [Fact]
    public void TheDocumentIsTheOneASerializeWouldHaveMade_WhateverTheTextIs()
    {
        // Hangul (3 bytes), an emoji (4 bytes, a surrogate pair), a quote and a newline (escaped), and plain text,
        // in enough rows that the serializer flushes its buffer many times at positions of every alignment.
        var rows = Enumerable.Range(0, 60_000)
            .Select(i => new { label = "한글" + new string('가', i % 7) + "😀\"\n" + i, score = i / 7.0, note = i % 5 == 0 ? null : "x" })
            .ToList();
        var payload = new { model = "default", count = rows.Count, predictions = rows };

        var writer = new StringWriter();
        StreamedJson.WriteLine(writer, payload, Options);

        Assert.Equal(JsonSerializer.Serialize(payload, Options) + Environment.NewLine, writer.ToString(), StringComparer.Ordinal);
        Assert.True(writer.ToString().Length > 3_000_000, "the document has to be larger than the serializer's buffer many times over");
    }

    [Fact]
    public void ADocumentWithNothingInIt_IsStillOneLine()
    {
        var writer = new StringWriter();

        StreamedJson.WriteLine(writer, new { }, Options);

        Assert.Equal("{}" + Environment.NewLine, writer.ToString());
    }
}
