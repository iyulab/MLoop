using System.Text.Json;
using MLoop.CLI.Commands;
using Xunit;

namespace MLoop.Tests.Commands;

public class AnalyzeCommandTests
{
    [Fact]
    public void Serialize_Envelope_UsesCamelCaseAndIncludesFields()
    {
        var env = new AnalyzeEnvelope(
            Aspect: "correlation",
            Available: true,
            Summary: "ok",
            Data: new { highPairs = new[] { new { column1 = "a", column2 = "b", pearson = 0.9 } } },
            Flags: new[] { "flag-1" });

        var json = AnalyzeJson.Serialize(env);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.Equal("correlation", root.GetProperty("aspect").GetString());
        Assert.True(root.GetProperty("available").GetBoolean());
        Assert.Equal("ok", root.GetProperty("summary").GetString());
        Assert.Equal("a", root.GetProperty("data").GetProperty("highPairs")[0].GetProperty("column1").GetString());
        Assert.Equal("flag-1", root.GetProperty("flags")[0].GetString());
    }

    [Fact]
    public void Unavailable_ProducesNotAvailableEnvelope()
    {
        var env = AnalyzeJson.Unavailable("profile");
        Assert.False(env.Available);
        Assert.Null(env.Data);
        Assert.Empty(env.Flags);

        var json = AnalyzeJson.Serialize(env);
        using var doc = JsonDocument.Parse(json);
        Assert.False(doc.RootElement.GetProperty("available").GetBoolean());
    }

    [Fact]
    public void MapProfile_FlagsConstantColumn()
    {
        var stats = new Dictionary<string, (long MissingCount, int UniqueCount)>
        {
            ["id"] = (0, 100),
            ["const_col"] = (0, 1),
            ["sparse"] = (40, 60)
        };
        var env = AnalyzeJson.MapProfile(report: null, stats, rowCount: 100);

        Assert.True(env.Available);
        Assert.Contains(env.Flags, f => f == "constant-column: const_col");
        Assert.Contains(env.Flags, f => f == "high-null: sparse (40.0%)");
    }
}
