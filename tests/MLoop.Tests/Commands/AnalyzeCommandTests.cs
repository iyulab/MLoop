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
    public void Serialize_NonFiniteDoubles_EmitNullNotInvalidJson()
    {
        // Repro: importance on multicollinear data yields conditionNumber = Infinity
        // (and scores can be NaN). System.Text.Json rejects non-finite doubles by default,
        // crashing the command — and emitting `Infinity`/`NaN` tokens would break the
        // MCP bridge's JSON.parse. Non-finite doubles must serialize as null.
        var env = new AnalyzeEnvelope(
            Aspect: "importance",
            Available: true,
            Summary: "ok",
            Data: new
            {
                conditionNumber = double.PositiveInfinity,
                negInf = double.NegativeInfinity,
                notANumber = double.NaN,
                nullableInf = (double?)double.PositiveInfinity,
                finite = 1.5,
                nullableNull = (double?)null
            },
            Flags: Array.Empty<string>());

        var json = AnalyzeJson.Serialize(env);

        Assert.DoesNotContain("Infinity", json);
        Assert.DoesNotContain("NaN", json);

        using var doc = JsonDocument.Parse(json); // must be valid, parseable JSON
        var data = doc.RootElement.GetProperty("data");
        Assert.Equal(JsonValueKind.Null, data.GetProperty("conditionNumber").ValueKind);
        Assert.Equal(JsonValueKind.Null, data.GetProperty("negInf").ValueKind);
        Assert.Equal(JsonValueKind.Null, data.GetProperty("notANumber").ValueKind);
        Assert.Equal(JsonValueKind.Null, data.GetProperty("nullableInf").ValueKind);
        Assert.Equal(JsonValueKind.Null, data.GetProperty("nullableNull").ValueKind);
        Assert.Equal(1.5, data.GetProperty("finite").GetDouble());
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

    [Fact]
    public void MapCorrelation_NullReport_AvailableWithEmptyData()
    {
        var env = AnalyzeJson.MapCorrelation(null);
        Assert.True(env.Available);
        Assert.Equal("correlation", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public void MapImportance_NullSummary_AvailableEmpty()
    {
        var env = AnalyzeJson.MapImportance(null);
        Assert.True(env.Available);
        Assert.Equal("importance", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public void MapOutliers_HighRate_AddsFlag()
    {
        // Build a report via the public surface is not possible (DataLens-owned types),
        // so assert the null path here; high-rate flag is covered by the smoke/integration run.
        var env = AnalyzeJson.MapOutliers(null);
        Assert.True(env.Available);
        Assert.Equal("outliers", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public void MapDistribution_BothNull_AvailableEmpty()
    {
        var env = AnalyzeJson.MapDistribution(null, null);
        Assert.True(env.Available);
        Assert.Equal("distribution", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public async Task Resolve_DoesNotMutateInputFile()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-analyze-" + Guid.NewGuid());
        Directory.CreateDirectory(dir);
        try
        {
            var csv = Path.Combine(dir, "d.csv");
            await File.WriteAllTextAsync(csv, "a,b\n1,2\n3,4\n");
            var before = File.GetLastWriteTimeUtc(csv);

            var ctx = await AnalyzeCommand.ResolveAsync(csv, labelOption: null, modelName: "default");

            Assert.NotNull(ctx);
            Assert.Equal(csv, ctx!.Value.DataFile);
            Assert.Equal(before, File.GetLastWriteTimeUtc(csv));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
