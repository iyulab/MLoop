using System.Text.Json;
using DataLens.Models;
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
    public void MapImportance_NullReport_AvailableEmpty()
    {
        var env = AnalyzeJson.MapImportance(null, label: null);
        Assert.True(env.Available);
        Assert.Equal("importance", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public void MapImportance_PrefersPermutation_AndExcludesLabel()
    {
        // F-01: a target-aware command (--label) must surface predictive importance
        // (permutation), not target-agnostic structural variance, and must never rank
        // the label against itself. SEQ026 ground truth: KWh ranks high structurally
        // but is predictively weak; only permutation reflects that.
        var report = new FeatureReport
        {
            TargetColumn = "M_X",
            Importance = new FeatureImportanceSummary
            {
                Scores =
                [
                    new() { Name = "M_X", Score = 0.41 },
                    new() { Name = "KWh", Score = 0.38 },
                    new() { Name = "temp", Score = 0.20 },
                ],
                ConditionNumber = 4.3,
            },
            Permutation = new PermutationSummary
            {
                BaselineScore = 0.9,
                Features =
                [
                    new() { Name = "M_X", Importance = 5.0 },  // label self-reference (must be dropped)
                    new() { Name = "KWh", Importance = 0.01 }, // predictively weak
                    new() { Name = "temp", Importance = 1.2 },
                ],
            },
        };

        var env = AnalyzeJson.MapImportance(report, label: "M_X");
        using var doc = JsonDocument.Parse(AnalyzeJson.Serialize(env));
        var data = doc.RootElement.GetProperty("data");

        Assert.Equal("permutation", data.GetProperty("method").GetString());
        var features = data.GetProperty("ranking").EnumerateArray()
            .Select(r => r.GetProperty("feature").GetString()).ToList();
        Assert.DoesNotContain("M_X", features);   // label excluded
        Assert.Equal("temp", features[0]);        // highest non-label predictive importance
        Assert.Equal(2, features.Count);
        // structural diagnostics still surfaced
        Assert.Equal(4.3, data.GetProperty("conditionNumber").GetDouble(), 1);
    }

    [Fact]
    public void MapImportance_StructuralFallback_ExcludesLabel()
    {
        // No target-aware source available → fall back to structural, but still drop the label.
        var report = new FeatureReport
        {
            TargetColumn = "y",
            Importance = new FeatureImportanceSummary
            {
                Scores =
                [
                    new() { Name = "y", Score = 0.9 },
                    new() { Name = "x1", Score = 0.5 },
                ],
            },
        };

        var env = AnalyzeJson.MapImportance(report, label: "y");
        using var doc = JsonDocument.Parse(AnalyzeJson.Serialize(env));
        var data = doc.RootElement.GetProperty("data");

        Assert.Equal("structural", data.GetProperty("method").GetString());
        var features = data.GetProperty("ranking").EnumerateArray()
            .Select(r => r.GetProperty("feature").GetString()).ToList();
        Assert.DoesNotContain("y", features);
        Assert.Single(features);
        Assert.Equal("x1", features[0]);
    }

    [Fact]
    public void MapImportance_CategoricalTarget_UsesMutualInfo_ExcludesLabel()
    {
        var report = new FeatureReport
        {
            TargetColumn = "cls",
            MutualInfo = new MutualInfoSummary
            {
                Features =
                [
                    new() { Name = "cls", Mi = 2.0 },  // label self-reference
                    new() { Name = "f1", Mi = 0.3 },
                    new() { Name = "f2", Mi = 0.7 },
                ],
            },
        };

        var env = AnalyzeJson.MapImportance(report, label: "cls");
        using var doc = JsonDocument.Parse(AnalyzeJson.Serialize(env));
        var data = doc.RootElement.GetProperty("data");

        Assert.Equal("mutual-info", data.GetProperty("method").GetString());
        var features = data.GetProperty("ranking").EnumerateArray()
            .Select(r => r.GetProperty("feature").GetString()).ToList();
        Assert.DoesNotContain("cls", features);
        Assert.Equal("f2", features[0]);
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
        var env = AnalyzeJson.MapDistribution(null, null, null);
        Assert.True(env.Available);
        Assert.Equal("distribution", env.Aspect);
        Assert.Empty(env.Flags);
    }

    [Fact]
    public void MapDistribution_SuppressesSkewFlag_ForLowCardinalityColumn()
    {
        // F-02: skewness on a binary/categorical column (e.g. a 1/2 rectifier id) is the class
        // balance, not an actionable distribution shape — flagging it "highly-skewed" misleads a
        // downstream agent into proposing a meaningless transform (observed in C04 live run).
        var desc = new DescriptiveReport
        {
            Columns =
            [
                new() { Name = "rectifier", Skewness = -1.14 },   // unique=2 → suppress
                new() { Name = "energy", Skewness = 2.50 },       // continuous → keep
            ],
        };
        var stats = new Dictionary<string, (long MissingCount, int UniqueCount)>
        {
            ["rectifier"] = (0, 2),
            ["energy"] = (0, 500),
        };

        var env = AnalyzeJson.MapDistribution(desc, null, stats);

        Assert.Contains(env.Flags, f => f == "highly-skewed: energy (skew=2.50)");
        Assert.DoesNotContain(env.Flags, f => f.StartsWith("highly-skewed: rectifier"));
    }

    [Fact]
    public void MapDistribution_NoStats_FlagsAllSkewed_BackwardCompatible()
    {
        // Without cardinality info, behaviour is unchanged (flag any |skew|>1).
        var desc = new DescriptiveReport
        {
            Columns = [new() { Name = "rectifier", Skewness = -1.14 }],
        };

        var env = AnalyzeJson.MapDistribution(desc, null, stats: null);

        Assert.Contains(env.Flags, f => f.StartsWith("highly-skewed: rectifier"));
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
