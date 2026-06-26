using MLoop.CLI.Commands.Policy;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands.Policy;

public class PolicyJsonTests
{
    [Fact]
    public void BuildStepViews_NormalizeOnRegression_IsFoldSafeWithNoWarning()
    {
        var prep = new List<PrepStep> { new() { Type = "normalize", Method = "z-score", Columns = ["a", "b"] } };
        var views = PolicyJson.BuildStepViews(prep, "regression");
        Assert.Single(views);
        Assert.Equal(1, views[0].Index);
        Assert.Equal("normalize", views[0].Type);
        Assert.Equal("preFeaturizer", views[0].Category);
        Assert.True(views[0].FoldSafe);
        Assert.Null(views[0].LeakageWarning);
    }

    [Fact]
    public void BuildStepViews_NormalizeOnUnsupportedTask_NotFoldSafeWithWarning()
    {
        var prep = new List<PrepStep> { new() { Type = "normalize", Method = "z-score", Columns = ["a"] } };
        var views = PolicyJson.BuildStepViews(prep, "clustering");
        Assert.Equal("preFeaturizer", views[0].Category);
        Assert.False(views[0].FoldSafe);
        Assert.NotNull(views[0].LeakageWarning); // task does not support preFeaturizer → leakage
    }

    [Fact]
    public void BuildStepViews_MedianFill_IsUnsupportedLeakageCategory()
    {
        var prep = new List<PrepStep> { new() { Type = "fill-missing", Method = "median", Columns = ["a"] } };
        var views = PolicyJson.BuildStepViews(prep, "regression");
        Assert.Equal("unsupportedLeakageWarn", views[0].Category);
        Assert.False(views[0].FoldSafe);
        Assert.NotNull(views[0].LeakageWarning);
    }

    [Fact]
    public void BuildStepViews_RemoveColumns_IsCsvStageWithNoWarning()
    {
        var prep = new List<PrepStep> { new() { Type = "remove-columns", Columns = ["a"] } };
        var views = PolicyJson.BuildStepViews(prep, "regression");
        Assert.Equal("csvStage", views[0].Category);
        Assert.False(views[0].FoldSafe);
        Assert.Null(views[0].LeakageWarning);
    }

    [Fact]
    public void Serialize_PrepPlanEnvelope_UsesCamelCaseAndIsValidJson()
    {
        var env = new PrepPlanEnvelope(
            Command: "prep plan",
            Model: "default",
            Task: "regression",
            Applied: new PrepPlanApplied("set", "normalize", "z-score", ["a"], RemovedCount: null),
            Prep: PolicyJson.BuildStepViews(
                new List<PrepStep> { new() { Type = "normalize", Method = "z-score", Columns = ["a"] } }, "regression"),
            Warnings: []);

        var json = PolicyJson.Serialize(env);

        Assert.Contains("\"command\": \"prep plan\"", json);
        Assert.Contains("\"foldSafe\": true", json);   // camelCase, not FoldSafe
        Assert.Contains("\"action\": \"set\"", json);
        // round-trips through a strict JSON parser
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.Equal("default", doc.RootElement.GetProperty("model").GetString());
    }

    [Fact]
    public void Serialize_FeaturesSelectEnvelope_UsesCamelCase()
    {
        var env = new FeaturesSelectEnvelope(
            Command: "features select",
            Model: "default",
            Applied: new FeaturesApplied("drop", ["id", "ts"], ResetCount: null),
            Ignored: ["id", "ts"],
            Warnings: []);

        var json = PolicyJson.Serialize(env);

        Assert.Contains("\"command\": \"features select\"", json);
        Assert.Contains("\"action\": \"drop\"", json);
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.Equal(2, doc.RootElement.GetProperty("ignored").GetArrayLength());
    }
}
