using System.Text.Json;
using MLoop.Core.AutoML;
using MLoop.Core.Preprocessing;

namespace MLoop.CLI.Commands.Policy;

/// <summary>One prep step as seen by an agent: its leakage category and fold-safety.</summary>
public sealed record PrepStepView(
    int Index,
    string Type,
    string? Method,
    IReadOnlyList<string>? Columns,
    string Category,
    bool FoldSafe,
    string? LeakageWarning);

/// <summary>The mutation a `prep plan` invocation applied (null for a list-only call).</summary>
public sealed record PrepPlanApplied(
    string Action,
    string? Type,
    string? Method,
    IReadOnlyList<string>? Columns,
    int? RemovedCount);

/// <summary>Structured JSON envelope for `mloop prep plan --json` (LLM-consumable).</summary>
public sealed record PrepPlanEnvelope(
    string Command,
    string Model,
    string Task,
    PrepPlanApplied? Applied,
    IReadOnlyList<PrepStepView> Prep,
    IReadOnlyList<string> Warnings);

/// <summary>The mutation a `features select` invocation applied (null is unused — always set).</summary>
public sealed record FeaturesApplied(
    string Action,
    IReadOnlyList<string>? Columns,
    int? ResetCount);

/// <summary>Structured JSON envelope for `mloop features select --json` (LLM-consumable).</summary>
public sealed record FeaturesSelectEnvelope(
    string Command,
    string Model,
    FeaturesApplied? Applied,
    IReadOnlyList<string> Ignored,
    IReadOnlyList<string> Warnings);

/// <summary>
/// Maps Phase 2 policy-command state to camelCase JSON envelopes, mirroring the
/// <see cref="AnalyzeJson"/> pattern so an agent reads the resulting plan structurally
/// rather than parsing the Spectre console table. Pure: no filesystem, no data fit.
/// </summary>
public static class PolicyJson
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static string Serialize(object envelope) => JsonSerializer.Serialize(envelope, Options);

    /// <summary>
    /// Projects each prep step to its leakage category, fold-safety, and (if leaky) the
    /// shared warning message. Reuses <see cref="PrepStepClassifier"/> and
    /// <see cref="AutoMLRunner.SupportsPreFeaturizer"/> — no new wording or routing logic.
    /// </summary>
    public static IReadOnlyList<PrepStepView> BuildStepViews(IReadOnlyList<PrepStep> prep, string task)
    {
        var supports = AutoMLRunner.SupportsPreFeaturizer(task);
        var views = new List<PrepStepView>(prep.Count);
        for (var i = 0; i < prep.Count; i++)
        {
            var step = prep[i];
            var category = PrepStepClassifier.Classify(step);
            var foldSafe = category == PrepCategory.PreFeaturizer && supports;
            var warning = category switch
            {
                PrepCategory.UnsupportedLeakageWarn => PrepStepClassifier.LeakageWarning(step),
                PrepCategory.PreFeaturizer when !supports => PrepStepClassifier.UnsupportedTaskLeakageWarning(step),
                _ => null
            };
            views.Add(new PrepStepView(
                i + 1, step.Type, step.Method, step.Columns, CategoryName(category), foldSafe, warning));
        }
        return views;
    }

    /// <summary>Distinct leakage warnings across the plan (envelope-level summary).</summary>
    public static IReadOnlyList<string> CollectWarnings(IReadOnlyList<PrepStepView> views) =>
        views.Where(v => v.LeakageWarning != null).Select(v => v.LeakageWarning!).Distinct().ToList();

    private static string CategoryName(PrepCategory category) => category switch
    {
        PrepCategory.PreFeaturizer => "preFeaturizer",
        PrepCategory.UnsupportedLeakageWarn => "unsupportedLeakageWarn",
        _ => "csvStage"
    };
}
