using System.Globalization;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Judges a model against its project's gate policy (<see cref="GatePolicy"/>): scores it on the
/// policy's evaluation set and returns why it is refused, or <c>null</c> when it passes. Read by every
/// path that promotes — the same decision whether a person, a training run or the REST API asks.
/// </summary>
public static class GatePolicyCheck
{
    /// <summary>Slices with fewer rows than this are not held to the limits: their shares move too much from one sample to the next.</summary>
    public const int MinimumSliceRows = 30;

    /// <summary>
    /// Why <paramref name="experimentId"/> does not pass <paramref name="policy"/>, or <c>null</c> when it
    /// does or the policy has no rule. A rule that cannot be judged — no evaluation set, a missing file, a
    /// class limit on a task without classes, a production model trained for another question — refuses.
    /// </summary>
    public static async Task<string?> RefusalAsync(
        GatePolicy? policy, string projectRoot, string modelName, string experimentId, ExperimentData candidate,
        IExperimentStore store, IModelRegistry registry, CancellationToken cancellationToken = default)
    {
        if (policy is not { HasRules: true })
            return null;
        if (string.IsNullOrWhiteSpace(policy.Data))
            return "the gate policy has rules but no evaluation set (gate.data)";
        var data = ProjectPaths.Resolve(policy.Data, projectRoot);
        if (!TabularDataFile.Exists(data))
            return $"the gate policy's evaluation set was not found: {policy.Data}";
        data = await TabularDataFile.AsCsvAsync(data);

        var modelPath = Path.Combine(store.GetExperimentPath(modelName, experimentId), Core.Storage.ExperimentLayout.ModelFileName);
        var (weight, _) = ModelComparison.ResolveWeight(policy.Weight, candidate, data);
        var engine = new EvaluationEngine();
        var reasons = new List<string>();

        if (policy.Limits is { Count: > 0 } limits)
        {
            var overall = await engine.ConfusionAsync(modelPath, data, candidate.Config.LabelColumn, candidate.Task,
                candidate.Config.InputSchema, weight, cancellationToken);
            if (overall is null)
                reasons.Add($"class limits apply to classifiers, and {candidate.Task} has no classes");
            else
            {
                reasons.AddRange(Exceeded(limits, overall, scope: null));
                if (!string.IsNullOrWhiteSpace(policy.SliceBy))
                    reasons.AddRange(await SliceReasonsAsync(policy.SliceBy.Trim(), limits, data, modelPath, candidate, weight, engine, cancellationToken));
            }
        }

        if (policy.NoWorseThanProduction)
        {
            var production = await registry.GetProductionAsync(modelName, cancellationToken);
            if (production is not null && !production.ExperimentId.Equals(experimentId, StringComparison.OrdinalIgnoreCase))
            {
                var baseline = await ModelComparison.ResolveBaselineAsync("production", modelName, experimentId, candidate, store, registry, cancellationToken);
                if (baseline.Error is { } error)
                    reasons.Add($"it cannot be compared with production: {error}");
                else
                {
                    var comparison = await ModelComparison.CompareAsync(engine, experimentId, modelPath, candidate, baseline, data, weight, cancellationToken);
                    if (comparison.Paired is { Verdict: PairedVerdict.BaselineBetter } p)
                        reasons.Add(string.Create(CultureInfo.InvariantCulture,
                            $"production ({baseline.ExperimentId}) is better on the gate data — {p.Measure} {p.Candidate:0.0000} vs {p.Baseline:0.0000}, difference {p.Difference:+0.0000;-0.0000} (95% interval {p.Low:+0.0000;-0.0000} to {p.High:+0.0000;-0.0000})"));
                    else if (comparison.Paired is null)
                        reasons.Add($"it cannot be compared with production row by row: {comparison.NotPaired}");
                }
            }
        }

        return reasons.Count == 0 ? null : string.Join("; ", reasons);
    }

    private static IEnumerable<string> Exceeded(IEnumerable<GateLimit> limits, ConfusionCounts confusion, string? scope)
    {
        foreach (var limit in limits)
        {
            if (confusion.ShareOf(limit.Actual, limit.Predicted) is not { } share || share <= limit.Max)
                continue;
            var where = scope is null ? "" : $" in {scope}";
            // Shares in the output culture, as every other percentage the CLI prints.
            yield return $"of the rows labeled {string.Join("|", limit.Actual)}{where}, {share:P1} were answered {string.Join("|", limit.Predicted)} (limit {limit.Max:P1})";
        }
    }

    private static async Task<IReadOnlyList<string>> SliceReasonsAsync(
        string sliceBy, IReadOnlyList<GateLimit> limits, string data, string modelPath, ExperimentData candidate,
        string? weight, EvaluationEngine engine, CancellationToken cancellationToken)
    {
        var directory = Path.Combine(Path.GetTempPath(), "mloop-gate-" + Guid.NewGuid().ToString("N"));
        try
        {
            var reasons = new List<string>();
            foreach (var part in await CsvPartition.ByColumnAsync(data, sliceBy, directory))
            {
                if (part.Rows < MinimumSliceRows)
                    continue;
                var confusion = await engine.ConfusionAsync(modelPath, part.Path, candidate.Config.LabelColumn, candidate.Task,
                    candidate.Config.InputSchema, weight, cancellationToken);
                if (confusion is not null)
                    reasons.AddRange(Exceeded(limits, confusion, $"{sliceBy}={part.Value}"));
            }
            return reasons;
        }
        finally
        {
            try { Directory.Delete(directory, recursive: true); } catch { /* temp files; the OS reclaims them */ }
        }
    }
}
