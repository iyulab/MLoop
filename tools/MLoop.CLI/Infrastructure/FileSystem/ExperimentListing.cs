using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// One experiment as a machine-readable list shows it: its summary, its own metrics, the quality
/// gate's verdict and whether it is the production model.
/// </summary>
/// <param name="QualityGate"><c>null</c> when the fixed quality gate passes; otherwise why a promotion is refused without force. The project's own <c>gate:</c> policy is judged at promotion — it costs an evaluation — and is not part of the listing.</param>
/// <param name="QualityWarning"><c>null</c> unless the gate passes a model whose predicted labels are still not usable.</param>
public sealed record ExperimentListRow(
    string ModelName,
    string ExperimentId,
    DateTime Timestamp,
    string Status,
    string? BestTrainer,
    string? MetricName,
    double? BestMetric,
    string? LabelColumn,
    double? TrainingTimeSeconds,
    Dictionary<string, double>? Metrics,
    string? QualityGate,
    string? QualityWarning,
    bool IsProduction);

/// <summary>
/// The rows of <c>mloop list --json</c> and <c>GET /experiments</c> — one construction, so the two
/// surfaces answer the same question the same way. Before it, the HTTP rows carried neither the
/// metrics nor the production flag, and a client gating promotion rebuilt them with a call per row.
/// </summary>
public static class ExperimentListing
{
    /// <summary>The experiments of <paramref name="modelName"/> (all models when <c>null</c>), newest first, and each model's production experiment.</summary>
    public static async Task<(IReadOnlyList<ExperimentListRow> Rows, IReadOnlyDictionary<string, string> Production)> ListAsync(
        IExperimentStore experimentStore,
        IModelRegistry registry,
        string? modelName,
        bool completedOnly,
        CancellationToken cancellationToken = default)
    {
        var production = (await registry.ListAsync(null, cancellationToken))
            .ToDictionary(m => m.ModelName, m => m.ExperimentId);

        var rows = new List<ExperimentListRow>();
        foreach (var e in (await experimentStore.ListAsync(modelName, cancellationToken)).OrderByDescending(e => e.Timestamp))
        {
            if (completedOnly && !e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                continue;
            var expModelName = e.ModelName ?? ConfigDefaults.DefaultModelName;
            ExperimentData? detail;
            try { detail = await experimentStore.LoadAsync(expModelName, e.ExperimentId, cancellationToken); }
            catch (Exception ex) when (ex is not OperationCanceledException) { detail = null; } // a partial experiment directory lists without them

            rows.Add(new ExperimentListRow(
                expModelName,
                e.ExperimentId,
                e.Timestamp,
                e.Status,
                e.BestTrainer,
                e.MetricName,
                e.BestMetric,
                e.LabelColumn,
                e.TrainingTimeSeconds,
                detail?.Metrics,
                detail is null ? null : ModelRegistry.QualityGateRefusal(detail, detail.Config?.Metric),
                detail is null ? null : ModelRegistry.QualityGateWarning(detail),
                production.TryGetValue(expModelName, out var productionId) && e.ExperimentId == productionId));
        }
        return (rows, production);
    }
}
