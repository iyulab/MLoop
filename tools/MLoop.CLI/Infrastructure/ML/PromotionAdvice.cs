using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>An experiment ranked ahead but kept out of production, and why.</summary>
public sealed record HeldExperiment(string ExperimentId, string Reason);

/// <summary>
/// Which experiment a ranking may recommend for production: the best one the gates accept. The ranking is
/// by metric, and a metric does not know the fixed quality gate or the project's own <c>gate:</c> policy —
/// recommending the first of the ranking sent users to a <c>promote</c> that refused it.
/// </summary>
/// <remarks>
/// The same two checks <c>promote</c> applies to an experiment named outright
/// (<see cref="ModelRegistry.QualityGateRefusal"/>, then <see cref="IModelRegistry.GatePolicyRefusalAsync"/>);
/// the second scores the experiment on the policy's evaluation set, so the walk stops at the first experiment
/// accepted or the one already in production, and never evaluates the ones below.
/// </remarks>
public static class PromotionAdvice
{
    /// <param name="bestFirst">The experiments, best first, of any models.</param>
    /// <param name="productionByModel">Each model's production experiment.</param>
    public static async Task<(string? Recommended, IReadOnlyList<HeldExperiment> Held)> ChooseAsync(
        IEnumerable<ExperimentData> bestFirst,
        IReadOnlyDictionary<string, string> productionByModel,
        IModelRegistry registry,
        CancellationToken cancellationToken = default)
    {
        var held = new List<HeldExperiment>();
        foreach (var experiment in bestFirst)
        {
            if (productionByModel.TryGetValue(experiment.ModelName, out var production) && production == experiment.ExperimentId)
                return (null, held);   // production is the best the gates let through

            var refusal = ModelRegistry.QualityGateRefusal(experiment, experiment.Config?.Metric)
                ?? await registry.GatePolicyRefusalAsync(experiment.ModelName, experiment.ExperimentId, cancellationToken);
            if (refusal is null)
                return (experiment.ExperimentId, held);
            held.Add(new HeldExperiment(experiment.ExperimentId, refusal));
        }
        return (null, held);
    }
}
