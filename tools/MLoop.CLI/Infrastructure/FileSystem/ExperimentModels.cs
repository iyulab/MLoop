using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Where a named experiment's trained model is, or why there is none to use — one answer, and one
/// pair of sentences, for every command and endpoint that takes an experiment to score with.
/// </summary>
/// <remarks>
/// Each site used to check existence, build the path and check the file itself, and they disagreed:
/// one printed the absolute path, another named no reason, and the evaluate endpoint, finding no
/// model in the experiment, evaluated <i>production's</i> model while reporting the experiment's id.
/// </remarks>
public static class ExperimentModels
{
    /// <summary>The sentence for an experiment that does not exist.</summary>
    public static string NotFound(string modelName, string experimentId)
        => $"Experiment not found: {experimentId} for model '{modelName}'.";

    /// <summary>The sentence for an experiment that kept its record but has no model.</summary>
    public static string NoTrainedModel(string experimentId)
        => $"'{experimentId}' has no trained model ({ExperimentLayout.ModelFileName}) — a failed or interrupted experiment keeps its record but not a model.";

    /// <summary>
    /// The trained model file of <paramref name="experimentId"/>, or <c>null</c> and the reason:
    /// the experiment does not exist, or it has no model.
    /// </summary>
    public static (string? ModelPath, string? Refusal) Find(IExperimentStore store, string modelName, string experimentId)
    {
        ArgumentNullException.ThrowIfNull(store);
        if (!store.ExperimentExists(modelName, experimentId))
            return (null, NotFound(modelName, experimentId));
        var path = Path.Combine(store.GetExperimentPath(modelName, experimentId), ExperimentLayout.ModelFileName);
        return File.Exists(path) ? (path, null) : (null, NoTrainedModel(experimentId));
    }
}
