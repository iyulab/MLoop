using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// What a retraining set from feedback is written with — <c>mloop sample create --name</c> and
/// <c>POST /samples</c> read it here, so the two surfaces write one file the same way.
/// </summary>
public static class FeedbackSamples
{
    /// <summary>
    /// The columns the model trains with: its production experiment's label and weight, else the model's
    /// definition in mloop.yaml. <c>null</c> when neither names a label.
    /// </summary>
    public static async Task<RetrainingColumns?> ColumnsAsync(
        IFileSystemManager fileSystem, IProjectDiscovery discovery, string modelName, CancellationToken cancellationToken = default)
    {
        var store = new ExperimentStore(fileSystem, discovery);
        var registry = new ModelRegistry(fileSystem, discovery, store);

        var production = await registry.GetProductionAsync(modelName, cancellationToken);
        if (production?.ExperimentId is { } id && store.ExperimentExists(modelName, id))
        {
            var experiment = await store.LoadAsync(modelName, id, cancellationToken);
            if (!string.IsNullOrWhiteSpace(experiment.Config.LabelColumn))
                return new RetrainingColumns(experiment.Config.LabelColumn, experiment.Config.WeightColumn)
                {
                    Inputs = experiment.Config.InputSchema?.Columns.Select(c => c.Name).ToList() ?? [],
                };
        }

        var config = await new ConfigLoader(fileSystem, discovery).LoadUserConfigAsync(cancellationToken);
        return config.Models.TryGetValue(modelName, out var definition) && !string.IsNullOrWhiteSpace(definition.Label)
            ? new RetrainingColumns(definition.Label, definition.Weight)
            : null;
    }

    /// <summary>Why a model has no columns to write under — the sentence both surfaces refuse with.</summary>
    public static string NoLabelError(string modelName) => $"No label column is known for model '{modelName}'.";

    /// <summary>The guidance that goes with <see cref="NoLabelError"/>.</summary>
    public const string NoLabelGuidance =
        "The dataset is written under the label the model trains on — read from its production experiment, "
        + "or from `label:` under the model in mloop.yaml.";

    /// <summary>The sampling strategy a name stands for; <c>null</c> for an unknown name.</summary>
    public static SamplingStrategy? ParseStrategy(string name) => name.ToLowerInvariant() switch
    {
        "random" => SamplingStrategy.Random,
        "recent" => SamplingStrategy.Recent,
        "feedback-priority" or "feedbackpriority" => SamplingStrategy.FeedbackPriority,
        "stratified" => SamplingStrategy.Stratified,
        "low-confidence" or "lowconfidence" => SamplingStrategy.LowConfidence,
        _ => null
    };

    /// <summary>What to say about an unknown strategy.</summary>
    public const string StrategyGuidance = "Valid strategies: random, recent, feedback-priority, stratified, low-confidence";

    /// <summary>Where a sample goes when no path is given: <c>.mloop/samples/&lt;model&gt;_&lt;local time&gt;.csv</c>.</summary>
    public static string DefaultPath(string projectRoot, string modelName) =>
        Path.Combine(projectRoot, ".mloop", "samples", $"{modelName.ToLowerInvariant()}_{TimestampKeys.Stamp(DateTime.Now)}.csv");
}
