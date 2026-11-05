using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;

/// <summary>
/// The model definition a <c>POST /train</c> job trains under: the project's mloop.yaml for that
/// model, with the request's task, label, time limit, metric, split and weight taking the place a
/// <c>mloop train</c> command line takes — the same merge, through the same
/// <see cref="ConfigMerger"/>, so the two surfaces cannot train one project two ways.
/// </summary>
public static class TrainingDefinition
{
    /// <summary>
    /// What a <c>POST /train</c> job asks of its model's definition: the command line's place. Named, not positional —
    /// two string fields (task, label) and three integers (time, trials, epochs) sit side by side, and a new field
    /// is one line here and one at each call.
    /// </summary>
    /// <param name="ModelName">The model whose mloop.yaml definition is read.</param>
    /// <param name="Task">The task (<c>--task</c>); null leaves it to mloop.yaml.</param>
    /// <param name="Label">The label column (<c>--label</c>); null leaves it to mloop.yaml.</param>
    /// <param name="TimeLimitSeconds">The training budget (<c>--time</c>); null leaves it to mloop.yaml.</param>
    /// <param name="Metric">The metric to optimize; <c>auto</c> leaves it to mloop.yaml, then the task.</param>
    /// <param name="TestSplit">The held-out share (<c>--test-split</c>); null leaves it to mloop.yaml.</param>
    /// <param name="Weight">The weight column (<c>--weight</c>).</param>
    /// <param name="MaxTrials">At most this many trials (<c>--max-trials</c>).</param>
    /// <param name="Epochs">Epochs of a deep-learning task (<c>--epochs</c>).</param>
    /// <param name="Ensemble"><c>true</c> asks for a blend (<c>--ensemble</c>); false leaves it to mloop.yaml.</param>
    public sealed record Request(
        string ModelName,
        string? Task,
        string? Label,
        int? TimeLimitSeconds,
        string Metric,
        double? TestSplit,
        string? Weight = null,
        int? MaxTrials = null,
        int? Epochs = null,
        bool Ensemble = false);

    /// <summary>The definition, or why the request cannot be trained as asked.</summary>
    public static async Task<(ModelDefinition? Definition, string? Error)> ResolveAsync(
        ConfigLoader configLoader, Request request, CancellationToken cancellationToken = default)
    {
        var (modelName, task, label, timeLimitSeconds, metric, testSplit, weight, maxTrials, epochs, ensemble) = request;
        var config = await configLoader.LoadUserConfigAsync(cancellationToken);
        ModelDefinition definition;
        try
        {
            definition = new ConfigMerger().GetEffectiveModelDefinition(
                config, modelName, cliLabel: label, cliTask: task,
                cliTraining: new TrainingSettings
                {
                    TimeLimitSeconds = timeLimitSeconds,
                    MaxTrials = maxTrials,
                    Epochs = epochs,
                    // "auto" is the request's default — it leaves the choice to mloop.yaml, then the task.
                    Metric = string.Equals(metric, "auto", StringComparison.OrdinalIgnoreCase) ? null : metric,
                    TestSplit = testSplit,
                });
        }
        catch (InvalidOperationException ex)
        {
            return (null, ex.Message);
        }

        if (definition.Prep is { Count: > 0 })
            return (null, $"mloop.yaml defines preprocessing (prep) for model '{modelName}', which POST /train does not run; " +
                          "train it with `mloop train`, or remove the prep steps.");
        // The request's weight in --weight's place; refused, like the definition's, where the run could not honour it.
        if (!string.IsNullOrWhiteSpace(weight))
            definition.Weight = weight.Trim();
        // The request's ensemble in --ensemble's place: it asks for a blend; leaving it out leaves the choice to mloop.yaml.
        if (ensemble)
            definition.Ensemble = true;
        if ((ModelTrainingConfig.MaxTrialsConflict(definition) ?? ModelTrainingConfig.TestSplitConflict(definition) ?? ModelTrainingConfig.EpochsConflict(definition) ?? ModelTrainingConfig.AnomalyThresholdConflict(definition)) is { } maxTrialsError)
            return (null, maxTrialsError);
        if (ModelTrainingConfig.WeightConflict(definition) is { } weightError)
            return (null, weightError);
        if (ModelTrainingConfig.PortableConflict(definition) is { } portableError)
            return (null, portableError);
        if (ModelTrainingConfig.EnsembleConflict(definition) is { } ensembleError)
            return (null, ensembleError);
        // The task, the label and the columns the task needs — as `mloop train` asks them.
        if (TrainingInput.DefinitionRefusal(definition) is { } refusal)
            return (null, refusal.Hint is null ? refusal.Message : $"{refusal.Message} {refusal.Hint}");
        return (definition, null);
    }
}
