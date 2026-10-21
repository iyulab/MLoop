using MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// The model definition a <c>POST /train</c> job trains under: the project's mloop.yaml for that
/// model, with the request's task, label, time limit, metric, split and weight taking the place a
/// <c>mloop train</c> command line takes — the same merge, through the same
/// <see cref="ConfigMerger"/>, so the two surfaces cannot train one project two ways.
/// </summary>
public static class TrainingDefinition
{
    /// <summary>The definition, or why the request cannot be trained as asked.</summary>
    public static async Task<(ModelDefinition? Definition, string? Error)> ResolveAsync(
        ConfigLoader configLoader, string modelName, string task, string label, int? timeLimitSeconds,
        string metric, double? testSplit, string? weight = null, CancellationToken cancellationToken = default)
    {
        var config = await configLoader.LoadUserConfigAsync(cancellationToken);
        ModelDefinition definition;
        try
        {
            definition = new ConfigMerger().GetEffectiveModelDefinition(
                config, modelName, cliLabel: label, cliTask: task,
                cliTraining: new TrainingSettings
                {
                    TimeLimitSeconds = timeLimitSeconds,
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
        if (ModelTrainingConfig.WeightConflict(definition) is { } weightError)
            return (null, weightError);
        if (ModelTrainingConfig.PortableConflict(definition) is { } portableError)
            return (null, portableError);
        return (definition, null);
    }
}
