using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// The one place a model definition (mloop.yaml merged with the caller's overrides) becomes a
/// <see cref="TrainingConfig"/> — shared by <c>mloop train</c> and <c>POST /train</c>, so a project
/// trains the same way whichever surface starts it. Before it, the API built its config from the
/// request alone: column overrides, the weight, and the ranking/recommendation/time-series columns of
/// the definition were never read, so an <c>ignore</c>d column was a feature and a weighted model was
/// trained unweighted.
/// </summary>
public static class ModelTrainingConfig
{
    /// <summary>The training configuration <paramref name="definition"/> describes.</summary>
    public static TrainingConfig From(
        ModelDefinition definition,
        string modelName,
        string dataFile,
        int timeLimitSeconds,
        double testSplit,
        string? testDataFile = null,
        bool useAutoTime = false,
        IEstimator<ITransformer>? preFeaturizer = null,
        List<string>? preFeaturizerColumns = null)
        => new()
        {
            ModelName = modelName,
            DataFile = dataFile,
            LabelColumn = definition.Label,
            Task = definition.Task,
            TimeLimitSeconds = timeLimitSeconds,
            MaxTrials = definition.Training?.MaxTrials,
            Metric = definition.Training?.Metric ?? ConfigDefaults.DefaultMetric,
            TestSplit = testSplit,
            TestDataFile = testDataFile,
            UseAutoTime = useAutoTime,
            ColumnOverrides = definition.Columns?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.Type),
            TextEncoders = TextEncodersOf(definition),
            ImageEncoder = string.IsNullOrWhiteSpace(definition.Encoder) ? null : definition.Encoder.Trim(),
            BankSize = definition.BankSize,
            NumClusters = definition.NumClusters ?? 0,
            GroupColumn = definition.GroupColumn,
            Horizon = definition.Horizon ?? 0,
            WindowSize = definition.WindowSize ?? 0,
            SeriesLength = definition.SeriesLength ?? 0,
            UserColumn = definition.UserColumn,
            ItemColumn = definition.ItemColumn,
            PreFeaturizer = preFeaturizer,
            PreFeaturizerColumns = preFeaturizerColumns ?? [],
            WeightColumn = string.IsNullOrWhiteSpace(definition.Weight) ? null : definition.Weight.Trim(),
            Portable = definition.Portable == true,
        };

    /// <summary>
    /// Why <paramref name="definition"/> cannot be trained portable, or <c>null</c>. A setting that
    /// silently did nothing would leave a model the user expects to export unexportable.
    /// </summary>
    public static string? PortableConflict(ModelDefinition definition)
    {
        if (definition.Portable != true)
            return null;
        if (!MLoop.Core.Training.TrainingRunner.SupportsPreFeaturizer(definition.Task))
            return $"portable: a portable model is trained for binary-classification, multiclass-classification or regression, not '{definition.Task}'.";
        return TextEncodersOf(definition) is { } encoders
            ? $"portable: column {string.Join(", ", encoders.Keys.Select(c => $"'{c}'"))} is read through a text encoder, which an exported graph does not carry."
            : null;
    }

    /// <summary>The columns <paramref name="definition"/> reads through a text encoder, with the encoders' ids; null when none.</summary>
    public static IReadOnlyDictionary<string, string>? TextEncodersOf(ModelDefinition definition)
    {
        var encoders = definition.Columns?
            .Where(c => !string.IsNullOrWhiteSpace(c.Value.Encoder))
            .ToDictionary(c => c.Key, c => c.Value.Encoder!.Trim());
        return encoders is { Count: > 0 } ? encoders : null;
    }

    /// <summary>
    /// Why the trial limit in <paramref name="definition"/> cannot be honoured, or <c>null</c>. A limit on trials a task
    /// does not run, or one below a single trial, would be set and silently meant nothing.
    /// </summary>
    public static string? MaxTrialsConflict(ModelDefinition definition)
    {
        if (definition.Training?.MaxTrials is not { } maxTrials)
            return null;
        if (maxTrials < 1)
            return $"max trials: {maxTrials} is not a number of trials; use 1 or more, or leave it unset for the time limit alone.";
        return MLoop.Core.Training.TrainingRunner.TimeBudgetUnused(definition.Task) is { } why
            ? $"max trials: '{definition.Task}' {why}, so it has no trials to count; leave --max-trials / training.max_trials unset."
            : null;
    }

    /// <summary>
    /// Why the weight column in <paramref name="definition"/> cannot be honoured, or <c>null</c>.
    /// A weight that silently did nothing would train an unweighted model believed weighted.
    /// </summary>
    public static string? WeightConflict(ModelDefinition definition, string? balance = null)
    {
        var weight = definition.Weight?.Trim();
        if (string.IsNullOrEmpty(weight))
            return null;
        if (!RowWeights.Supports(definition.Task))
            return $"weight '{weight}': weighted training applies to binary-classification, multiclass-classification and regression, not '{definition.Task}'.";
        if (string.Equals(weight, definition.Label, StringComparison.OrdinalIgnoreCase))
            return $"weight '{weight}' is the label column; the weight must be a column of its own.";
        if (definition.Columns?.FirstOrDefault(c => string.Equals(c.Key.Trim(), weight, StringComparison.OrdinalIgnoreCase)) is { Key: not null } declared
            && !string.Equals(declared.Value.Type, "ignore", StringComparison.OrdinalIgnoreCase))
            return $"weight '{weight}' is also declared as a {declared.Value.Type} column in mloop.yaml; a weight is never a feature.";
        if (!string.IsNullOrEmpty(balance) && !string.Equals(balance, "none", StringComparison.OrdinalIgnoreCase))
            return $"--balance and weight '{weight}' both change how often rows count; use one of them.";
        return null;
    }
}
