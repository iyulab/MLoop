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
            Epochs = definition.Training?.Epochs,
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
            AnomalyThreshold = definition.AnomalyThreshold,
            UserColumn = definition.UserColumn,
            ItemColumn = definition.ItemColumn,
            PreFeaturizer = preFeaturizer,
            PreFeaturizerColumns = preFeaturizerColumns ?? [],
            WeightColumn = string.IsNullOrWhiteSpace(definition.Weight) ? null : definition.Weight.Trim(),
            Portable = definition.Portable == true,
            Ensemble = definition.Ensemble == true,
        };

    /// <summary>
    /// Why <paramref name="definition"/> cannot be trained as a blend, or <c>null</c>. A setting that silently did nothing would leave
    /// a single model where the user expects a blend, and a blend cannot be portable: the ONNX graph a portable model promises has no
    /// form for the weighted average that joins its members.
    /// </summary>
    public static string? EnsembleConflict(ModelDefinition definition)
    {
        if (definition.Ensemble != true)
            return null;
        if (!MLoop.Core.Training.TrainingRunner.SupportsEnsemble(definition.Task))
            return $"ensemble: a blend is made for binary-classification or regression, not '{definition.Task}'.";
        return definition.Portable == true
            ? "ensemble: a blend of several models cannot be exported to ONNX, which a portable model promises; use one of --ensemble and --portable."
            : null;
    }

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
    /// Why the test split in <paramref name="definition"/> cannot be honoured, or <c>null</c>. It is the share of the rows held out to
    /// measure the model on, so it lies between 0 and 1: a share of nothing leaves the scores to be those of rows the model was trained on,
    /// reported as if they were a test's. A task that holds nothing out (a time series keeps its order, an unsupervised task has no label)
    /// has no share to refuse.
    /// </summary>
    public static string? TestSplitConflict(ModelDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (definition.Training?.TestSplit is not { } share)
            return null;
        if (!MLoop.Core.Training.TrainingRunner.RequiresLabel(definition.Task) || MLoop.Core.Training.TrainingRunner.IsTimeSeriesTask(definition.Task))
            return null;
        return share > 0 && share < 1
            ? null
            : string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"test split: {share} is not a share of the rows. It must be between 0 and 1 (exclusive): the part held out to measure the model on - with none held out the scores would be those of rows the model was trained on.");
    }

    /// <summary>
    /// Why the epoch count in <paramref name="definition"/> cannot be honoured, or <c>null</c>. A count for a task
    /// that does not train by epochs, or below one epoch, would be set and silently meant nothing.
    /// </summary>
    public static string? EpochsConflict(ModelDefinition definition)
    {
        if (definition.Training?.Epochs is not { } epochs)
            return null;
        if (epochs < 1)
            return $"epochs: {epochs} is not a number of epochs; use 1 or more, or leave it unset for the default.";
        return MLoop.Core.Training.TrainingRunner.EpochsUnused(definition.Task) is { } why
            ? $"epochs: '{definition.Task}' {why}, so it has no epoch count to set; leave --epochs / training.epochs unset."
            : null;
    }

    /// <summary>
    /// Why the anomaly threshold in <paramref name="definition"/> cannot be honoured, or <c>null</c>. It is a score of the spectral-residual
    /// detector, which lies above 0 and below 1, and only a <c>time-series-anomaly</c> model has one: set on another task it would be
    /// accepted and meant nothing.
    /// </summary>
    public static string? AnomalyThresholdConflict(ModelDefinition definition)
    {
        if (definition.AnomalyThreshold is not { } threshold)
            return null;
        if (!MLoop.Core.Training.TrainingRunner.SupportsAnomalyThreshold(definition.Task))
            return $"anomaly threshold: '{definition.Task}' has no detector score to set a threshold on; it is a setting of time-series-anomaly.";
        return MLoop.Core.Training.TrainingRunner.IsValidAnomalyThreshold(threshold)
            ? null
            : string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"anomaly threshold: {threshold} is not a score threshold. It must be above 0 and below 1 (the default is {MLoop.Core.Training.TrainingRunner.DefaultAnomalyThreshold}); a lower value flags more points.");
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
