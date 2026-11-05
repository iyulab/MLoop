using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Builds the effective definition of a model from mloop.yaml and the command line, with priority:
/// CLI Args > mloop.yaml > Defaults
/// </summary>
public class ConfigMerger
{
    /// <summary>
    /// Gets effective model definition by merging model-specific settings with defaults
    /// </summary>
    public ModelDefinition GetEffectiveModelDefinition(
        MLoopConfig config,
        string modelName,
        string? cliLabel = null,
        string? cliTask = null,
        TrainingSettings? cliTraining = null)
    {
        // Get base model definition or create new one
        ModelDefinition? baseDefinition = null;
        config.Models.TryGetValue(modelName, out baseDefinition);

        // Determine final values with CLI override priority
        var task = cliTask
            ?? baseDefinition?.Task
            ?? throw new InvalidOperationException($"Task not specified for model '{modelName}'. Use --task or define in mloop.yaml");

        // Canonicalize once, here — the same reason the metric sentinel is resolved below: this is
        // the one place in the merge chain where the task is known, and everything downstream used
        // to normalize it again, differently. An unrecognized spelling is kept verbatim so the
        // caller's error message can quote what the user actually wrote.
        task = TaskTypes.Canonical(task) ?? task;

        // Label is optional for unsupervised tasks (TrainingRunner.RequiresLabel — single source).
        var label = cliLabel
            ?? baseDefinition?.Label
            ?? (TrainingRunner.LabelColumnMustBeNamed(task)
                ? throw new InvalidOperationException(
                    $"Label not specified for model '{modelName}'. Use <label> argument or define in mloop.yaml")
                // A directory of images has its loader make the label column; the name it makes is this one.
                : DataLoaderFactory.IsDirectoryBased(task) ? CocoDataLoader.DefaultLabelColumn : "");

        var training = MergeTrainingSettings(
            ConfigDefaults.CreateDefaultTrainingSettings(),
            baseDefinition?.Training,
            cliTraining);

        // "auto" is a *request* ("let the task decide"), not a *result*. Resolve it here — the one
        // place in the merge chain where the task is known — so everything downstream records the
        // metric the experiment actually optimized. Leaving the sentinel to flow through made
        // metadata.json/config.json/experiment-index report metricName "auto" for any model without
        // an explicit `training.metric` in mloop.yaml, i.e. models registered by
        // `train --name <new>` rather than `init` (which resolves via PrimaryMetricOrAuto itself).
        // That mislabels the score and makes direction lookups treat it as an unknown metric.
        // The same family — TaskMetadata is the authority; ConfigMerger was the one
        // consumer still missing from it.
        // Tasks with no canonical primary (object detection, unknown) keep "auto": that deferred
        // case is intentional and PrimaryMetricOrAuto preserves it.
        // A classifier is the exception: its `auto` is a decision the search takes (MetricNames.SearchObjective — AUC /
        // log-loss), so it flows on. Every record and the quality gate canonicalize it to the primary metric
        // (TrainingConfig.CanonicalMetric), so nothing reads the sentinel as a metric.
        if (string.Equals(training.Metric, ConfigDefaults.DefaultMetric, StringComparison.OrdinalIgnoreCase)
            && MetricNames.SearchObjective(task, training.Metric) is null)
        {
            training.Metric = TaskMetadata.PrimaryMetricOrAuto(task);
        }

        return new ModelDefinition
        {
            Task = task,
            Label = label,
            Training = training,
            Description = baseDefinition?.Description,
            Prep = baseDefinition?.Prep,
            Columns = baseDefinition?.Columns,
            NumClusters = baseDefinition?.NumClusters,
            GroupColumn = baseDefinition?.GroupColumn,
            Weight = baseDefinition?.Weight,
            Encoder = baseDefinition?.Encoder,
            BankSize = baseDefinition?.BankSize,
            Portable = baseDefinition?.Portable,
            Ensemble = baseDefinition?.Ensemble,
            Gate = baseDefinition?.Gate,
            Horizon = baseDefinition?.Horizon,
            WindowSize = baseDefinition?.WindowSize,
            SeriesLength = baseDefinition?.SeriesLength,
            AnomalyThreshold = baseDefinition?.AnomalyThreshold,
            UserColumn = baseDefinition?.UserColumn,
            ItemColumn = baseDefinition?.ItemColumn
        };
    }

    /// <summary>
    /// Merges training settings with priority: cli > model > defaults
    /// </summary>
    public TrainingSettings MergeTrainingSettings(
        TrainingSettings? defaults,
        TrainingSettings? modelSettings,
        TrainingSettings? cliSettings)
    {
        var merged = new TrainingSettings
        {
            // Leave TimeLimitSeconds unset (null) so an unspecified time flows through as
            // "auto-time" rather than a fixed 300s — the project workflow then honors the
            // documented auto-time default. Downstream resolves null via `?? DefaultTimeLimitSeconds`
            // when auto-time is off. Metric/TestSplit keep concrete defaults (no auto semantics).
            TimeLimitSeconds = null,
            Metric = ConfigDefaults.DefaultMetric,
            TestSplit = ConfigDefaults.DefaultTestSplit
        };

        // Apply defaults
        if (defaults != null)
        {
            ApplyTrainingSettings(merged, defaults);
        }

        // Apply model-specific settings
        if (modelSettings != null)
        {
            ApplyTrainingSettings(merged, modelSettings);
        }

        // Apply CLI settings (highest priority)
        if (cliSettings != null)
        {
            ApplyTrainingSettings(merged, cliSettings);
        }

        return merged;
    }

    private void ApplyTrainingSettings(TrainingSettings target, TrainingSettings source)
    {
        if (source.TimeLimitSeconds.HasValue)
            target.TimeLimitSeconds = source.TimeLimitSeconds;

        if (source.MaxTrials.HasValue)
            target.MaxTrials = source.MaxTrials;

        if (source.Epochs.HasValue)
            target.Epochs = source.Epochs;

        if (!string.IsNullOrEmpty(source.Metric))
            target.Metric = source.Metric;

        if (source.TestSplit.HasValue)
            target.TestSplit = source.TestSplit;
    }
}
