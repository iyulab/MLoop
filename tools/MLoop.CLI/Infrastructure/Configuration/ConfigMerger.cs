namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Merges configuration from multiple sources with priority:
/// CLI Args > mloop.yaml > .mloop/config.json > Defaults
/// </summary>
public class ConfigMerger
{
    /// <summary>
    /// Merges project-level configurations
    /// </summary>
    public MLoopConfig Merge(
        MLoopConfig? cliConfig = null,
        MLoopConfig? userConfig = null,
        MLoopConfig? projectConfig = null)
    {
        var merged = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>(),
            Data = new DataSettings()
        };

        // Apply project config (.mloop/config.json) - lowest priority
        if (projectConfig != null)
        {
            ApplyConfig(merged, projectConfig);
        }

        // Apply user config (mloop.yaml) - medium priority
        if (userConfig != null)
        {
            ApplyConfig(merged, userConfig);
        }

        // Apply CLI args - highest priority
        if (cliConfig != null)
        {
            ApplyConfig(merged, cliConfig);
        }

        return merged;
    }

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

        // Label is optional for unsupervised tasks
        var unsupervisedTasks = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { "anomaly-detection", "clustering", "time-series-anomaly" };
        var label = cliLabel
            ?? baseDefinition?.Label
            ?? (unsupervisedTasks.Contains(task) ? "" : throw new InvalidOperationException(
                $"Label not specified for model '{modelName}'. Use <label> argument or define in mloop.yaml"));

        var training = MergeTrainingSettings(
            ConfigDefaults.CreateDefaultTrainingSettings(),
            baseDefinition?.Training,
            cliTraining);

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
            Horizon = baseDefinition?.Horizon,
            WindowSize = baseDefinition?.WindowSize,
            SeriesLength = baseDefinition?.SeriesLength,
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
            TimeLimitSeconds = ConfigDefaults.DefaultTimeLimitSeconds,
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

    private void ApplyConfig(MLoopConfig target, MLoopConfig source)
    {
        // Apply project name
        if (!string.IsNullOrEmpty(source.Project))
        {
            target.Project = source.Project;
        }

        // Merge model definitions
        foreach (var (name, definition) in source.Models)
        {
            if (target.Models.TryGetValue(name, out var existing))
            {
                // Merge with existing definition
                target.Models[name] = MergeModelDefinitions(existing, definition);
            }
            else
            {
                // Add new definition
                target.Models[name] = definition;
            }
        }

        // Apply data settings
        if (source.Data != null)
        {
            target.Data ??= new DataSettings();

            if (!string.IsNullOrEmpty(source.Data.Train))
                target.Data.Train = source.Data.Train;

            if (!string.IsNullOrEmpty(source.Data.Test))
                target.Data.Test = source.Data.Test;

            if (!string.IsNullOrEmpty(source.Data.Predict))
                target.Data.Predict = source.Data.Predict;
        }
    }

    private ModelDefinition MergeModelDefinitions(ModelDefinition existing, ModelDefinition source)
    {
        return new ModelDefinition
        {
            Task = !string.IsNullOrEmpty(source.Task) ? source.Task : existing.Task,
            Label = !string.IsNullOrEmpty(source.Label) ? source.Label : existing.Label,
            Training = MergeTrainingSettings(existing.Training, source.Training, null),
            Description = !string.IsNullOrEmpty(source.Description) ? source.Description : existing.Description,
            Prep = source.Prep ?? existing.Prep,
            Columns = source.Columns ?? existing.Columns,
            NumClusters = source.NumClusters ?? existing.NumClusters,
            GroupColumn = !string.IsNullOrEmpty(source.GroupColumn) ? source.GroupColumn : existing.GroupColumn,
            Horizon = source.Horizon ?? existing.Horizon,
            WindowSize = source.WindowSize ?? existing.WindowSize,
            SeriesLength = source.SeriesLength ?? existing.SeriesLength,
            UserColumn = !string.IsNullOrEmpty(source.UserColumn) ? source.UserColumn : existing.UserColumn,
            ItemColumn = !string.IsNullOrEmpty(source.ItemColumn) ? source.ItemColumn : existing.ItemColumn
        };
    }

    private void ApplyTrainingSettings(TrainingSettings target, TrainingSettings source)
    {
        if (source.TimeLimitSeconds.HasValue)
            target.TimeLimitSeconds = source.TimeLimitSeconds;

        if (!string.IsNullOrEmpty(source.Metric))
            target.Metric = source.Metric;

        if (source.TestSplit.HasValue)
            target.TestSplit = source.TestSplit;
    }
}
