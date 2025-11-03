namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Merges configuration from multiple sources with priority:
/// CLI Args > mloop.yaml > .mloop/config.json > Defaults
/// </summary>
public class ConfigMerger
{
    /// <summary>
    /// Merges configurations with priority
    /// </summary>
    public MLoopConfig Merge(
        MLoopConfig? cliConfig = null,
        MLoopConfig? userConfig = null,
        MLoopConfig? projectConfig = null,
        MLoopConfig? defaults = null)
    {
        var merged = new MLoopConfig
        {
            Training = new TrainingSettings(),
            Data = new DataSettings(),
            Model = new ModelSettings()
        };

        // Apply defaults
        if (defaults != null)
        {
            ApplyConfig(merged, defaults);
        }

        // Apply project config (.mloop/config.json)
        if (projectConfig != null)
        {
            ApplyConfig(merged, projectConfig);
        }

        // Apply user config (mloop.yaml)
        if (userConfig != null)
        {
            ApplyConfig(merged, userConfig);
        }

        // Apply CLI args (highest priority)
        if (cliConfig != null)
        {
            ApplyConfig(merged, cliConfig);
        }

        return merged;
    }

    /// <summary>
    /// Creates default configuration
    /// </summary>
    public static MLoopConfig CreateDefaults()
    {
        return new MLoopConfig
        {
            Training = new TrainingSettings
            {
                TimeLimitSeconds = 300,
                Metric = "accuracy",
                TestSplit = 0.2
            },
            Data = new DataSettings
            {
                Train = "data/processed/train.csv",
                Test = "data/processed/test.csv"
            },
            Model = new ModelSettings
            {
                OutputDir = "models/staging"
            }
        };
    }

    private void ApplyConfig(MLoopConfig target, MLoopConfig source)
    {
        // Apply top-level properties
        if (!string.IsNullOrEmpty(source.ProjectName))
            target.ProjectName = source.ProjectName;

        if (!string.IsNullOrEmpty(source.Task))
            target.Task = source.Task;

        if (!string.IsNullOrEmpty(source.LabelColumn))
            target.LabelColumn = source.LabelColumn;

        // Apply training settings
        if (source.Training != null && target.Training != null)
        {
            if (source.Training.TimeLimitSeconds > 0)
                target.Training.TimeLimitSeconds = source.Training.TimeLimitSeconds;

            if (!string.IsNullOrEmpty(source.Training.Metric))
                target.Training.Metric = source.Training.Metric;

            if (source.Training.TestSplit >= 0)
                target.Training.TestSplit = source.Training.TestSplit;
        }

        // Apply data settings
        if (source.Data != null && target.Data != null)
        {
            if (!string.IsNullOrEmpty(source.Data.Train))
                target.Data.Train = source.Data.Train;

            if (!string.IsNullOrEmpty(source.Data.Test))
                target.Data.Test = source.Data.Test;
        }

        // Apply model settings
        if (source.Model != null && target.Model != null)
        {
            if (!string.IsNullOrEmpty(source.Model.OutputDir))
                target.Model.OutputDir = source.Model.OutputDir;
        }
    }
}
