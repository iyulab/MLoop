namespace MLoop.Core.Preprocessing;

/// <summary>
/// Categorizes preprocessing steps to manage data leakage during training.
/// </summary>
public enum PrepCategory
{
    /// <summary>
    /// Statistical fit transformations that can be moved to ML.NET preFeaturizer.
    /// Applied within fold boundaries to prevent data leakage.
    /// Examples: normalize (min-max, z-score), fill-missing (mean).
    /// </summary>
    PreFeaturizer,

    /// <summary>
    /// Row-independent or constant transformations with no statistical fit.
    /// Safe to keep in CSV preprocessing stage.
    /// Examples: remove-columns, extract-date, fill-missing (constant).
    /// </summary>
    CsvStage,

    /// <summary>
    /// Statistical transformations that cannot be reliably mapped to ML.NET.
    /// Contains potential data leakage; retained in CSV with warning.
    /// Future work: resolve via fold-aware scheduling or approximation.
    /// Examples: fill-missing (median), rolling, resample.
    /// </summary>
    UnsupportedLeakageWarn
}

/// <summary>
/// Classifies preparation steps by leakage risk for scheduling across CSV and ML.NET pipelines.
/// </summary>
public static class PrepStepClassifier
{
    /// <summary>
    /// Returns a user-facing leakage warning message for a step classified as UnsupportedLeakageWarn
    /// (median fill, rolling, resample). Both PrepRouter and ValidateCommand use this to ensure
    /// consistent wording (DRY). The advice to switch to normalize/fill-mean remains valid here.
    /// </summary>
    public static string LeakageWarning(PrepStep step) =>
        $"prep '{step.Type}'({step.Method}) is applied with statistics of the whole file, before the train/test " +
        "split, so the evaluation keeps a leak. Where possible use normalize (min-max/z-score) or " +
        "fill-missing (mean), which are fitted on the training rows only.";

    /// <summary>
    /// Returns a task-aware leakage warning for a step that IS a PreFeaturizer-category transform
    /// (normalize/scale/fill-mean) but cannot be routed to the preFeaturizer because the current
    /// task type does not support it (e.g. clustering, anomaly, ranking, recommendation).
    /// The transform is kept in the CSV stage (applied globally before train/test split), so
    /// leakage remains — but telling the user to "replace normalize with normalize" would be
    /// self-contradictory. This message explains WHY without giving misleading advice.
    /// </summary>
    public static string UnsupportedTaskLeakageWarning(PrepStep step) =>
        $"prep '{step.Type}'({step.Method}) cannot be fitted inside training for this task, so it is applied to the " +
        "whole file before the train/test split — the evaluation keeps a leak.";

    /// <summary>
    /// Classifies a preparation step into PreFeaturizer, CsvStage, or UnsupportedLeakageWarn category.
    /// </summary>
    public static PrepCategory Classify(PrepStep step)
    {
        var type = step.Type.ToLowerInvariant().Replace('_', '-');
        var method = step.Method?.ToLowerInvariant().Replace('_', '-');

        switch (type)
        {
            case "normalize":
            case "scale":
                // min-max / z-score / robust — all have ML.NET equivalents
                return PrepCategory.PreFeaturizer;

            case "fill-missing":
                return method switch
                {
                    "mean" => PrepCategory.PreFeaturizer,
                    "median" => PrepCategory.UnsupportedLeakageWarn,
                    // constant/forward/backward/mode: either no statistical fit or ML.NET unsupported → keep in CSV
                    _ => PrepCategory.CsvStage
                };

            case "rolling":
            case "resample":
                return PrepCategory.UnsupportedLeakageWarn;

            default:
                return PrepCategory.CsvStage;
        }
    }
}
