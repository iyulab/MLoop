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
