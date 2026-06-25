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
    /// Returns a user-facing leakage warning message for a step classified as UnsupportedLeakageWarn.
    /// Both PrepRouter and ValidateCommand use this to ensure consistent wording (DRY).
    /// </summary>
    public static string LeakageWarning(PrepStep step) =>
        $"prep '{step.Type}'({step.Method}) 는 train/test 분할 전 전역 통계로 적용되어 " +
        "평가에 데이터 누수가 남습니다. 가능하면 normalize(min-max/z-score) 또는 " +
        "fill-missing(mean)으로 대체하세요.";

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
