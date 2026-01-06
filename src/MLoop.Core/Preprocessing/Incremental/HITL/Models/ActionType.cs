namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Defines the action to take based on HITL decision.
/// </summary>
public enum ActionType
{
    /// <summary>
    /// Delete the affected records.
    /// </summary>
    Delete,

    /// <summary>
    /// Keep records as-is without modification.
    /// </summary>
    KeepAsIs,

    /// <summary>
    /// Impute missing values with column mean.
    /// </summary>
    ImputeMean,

    /// <summary>
    /// Impute missing values with column median.
    /// </summary>
    ImputeMedian,

    /// <summary>
    /// Impute with mode (most frequent value).
    /// </summary>
    ImputeMode,

    /// <summary>
    /// Impute with a custom user-specified value.
    /// </summary>
    ImputeCustom,

    /// <summary>
    /// Remove outlier records completely.
    /// </summary>
    RemoveOutliers,

    /// <summary>
    /// Cap outliers at a specified threshold.
    /// </summary>
    CapOutliers,

    /// <summary>
    /// Flag outliers for manual review.
    /// </summary>
    FlagForReview,

    /// <summary>
    /// Merge similar categories into one.
    /// </summary>
    MergeCategories,

    /// <summary>
    /// Convert to a different data type.
    /// </summary>
    ConvertType,

    /// <summary>
    /// Apply custom business logic.
    /// </summary>
    CustomLogic
}
