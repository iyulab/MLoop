using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Generates smart recommendations for HITL decisions.
/// </summary>
internal sealed class RecommendationEngine
{
    /// <summary>
    /// Gets the recommended option key for a rule.
    /// </summary>
    public string GetRecommendedOption(
        PreprocessingRule rule,
        DataFrame sample)
    {
        return rule.Type switch
        {
            PreprocessingRuleType.MissingValueStrategy => GetMissingValueRecommendation(rule, sample),
            PreprocessingRuleType.OutlierHandling => GetOutlierRecommendation(rule, sample),
            PreprocessingRuleType.CategoryMapping => "A", // Merge categories
            PreprocessingRuleType.TypeConversion => "A", // Convert to most common type
            PreprocessingRuleType.BusinessLogicDecision => "C", // Keep as-is (safest)
            _ => "A" // Default to first option
        };
    }

    /// <summary>
    /// Gets the rationale for the recommendation.
    /// </summary>
    public string GetRecommendationReason(
        PreprocessingRule rule,
        DataFrame sample,
        string recommendedOption)
    {
        return rule.Type switch
        {
            PreprocessingRuleType.MissingValueStrategy =>
                GetMissingValueRationale(rule, sample, recommendedOption),
            PreprocessingRuleType.OutlierHandling =>
                GetOutlierRationale(rule, sample, recommendedOption),
            PreprocessingRuleType.CategoryMapping =>
                "Merging category variations improves data consistency and reduces dimensionality",
            PreprocessingRuleType.TypeConversion =>
                "Converting to the most common type preserves the majority of data",
            PreprocessingRuleType.BusinessLogicDecision =>
                "Keeping as-is is the safest option until business logic is clarified",
            _ => "This is the recommended option based on statistical best practices"
        };
    }

    private string GetMissingValueRecommendation(PreprocessingRule rule, DataFrame sample)
    {
        var missingPercentage = (double)rule.AffectedRows / sample.Rows.Count;

        // If < 5% missing, recommend deletion (Option A)
        if (missingPercentage < 0.05)
            return "A";

        // If numeric column, recommend mean imputation (Option B)
        var column = sample.Columns[rule.ColumnNames[0]];
        if (column is PrimitiveDataFrameColumn<double> || column is PrimitiveDataFrameColumn<int>)
            return "B";

        // For categorical, recommend mode imputation (Option C)
        return "C";
    }

    private string GetMissingValueRationale(
        PreprocessingRule rule,
        DataFrame sample,
        string option)
    {
        var missingPercentage = (double)rule.AffectedRows / sample.Rows.Count;

        return option switch
        {
            "A" => $"Only {missingPercentage:P1} of data is affected. Deletion minimizes impact on analysis.",
            "B" => "Mean imputation preserves distribution and is statistically sound for numeric data with few outliers.",
            "C" => "Median imputation is robust to outliers and preserves central tendency.",
            _ => "This approach balances data preservation with statistical validity."
        };
    }

    private string GetOutlierRecommendation(PreprocessingRule rule, DataFrame sample)
    {
        var outlierPercentage = (double)rule.AffectedRows / sample.Rows.Count;

        // If < 1% outliers, might be data errors - recommend removal (Option B)
        if (outlierPercentage < 0.01)
            return "B";

        // If 1-5%, could be legitimate edge cases - recommend keeping (Option A)
        if (outlierPercentage < 0.05)
            return "A";

        // If > 5%, recommend capping (Option C)
        return "C";
    }

    private string GetOutlierRationale(
        PreprocessingRule rule,
        DataFrame sample,
        string option)
    {
        var outlierPercentage = (double)rule.AffectedRows / sample.Rows.Count;

        return option switch
        {
            "A" => outlierPercentage < 0.05
                ? "Small percentage of outliers may represent legitimate edge cases (e.g., executives, special events)."
                : "Keeping outliers preserves data completeness and may reveal important patterns.",
            "B" => outlierPercentage < 0.01
                ? "Very few outliers suggest data entry errors rather than legitimate values."
                : "Removing outliers improves model stability and reduces impact of extreme values.",
            "C" => "Capping outliers preserves all records while limiting the impact of extreme values.",
            _ => "This approach balances outlier impact with data preservation."
        };
    }
}
