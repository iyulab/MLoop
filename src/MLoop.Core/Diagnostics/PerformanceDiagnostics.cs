namespace MLoop.Core.Diagnostics;

/// <summary>
/// Analyzes model performance and provides diagnostic information.
/// Part of T4.4 - Low Performance Diagnostics feature.
/// </summary>
public class PerformanceDiagnostics
{
    /// <summary>
    /// Analyzes model performance metrics and returns diagnostic results.
    /// </summary>
    /// <param name="taskType">ML task type (regression, binaryclassification, etc.)</param>
    /// <param name="metrics">Dictionary of metric name to value</param>
    /// <param name="featureCount">Number of features used</param>
    /// <param name="rowCount">Number of training rows</param>
    /// <returns>Performance diagnostic result</returns>
    public PerformanceDiagnosticResult Analyze(
        string taskType,
        Dictionary<string, double> metrics,
        int featureCount = 0,
        int rowCount = 0)
    {
        var result = new PerformanceDiagnosticResult
        {
            TaskType = taskType,
            Metrics = metrics,
            FeatureCount = featureCount,
            RowCount = rowCount
        };

        // Analyze based on task type
        var normalizedTask = taskType.ToLowerInvariant().Replace(" ", "");

        switch (normalizedTask)
        {
            case "regression":
                AnalyzeRegression(result);
                break;
            case "binaryclassification":
                AnalyzeBinaryClassification(result);
                break;
            case "multiclassclassification":
                AnalyzeMulticlassClassification(result);
                break;
            default:
                result.OverallAssessment = PerformanceLevel.Unknown;
                result.Summary = $"Unknown task type: {taskType}";
                break;
        }

        // Add general suggestions based on data characteristics
        AddDataBasedSuggestions(result);

        return result;
    }

    private void AnalyzeRegression(PerformanceDiagnosticResult result)
    {
        var metrics = result.Metrics;

        // Check R-Squared (primary metric)
        if (metrics.TryGetValue("RSquared", out var rSquared) ||
            metrics.TryGetValue("R2", out rSquared) ||
            metrics.TryGetValue("r_squared", out rSquared))
        {
            result.PrimaryMetric = "R²";
            result.PrimaryMetricValue = rSquared;

            if (rSquared >= 0.9)
            {
                result.OverallAssessment = PerformanceLevel.Excellent;
                result.Summary = $"Excellent model fit (R² = {rSquared:F4})";
            }
            else if (rSquared >= 0.7)
            {
                result.OverallAssessment = PerformanceLevel.Good;
                result.Summary = $"Good model fit (R² = {rSquared:F4})";
            }
            else if (rSquared >= 0.5)
            {
                result.OverallAssessment = PerformanceLevel.Moderate;
                result.Summary = $"Moderate model fit (R² = {rSquared:F4})";
                result.Suggestions.Add("Consider adding more relevant features");
                result.Suggestions.Add("Try feature engineering or polynomial features");
            }
            else if (rSquared >= 0.3)
            {
                result.OverallAssessment = PerformanceLevel.Low;
                result.Summary = $"Low model fit (R² = {rSquared:F4})";
                result.Warnings.Add("Model explains less than 50% of variance");
                result.Suggestions.Add("Review feature selection - current features may not be predictive");
                result.Suggestions.Add("Check for data quality issues (outliers, missing values)");
                result.Suggestions.Add("Consider collecting additional training data");
            }
            else
            {
                result.OverallAssessment = PerformanceLevel.Poor;
                result.Summary = $"Poor model fit (R² = {rSquared:F4})";
                result.Warnings.Add("Model performance is near random");
                result.Suggestions.Add("Verify label column is correct");
                result.Suggestions.Add("Check if the problem is actually predictable with available features");
                result.Suggestions.Add("Consider if the target variable has high natural variance");
            }
        }

        // Check RMSE/MAE for additional context
        if (metrics.TryGetValue("RootMeanSquaredError", out var rmse) ||
            metrics.TryGetValue("RMSE", out rmse))
        {
            result.SecondaryMetrics["RMSE"] = rmse;
        }

        if (metrics.TryGetValue("MeanAbsoluteError", out var mae) ||
            metrics.TryGetValue("MAE", out mae))
        {
            result.SecondaryMetrics["MAE"] = mae;
        }
    }

    private void AnalyzeBinaryClassification(PerformanceDiagnosticResult result)
    {
        var metrics = result.Metrics;

        // Check AUC (primary metric for binary classification)
        if (metrics.TryGetValue("AreaUnderRocCurve", out var auc) ||
            metrics.TryGetValue("AUC", out auc) ||
            metrics.TryGetValue("auc", out auc))
        {
            result.PrimaryMetric = "AUC";
            result.PrimaryMetricValue = auc;

            if (auc >= 0.95)
            {
                result.OverallAssessment = PerformanceLevel.Excellent;
                result.Summary = $"Excellent classification (AUC = {auc:F4})";
            }
            else if (auc >= 0.85)
            {
                result.OverallAssessment = PerformanceLevel.Good;
                result.Summary = $"Good classification (AUC = {auc:F4})";
            }
            else if (auc >= 0.75)
            {
                result.OverallAssessment = PerformanceLevel.Moderate;
                result.Summary = $"Moderate classification (AUC = {auc:F4})";
                result.Suggestions.Add("Consider feature engineering to improve discrimination");
            }
            else if (auc >= 0.6)
            {
                result.OverallAssessment = PerformanceLevel.Low;
                result.Summary = $"Low classification performance (AUC = {auc:F4})";
                result.Warnings.Add("Model has weak discriminative ability");
                result.Suggestions.Add("Check for class imbalance and consider resampling");
                result.Suggestions.Add("Review feature relevance to the classification task");
            }
            else
            {
                result.OverallAssessment = PerformanceLevel.Poor;
                result.Summary = $"Poor classification (AUC = {auc:F4})";
                result.Warnings.Add("Model performance is near random (AUC ≈ 0.5)");
                result.Suggestions.Add("Verify positive/negative class labels are correct");
                result.Suggestions.Add("Check if classes are separable with current features");
            }
        }

        // Also check accuracy for context
        if (metrics.TryGetValue("Accuracy", out var accuracy) ||
            metrics.TryGetValue("accuracy", out accuracy))
        {
            result.SecondaryMetrics["Accuracy"] = accuracy;

            // Warn if accuracy is misleadingly high (possible class imbalance)
            if (accuracy > 0.9 && result.OverallAssessment <= PerformanceLevel.Moderate)
            {
                result.Warnings.Add($"High accuracy ({accuracy:P1}) but low AUC suggests class imbalance");
            }
        }

        // Check F1 Score
        if (metrics.TryGetValue("F1Score", out var f1) ||
            metrics.TryGetValue("F1", out f1))
        {
            result.SecondaryMetrics["F1"] = f1;
        }
    }

    private void AnalyzeMulticlassClassification(PerformanceDiagnosticResult result)
    {
        var metrics = result.Metrics;

        // Check MacroAccuracy or MicroAccuracy
        double? primaryMetricValue = null;
        string primaryMetricName = "";

        if (metrics.TryGetValue("MacroAccuracy", out var macroAcc))
        {
            primaryMetricValue = macroAcc;
            primaryMetricName = "Macro Accuracy";
        }
        else if (metrics.TryGetValue("MicroAccuracy", out var microAcc))
        {
            primaryMetricValue = microAcc;
            primaryMetricName = "Micro Accuracy";
        }
        else if (metrics.TryGetValue("Accuracy", out var acc))
        {
            primaryMetricValue = acc;
            primaryMetricName = "Accuracy";
        }

        if (primaryMetricValue.HasValue)
        {
            result.PrimaryMetric = primaryMetricName;
            result.PrimaryMetricValue = primaryMetricValue.Value;

            if (primaryMetricValue >= 0.9)
            {
                result.OverallAssessment = PerformanceLevel.Excellent;
                result.Summary = $"Excellent multiclass classification ({primaryMetricName} = {primaryMetricValue:F4})";
            }
            else if (primaryMetricValue >= 0.75)
            {
                result.OverallAssessment = PerformanceLevel.Good;
                result.Summary = $"Good multiclass classification ({primaryMetricName} = {primaryMetricValue:F4})";
            }
            else if (primaryMetricValue >= 0.6)
            {
                result.OverallAssessment = PerformanceLevel.Moderate;
                result.Summary = $"Moderate multiclass classification ({primaryMetricName} = {primaryMetricValue:F4})";
                result.Suggestions.Add("Consider if some classes are too similar");
                result.Suggestions.Add("Review per-class performance for weak classes");
            }
            else if (primaryMetricValue >= 0.4)
            {
                result.OverallAssessment = PerformanceLevel.Low;
                result.Summary = $"Low multiclass classification ({primaryMetricName} = {primaryMetricValue:F4})";
                result.Warnings.Add("Classification accuracy is below acceptable threshold");
                result.Suggestions.Add("Check for severe class imbalance");
                result.Suggestions.Add("Consider hierarchical classification or class grouping");
            }
            else
            {
                result.OverallAssessment = PerformanceLevel.Poor;
                result.Summary = $"Poor multiclass classification ({primaryMetricName} = {primaryMetricValue:F4})";
                result.Warnings.Add("Model performance is near random");
                result.Suggestions.Add("Verify class labels and data quality");
            }
        }

        // Check Log Loss for additional context
        if (metrics.TryGetValue("LogLoss", out var logLoss))
        {
            result.SecondaryMetrics["LogLoss"] = logLoss;
        }
    }

    private void AddDataBasedSuggestions(PerformanceDiagnosticResult result)
    {
        // Suggestions based on data characteristics
        if (result.RowCount > 0 && result.FeatureCount > 0)
        {
            var ratio = (double)result.RowCount / result.FeatureCount;

            if (ratio < 10)
            {
                result.Warnings.Add($"Low samples-to-features ratio ({ratio:F1}:1) - risk of overfitting");
                result.Suggestions.Add("Consider reducing features or collecting more data");
            }

            if (result.RowCount < 100)
            {
                result.Warnings.Add($"Small dataset ({result.RowCount} rows) - results may not generalize");
            }

            if (result.FeatureCount > 50)
            {
                result.Suggestions.Add($"High feature count ({result.FeatureCount}) - consider feature selection");
            }
        }
    }
}

/// <summary>
/// Result of performance diagnostics analysis.
/// </summary>
public class PerformanceDiagnosticResult
{
    /// <summary>
    /// ML task type analyzed.
    /// </summary>
    public string TaskType { get; set; } = "";

    /// <summary>
    /// Original metrics provided for analysis.
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Number of features in the model.
    /// </summary>
    public int FeatureCount { get; set; }

    /// <summary>
    /// Number of training rows.
    /// </summary>
    public int RowCount { get; set; }

    /// <summary>
    /// Overall performance assessment.
    /// </summary>
    public PerformanceLevel OverallAssessment { get; set; } = PerformanceLevel.Unknown;

    /// <summary>
    /// Primary metric name used for assessment.
    /// </summary>
    public string PrimaryMetric { get; set; } = "";

    /// <summary>
    /// Primary metric value.
    /// </summary>
    public double PrimaryMetricValue { get; set; }

    /// <summary>
    /// Additional metrics for context.
    /// </summary>
    public Dictionary<string, double> SecondaryMetrics { get; set; } = new();

    /// <summary>
    /// Human-readable summary of performance.
    /// </summary>
    public string Summary { get; set; } = "";

    /// <summary>
    /// Warning messages about potential issues.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Actionable suggestions for improvement.
    /// </summary>
    public List<string> Suggestions { get; set; } = new();

    /// <summary>
    /// Whether the performance needs attention.
    /// </summary>
    public bool NeedsAttention => OverallAssessment <= PerformanceLevel.Low;
}

/// <summary>
/// Performance level classification.
/// </summary>
public enum PerformanceLevel
{
    Unknown = 0,
    Poor = 1,
    Low = 2,
    Moderate = 3,
    Good = 4,
    Excellent = 5
}
