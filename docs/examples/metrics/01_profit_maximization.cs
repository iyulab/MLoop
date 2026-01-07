using MLoop.Extensibility.Metrics;

/// <summary>
/// Profit Maximization Metric for Binary Classification
///
/// Business Context:
/// - True Positive: Customer accepts offer → Revenue earned
/// - False Positive: Customer rejects offer → Marketing cost wasted
/// - True Negative: Correctly identified non-buyer → Cost saved
/// - False Negative: Missed opportunity → Revenue lost
///
/// Formula: Expected Profit = (TP × Revenue) - (FP × Cost)
///
/// Example Use Case:
/// - Marketing campaign optimization
/// - Fraud detection with intervention costs
/// - Medical diagnosis with treatment costs
/// </summary>
public class ProfitMaximizationMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public string Description => "Business profit from ML predictions";

    // Configuration: Adjust these based on your business model
    private const double REVENUE_PER_TRUE_POSITIVE = 100.0;  // Revenue from successful prediction
    private const double COST_PER_FALSE_POSITIVE = 20.0;     // Cost of wrong positive prediction

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        // Validate task type
        if (ctx.TaskType != "BinaryClassification")
        {
            throw new InvalidOperationException(
                $"Profit metric requires binary classification, got: {ctx.TaskType}");
        }

        // Extract predictions and labels
        var predictions = (bool[])ctx.Predictions;
        var labels = (bool[])ctx.Labels;

        if (predictions.Length != labels.Length)
        {
            throw new InvalidOperationException(
                $"Prediction count ({predictions.Length}) != Label count ({labels.Length})");
        }

        // Calculate confusion matrix
        int truePositives = 0, falsePositives = 0;
        int trueNegatives = 0, falseNegatives = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] && labels[i])
                truePositives++;      // Correct positive prediction
            else if (predictions[i] && !labels[i])
                falsePositives++;     // Wrong positive prediction
            else if (!predictions[i] && !labels[i])
                trueNegatives++;      // Correct negative prediction
            else
                falseNegatives++;     // Wrong negative prediction (missed opportunity)
        }

        // Calculate expected profit
        double profit = (truePositives * REVENUE_PER_TRUE_POSITIVE)
                      - (falsePositives * COST_PER_FALSE_POSITIVE);

        // Calculate additional business metrics
        double totalRevenue = truePositives * REVENUE_PER_TRUE_POSITIVE;
        double totalCost = falsePositives * COST_PER_FALSE_POSITIVE;
        double missedRevenue = falseNegatives * REVENUE_PER_TRUE_POSITIVE;

        // Logging for transparency
        ctx.Logger.Info($"📊 Confusion Matrix:");
        ctx.Logger.Info($"   TP: {truePositives,4} | FN: {falseNegatives,4}");
        ctx.Logger.Info($"   FP: {falsePositives,4} | TN: {trueNegatives,4}");
        ctx.Logger.Info($"");
        ctx.Logger.Info($"💰 Financial Impact:");
        ctx.Logger.Info($"   Revenue:        ${totalRevenue,8:F2}");
        ctx.Logger.Info($"   Cost:          -${totalCost,8:F2}");
        ctx.Logger.Info($"   Missed Revenue: ${missedRevenue,8:F2}");
        ctx.Logger.Info($"   Net Profit:     ${profit,8:F2}");

        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = profit,
            Description = $"${profit:F2} profit ({truePositives} TPs × ${REVENUE_PER_TRUE_POSITIVE} - {falsePositives} FPs × ${COST_PER_FALSE_POSITIVE})",
            Details = new Dictionary<string, object>
            {
                ["TruePositives"] = truePositives,
                ["FalsePositives"] = falsePositives,
                ["TrueNegatives"] = trueNegatives,
                ["FalseNegatives"] = falseNegatives,
                ["TotalRevenue"] = totalRevenue,
                ["TotalCost"] = totalCost,
                ["MissedRevenue"] = missedRevenue,
                ["RevenuePerTP"] = REVENUE_PER_TRUE_POSITIVE,
                ["CostPerFP"] = COST_PER_FALSE_POSITIVE
            }
        });
    }
}
