using MLoop.Extensibility.Metrics;

/// <summary>
/// Churn Prevention Value Metric for Binary Classification
///
/// Business Context:
/// - Predicting which customers will churn (cancel subscription)
/// - True Positive: Correctly identified churner → Retention campaign succeeds
/// - False Positive: Non-churner gets campaign → Intervention cost wasted
/// - False Negative: Missed churner → Customer lifetime value lost
///
/// Formula: Value = (TP × Prevented LTV) - (All Interventions × Cost)
///
/// Example Use Case:
/// - SaaS customer retention
/// - Telecom churn prediction
/// - Subscription service optimization
/// </summary>
public class ChurnPreventionValueMetric : IMLoopMetric
{
    public string Name => "Churn Prevention Value";
    public string Description => "Business value from preventing customer churn";

    // Configuration: Adjust based on your business model
    private const double CUSTOMER_LIFETIME_VALUE = 5000.0;    // Average LTV of a customer
    private const double INTERVENTION_COST = 50.0;            // Cost of retention campaign per customer
    private const double INTERVENTION_SUCCESS_RATE = 0.70;    // 70% of interventions prevent churn

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        // Validate task type
        if (ctx.TaskType != "BinaryClassification")
        {
            throw new InvalidOperationException(
                $"Churn metric requires binary classification, got: {ctx.TaskType}");
        }

        // Extract predictions and labels
        var predictions = (bool[])ctx.Predictions;  // True = will churn
        var labels = (bool[])ctx.Labels;            // True = actually churned

        if (predictions.Length != labels.Length)
        {
            throw new InvalidOperationException(
                $"Prediction count ({predictions.Length}) != Label count ({labels.Length})");
        }

        // Calculate metrics
        int truePositives = 0;   // Correctly identified churners
        int falsePositives = 0;  // Non-churners flagged as churners
        int falseNegatives = 0;  // Churners we missed

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] && labels[i])
                truePositives++;
            else if (predictions[i] && !labels[i])
                falsePositives++;
            else if (!predictions[i] && labels[i])
                falseNegatives++;
        }

        // Calculate business value
        int totalInterventions = truePositives + falsePositives;
        double preventedChurns = truePositives * INTERVENTION_SUCCESS_RATE;
        double preventedLTV = preventedChurns * CUSTOMER_LIFETIME_VALUE;
        double interventionCosts = totalInterventions * INTERVENTION_COST;
        double value = preventedLTV - interventionCosts;

        // Calculate opportunity cost
        double missedChurns = falseNegatives;
        double missedLTV = missedChurns * CUSTOMER_LIFETIME_VALUE;

        // Calculate ROI
        double roi = interventionCosts > 0
            ? ((preventedLTV - interventionCosts) / interventionCosts) * 100
            : 0;

        // Logging
        ctx.Logger.Info($"👥 Churn Prevention Analysis:");
        ctx.Logger.Info($"   Identified Churners:     {truePositives,4}");
        ctx.Logger.Info($"   False Alarms:            {falsePositives,4}");
        ctx.Logger.Info($"   Missed Churners:         {falseNegatives,4}");
        ctx.Logger.Info($"");
        ctx.Logger.Info($"💵 Financial Impact:");
        ctx.Logger.Info($"   Interventions:           {totalInterventions,4} × ${INTERVENTION_COST} = ${interventionCosts,8:F2}");
        ctx.Logger.Info($"   Prevented Churns:        {preventedChurns,6:F1} × ${CUSTOMER_LIFETIME_VALUE} = ${preventedLTV,8:F2}");
        ctx.Logger.Info($"   Missed LTV:             ${missedLTV,8:F2}");
        ctx.Logger.Info($"   Net Value:              ${value,8:F2}");
        ctx.Logger.Info($"   ROI:                     {roi,7:F1}%");

        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = value,
            Description = $"${value:F2} value (prevented {preventedChurns:F0} churns, ROI: {roi:F1}%)",
            Details = new Dictionary<string, object>
            {
                ["TruePositives"] = truePositives,
                ["FalsePositives"] = falsePositives,
                ["FalseNegatives"] = falseNegatives,
                ["TotalInterventions"] = totalInterventions,
                ["PreventedChurns"] = preventedChurns,
                ["PreventedLTV"] = preventedLTV,
                ["InterventionCosts"] = interventionCosts,
                ["MissedLTV"] = missedLTV,
                ["ROI"] = roi,
                ["CustomerLTV"] = CUSTOMER_LIFETIME_VALUE,
                ["InterventionCost"] = INTERVENTION_COST,
                ["InterventionSuccessRate"] = INTERVENTION_SUCCESS_RATE
            }
        });
    }
}
