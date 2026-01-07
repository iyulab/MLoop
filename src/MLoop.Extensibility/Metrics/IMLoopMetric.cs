namespace MLoop.Extensibility.Metrics;

/// <summary>
/// Interface for custom business metrics that calculate domain-specific values from ML predictions.
/// Metrics enable business impact analysis beyond standard ML evaluation metrics.
/// </summary>
/// <remarks>
/// <para>
/// Custom metrics allow expert users to translate ML performance into business outcomes:
/// - Financial metrics (profit, revenue, cost)
/// - Domain-specific metrics (churn cost, customer lifetime value)
/// - ROI and business impact calculations
/// </para>
/// <para>
/// Metrics are discovered in `.mloop/scripts/metrics/` and calculated after model evaluation.
/// Each metric receives predictions, ground truth labels, and model metadata to compute business value.
/// </para>
/// <para>
/// Example: Profit maximization metric (classification)
/// <code>
/// public class ProfitMetric : IMLoopMetric
/// {
///     public string Name => "Expected Profit";
///     public string Description => "Profit = (TP × Revenue) - (FP × Cost)";
///
///     private const double REVENUE_PER_TRUE_POSITIVE = 100.0;
///     private const double COST_PER_FALSE_POSITIVE = 20.0;
///
///     public Task&lt;MetricResult&gt; CalculateAsync(MetricContext ctx)
///     {
///         var predictions = ctx.Predictions;
///         var labels = ctx.Labels;
///
///         int truePositives = 0, falsePositives = 0;
///
///         for (int i = 0; i &lt; predictions.Length; i++)
///         {
///             if (predictions[i] &amp;&amp; labels[i]) truePositives++;
///             else if (predictions[i] &amp;&amp; !labels[i]) falsePositives++;
///         }
///
///         double profit = (truePositives * REVENUE_PER_TRUE_POSITIVE)
///                       - (falsePositives * COST_PER_FALSE_POSITIVE);
///
///         ctx.Logger.Info($"TP: {truePositives}, FP: {falsePositives}");
///         ctx.Logger.Info($"💰 Expected Profit: ${profit:F2}");
///
///         return Task.FromResult(new MetricResult
///         {
///             Name = Name,
///             Value = profit,
///             Description = $"Profit from {truePositives} TPs - {falsePositives} FPs"
///         });
///     }
/// }
/// </code>
/// </para>
/// <para>
/// Example: Churn cost minimization metric
/// <code>
/// public class ChurnCostMetric : IMLoopMetric
/// {
///     public string Name => "Churn Prevention Value";
///     public string Description => "Value = (TP × LTV) - (Intervention Cost)";
///
///     private const double CUSTOMER_LTV = 5000.0;
///     private const double INTERVENTION_COST = 50.0;
///
///     public Task&lt;MetricResult&gt; CalculateAsync(MetricContext ctx)
///     {
///         var predictions = ctx.Predictions;
///         var labels = ctx.Labels;
///
///         int prevented = 0;
///         for (int i = 0; i &lt; predictions.Length; i++)
///         {
///             if (predictions[i] &amp;&amp; labels[i]) prevented++;
///         }
///
///         double value = (prevented * CUSTOMER_LTV)
///                      - (predictions.Count(p => p) * INTERVENTION_COST);
///
///         return Task.FromResult(new MetricResult
///         {
///             Name = Name,
///             Value = value,
///             Description = $"Prevented {prevented} churns, value: ${value:F2}"
///         });
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public interface IMLoopMetric
{
    /// <summary>
    /// Human-readable metric name for reporting and display.
    /// Example: "Expected Profit", "Customer Lifetime Value", "ROI"
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Metric description explaining calculation formula or business meaning.
    /// Example: "Profit = (TP × $100) - (FP × $20)"
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Calculates custom business metric from model predictions and ground truth labels.
    /// </summary>
    /// <param name="context">
    /// Metric calculation context providing:
    /// - Model predictions (classification or regression)
    /// - Ground truth labels from test/evaluation dataset
    /// - Model metadata (task type, model name, experiment info)
    /// - Logger for progress and debugging
    /// </param>
    /// <returns>
    /// Metric result containing:
    /// - Name: Metric identifier
    /// - Value: Calculated business metric value
    /// - Description: Human-readable interpretation of the result
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when metric calculation fails (e.g., incompatible data types, missing context).
    /// Metrics should validate context data before calculation.
    /// </exception>
    Task<MetricResult> CalculateAsync(MetricContext context);
}
