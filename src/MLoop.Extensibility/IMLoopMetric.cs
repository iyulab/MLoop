namespace MLoop.Extensibility;

/// <summary>
/// Defines a custom evaluation metric for AutoML optimization.
/// Allows business-specific optimization objectives instead of generic accuracy metrics.
/// </summary>
/// <remarks>
/// <para>
/// Custom metrics enable AutoML to optimize for business outcomes:
/// </para>
/// <list type="bullet">
/// <item><description>Expected Profit (revenue vs. cost)</description></item>
/// <item><description>Customer Retention Cost</description></item>
/// <item><description>ROI (Return on Investment)</description></item>
/// <item><description>Business-specific KPIs</description></item>
/// </list>
/// <para>
/// Metrics are discovered automatically from .mloop/scripts/metrics/ directory.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// public class ProfitMetric : IMLoopMetric
/// {
///     public string Name => "Expected Profit";
///     public bool HigherIsBetter => true;
///
///     private const double PROFIT_PER_TP = 100.0;
///     private const double LOSS_PER_FP = -50.0;
///
///     public async Task&lt;double&gt; CalculateAsync(MetricContext ctx)
///     {
///         var metrics = ctx.MLContext.BinaryClassification.Evaluate(ctx.Predictions);
///
///         return (metrics.PositiveRecall * PROFIT_PER_TP) +
///                (metrics.FalsePositiveRate * LOSS_PER_FP);
///     }
/// }
/// </code>
/// </example>
public interface IMLoopMetric
{
    /// <summary>
    /// Gets the display name of this metric.
    /// Used for logging and user feedback during AutoML training.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the optimization direction for this metric.
    /// </summary>
    /// <value>
    /// <c>true</c> if higher values are better (e.g., profit, revenue);
    /// <c>false</c> if lower values are better (e.g., cost, loss).
    /// </value>
    bool HigherIsBetter { get; }

    /// <summary>
    /// Calculates the metric value for a given set of predictions.
    /// This value is used by AutoML to select the best model.
    /// </summary>
    /// <param name="context">
    /// The evaluation context providing access to predictions, ML context, and logger.
    /// </param>
    /// <returns>
    /// The calculated metric value. AutoML will optimize to maximize or minimize
    /// this value based on <see cref="HigherIsBetter"/>.
    /// </returns>
    /// <exception cref="Exception">
    /// Exceptions thrown during metric calculation are caught and logged.
    /// AutoML falls back to default metrics to ensure training continues.
    /// </exception>
    Task<double> CalculateAsync(MetricContext context);
}
