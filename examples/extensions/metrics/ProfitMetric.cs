using MLoop.Extensibility;
using Microsoft.ML;
using Microsoft.ML.Data;

/// <summary>
/// Custom metric that optimizes for expected profit instead of accuracy.
/// Useful for business scenarios where different prediction outcomes
/// have different financial impacts.
/// </summary>
/// <example>
/// Marketing Campaign:
/// - True Positive: Customer responds → $100 profit
/// - False Positive: Wasted campaign cost → -$50
/// - False Negative: Missed opportunity → -$30
/// </example>
public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public bool HigherIsBetter => true;  // Maximize profit

    // 💰 Business Parameters - ADJUST FOR YOUR DOMAIN
    private const double PROFIT_PER_TP = 100.0;   // Revenue per successful prediction
    private const double LOSS_PER_FP = -50.0;     // Cost per false alarm
    private const double LOSS_PER_FN = -30.0;     // Opportunity cost per missed prediction

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        try
        {
            // Evaluate model using ML.NET metrics
            var metrics = ctx.MLContext.BinaryClassification.Evaluate(
                ctx.Predictions,
                labelColumnName: ctx.LabelColumn,
                scoreColumnName: ctx.ScoreColumn
            );

            // Calculate expected profit per prediction
            // Formula: (TP% * Profit) + (FP% * Loss) + (FN% * Loss)
            var expectedProfit =
                (metrics.PositiveRecall * PROFIT_PER_TP) +      // True Positives
                (metrics.FalsePositiveRate * LOSS_PER_FP) +     // False Positives
                ((1 - metrics.PositiveRecall) * LOSS_PER_FN);   // False Negatives

            // Log detailed breakdown for transparency
            ctx.Logger.Info($"📊 Profit Analysis:");
            ctx.Logger.Info($"   Positive Recall (TPR): {metrics.PositiveRecall:F3}");
            ctx.Logger.Info($"   False Positive Rate: {metrics.FalsePositiveRate:F3}");
            ctx.Logger.Info($"   False Negative Rate: {1 - metrics.PositiveRecall:F3}");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"💰 Expected Profit: ${expectedProfit:F2} per prediction");
            ctx.Logger.Info($"   ✅ TP contribution: ${metrics.PositiveRecall * PROFIT_PER_TP:F2}");
            ctx.Logger.Info($"   ❌ FP cost: ${metrics.FalsePositiveRate * LOSS_PER_FP:F2}");
            ctx.Logger.Info($"   ⚠️  FN cost: ${(1 - metrics.PositiveRecall) * LOSS_PER_FN:F2}");

            // For 1000 predictions, estimate total profit
            var estimatedTotal = expectedProfit * 1000;
            ctx.Logger.Info($"");
            ctx.Logger.Info($"📈 Estimated for 1000 predictions: ${estimatedTotal:F2}");

            return expectedProfit;
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"❌ Profit metric calculation failed: {ex.Message}");
            // Return 0 profit on error (worst case scenario)
            return 0.0;
        }
    }
}

/// <summary>
/// Alternative: Weighted Profit Metric with configurable parameters
/// </summary>
public class ConfigurableProfitMetric : IMLoopMetric
{
    public string Name => "Weighted Profit Metric";
    public bool HigherIsBetter => true;

    // Allow runtime configuration via constructor or properties
    public double ProfitPerTP { get; set; } = 100.0;
    public double LossPerFP { get; set; } = -50.0;
    public double LossPerFN { get; set; } = -30.0;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(
            ctx.Predictions,
            labelColumnName: ctx.LabelColumn,
            scoreColumnName: ctx.ScoreColumn
        );

        var expectedProfit =
            (metrics.PositiveRecall * ProfitPerTP) +
            (metrics.FalsePositiveRate * LossPerFP) +
            ((1 - metrics.PositiveRecall) * LossPerFN);

        ctx.Logger.Info($"💰 Expected Profit: ${expectedProfit:F2} per prediction");
        ctx.Logger.Info($"   (TP: ${ProfitPerTP}, FP: ${LossPerFP}, FN: ${LossPerFN})");

        return expectedProfit;
    }
}
