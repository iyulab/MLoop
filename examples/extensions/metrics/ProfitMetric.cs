// A business metric: the expected profit of the model's answers on the held-out rows.
// Put it in .mloop/scripts/metrics/ — it is computed after training and recorded with the run's metrics.
// It is a report, not an optimization target: the search still optimizes the project's `metric:`.
using System;
using System.Linq;
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public string Description => "Profit = TP × 100 − FP × 50 − FN × 30 (held-out rows)";

    // 💰 Business parameters — adjust for your domain
    private const double ProfitPerTruePositive = 100.0;  // revenue per successful prediction
    private const double LossPerFalsePositive = 50.0;    // cost per false alarm
    private const double LossPerFalseNegative = 30.0;    // opportunity cost per missed one

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        // Binary classification gives bool[] for both answers and truth.
        if (ctx.Predictions is not bool[] predicted || ctx.Labels is not bool[] actual)
            throw new InvalidOperationException($"{Name} is for binary classification (task was {ctx.TaskType})");

        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            if (predicted[i] && actual[i]) tp++;
            else if (predicted[i]) fp++;
            else if (actual[i]) fn++;
        }

        double profit = tp * ProfitPerTruePositive - fp * LossPerFalsePositive - fn * LossPerFalseNegative;
        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = profit,
            Description = $"{tp} true positives, {fp} false positives, {fn} false negatives → {profit:F0}"
        });
    }
}
