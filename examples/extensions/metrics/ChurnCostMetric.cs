// A business metric: what the model's answers would cost a retention programme, per 1,000 customers.
// Put it in .mloop/scripts/metrics/ — it is computed after training and recorded with the run's metrics.
// It is a report, not an optimization target: the search still optimizes the project's `metric:`.
using System;
using System.Linq;
using System.Threading.Tasks;
using MLoop.Extensibility.Metrics;

/// <summary>
/// Churn prevention cost: a retention campaign costs money, a campaign keeps a customer only part of the time, and a churner nobody
/// called is lost. Lower is better.
/// </summary>
/// <remarks>
/// Cost per customer = campaigns run (every predicted churner, right or wrong) × campaign cost
///                   + churners lost (missed ones, plus the calls that failed to keep one) × the value of a customer.
/// </remarks>
public class ChurnCostMetric : IMLoopMetric
{
    public string Name => "Churn Prevention Cost";
    public string Description => "Cost per 1,000 customers = campaigns × 20 + lost churners × 500 (lower is better)";

    // 💰 Business parameters — adjust for your business model
    private const double CampaignCost = 20.0;          // one retention campaign (email, call, discount)
    private const double CustomerValue = 500.0;        // what a churned customer costs you
    private const double CampaignSuccessRate = 0.4;    // share of campaigns that keep a customer who would have left
    private const int CustomerBase = 1000;             // report per this many customers

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        if (ctx.Predictions is not bool[] predicted || ctx.Labels is not bool[] churned)
            throw new InvalidOperationException($"{Name} is for binary classification (task was {ctx.TaskType})");
        if (predicted.Length == 0)
            throw new InvalidOperationException("No held-out rows to score");

        double scale = (double)CustomerBase / predicted.Length;
        double campaigns = predicted.Count(p => p) * scale;                                       // calls made: TP + FP
        double truePositives = predicted.Zip(churned, (p, c) => p && c).Count(x => x) * scale;    // churners reached
        double missed = predicted.Zip(churned, (p, c) => !p && c).Count(x => x) * scale;          // churners nobody called
        double lost = missed + truePositives * (1 - CampaignSuccessRate);                          // reached, but the call did not keep them

        double cost = campaigns * CampaignCost + lost * CustomerValue;
        double doNothing = churned.Count(c => c) * scale * CustomerValue;                          // no programme at all
        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = cost,
            Description = $"{campaigns:F0} campaigns, {lost:F0} lost customers → {cost:F0} per {CustomerBase:N0} (doing nothing: {doNothing:F0})"
        });
    }
}
