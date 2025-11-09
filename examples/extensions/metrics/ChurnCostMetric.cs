using MLoop.Extensibility;
using Microsoft.ML;
using Microsoft.ML.Data;

/// <summary>
/// Churn prevention cost optimization metric.
/// Minimizes the total cost of customer retention campaigns
/// by balancing campaign costs vs. lost customer value.
/// </summary>
/// <remarks>
/// Business Model:
/// - Running retention campaign on predicted churners costs money
/// - Successfully retained customers preserve their lifetime value
/// - Missed churners result in lost revenue
/// - False alarms waste campaign budget
///
/// Use Case: SaaS, Telecom, Subscription services
/// </remarks>
public class ChurnCostMetric : IMLoopMetric
{
    public string Name => "Churn Prevention Cost";
    public bool HigherIsBetter => false;  // Minimize cost

    // 💰 Business Parameters - ADJUST FOR YOUR BUSINESS MODEL
    private const double CAMPAIGN_COST = 20.0;       // Cost per retention campaign (email, call, discount)
    private const double CHURN_LOSS = 500.0;         // Customer lifetime value (CLV)
    private const double CAMPAIGN_SUCCESS_RATE = 0.4; // 40% of campaigns successfully retain customers

    // Simulation parameters
    private const int CUSTOMER_BASE = 1000;  // Simulate on 1000 customers

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        try
        {
            // Get model performance metrics
            var metrics = ctx.MLContext.BinaryClassification.Evaluate(
                ctx.Predictions,
                labelColumnName: ctx.LabelColumn,
                scoreColumnName: ctx.ScoreColumn
            );

            // Calculate customer segments (per 1000 customers)
            var truePositives = metrics.PositiveRecall * CUSTOMER_BASE;        // Correctly identified churners
            var falsePositives = metrics.FalsePositiveRate * CUSTOMER_BASE;    // False alarms
            var falseNegatives = (1 - metrics.PositiveRecall) * CUSTOMER_BASE; // Missed churners
            var trueNegatives = (1 - metrics.FalsePositiveRate) * CUSTOMER_BASE; // Correctly identified non-churners

            // Cost calculations
            // 1. Campaign costs: Run campaigns on all predicted churners (TP + FP)
            var totalCampaignTargets = truePositives + falsePositives;
            var campaignCost = totalCampaignTargets * CAMPAIGN_COST;

            // 2. Prevented churns: TP * success rate
            var preventedChurns = truePositives * CAMPAIGN_SUCCESS_RATE;
            var remainingTPChurns = truePositives * (1 - CAMPAIGN_SUCCESS_RATE);

            // 3. Total churn losses: FN + remaining TP churns
            var totalChurnedCustomers = falseNegatives + remainingTPChurns;
            var churnLosses = totalChurnedCustomers * CHURN_LOSS;

            // 4. Total cost
            var totalCost = campaignCost + churnLosses;
            var costPerCustomer = totalCost / CUSTOMER_BASE;

            // Log detailed breakdown
            ctx.Logger.Info($"📊 Churn Prevention Analysis (per {CUSTOMER_BASE} customers):");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"🎯 Prediction Breakdown:");
            ctx.Logger.Info($"   True Positives:  {truePositives:F0} (correctly identified churners)");
            ctx.Logger.Info($"   False Positives: {falsePositives:F0} (false alarms)");
            ctx.Logger.Info($"   False Negatives: {falseNegatives:F0} (missed churners)");
            ctx.Logger.Info($"   True Negatives:  {trueNegatives:F0} (correctly identified non-churners)");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"💰 Cost Breakdown:");
            ctx.Logger.Info($"   Campaign targets: {totalCampaignTargets:F0} customers");
            ctx.Logger.Info($"   Campaign cost: ${campaignCost:F2}");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"✅ Retention Success:");
            ctx.Logger.Info($"   Prevented churns: {preventedChurns:F0} customers");
            ctx.Logger.Info($"   Saved value: ${preventedChurns * CHURN_LOSS:F2}");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"❌ Churn Losses:");
            ctx.Logger.Info($"   Total churned: {totalChurnedCustomers:F0} customers");
            ctx.Logger.Info($"   Lost value: ${churnLosses:F2}");
            ctx.Logger.Info($"");
            ctx.Logger.Info($"💵 Total Cost: ${totalCost:F2}");
            ctx.Logger.Info($"📊 Cost per Customer: ${costPerCustomer:F2}");
            ctx.Logger.Info($"");

            // ROI calculation
            var savedValue = preventedChurns * CHURN_LOSS;
            var roi = ((savedValue - campaignCost) / campaignCost) * 100;
            ctx.Logger.Info($"📈 Campaign ROI: {roi:F1}%");

            return costPerCustomer;
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"❌ Churn cost calculation failed: {ex.Message}");
            // Return high cost on error (worst case)
            return 1000.0;
        }
    }
}

/// <summary>
/// Alternative: Multi-tier customer value churn cost metric
/// Different customer segments have different lifetime values
/// </summary>
public class TieredChurnCostMetric : IMLoopMetric
{
    public string Name => "Tiered Churn Cost";
    public bool HigherIsBetter => false;

    // Customer tiers with different CLV
    private const double PREMIUM_CLV = 1000.0;    // 20% of base
    private const double STANDARD_CLV = 500.0;    // 60% of base
    private const double BASIC_CLV = 200.0;       // 20% of base

    private const double CAMPAIGN_COST = 20.0;
    private const double CAMPAIGN_SUCCESS = 0.4;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(
            ctx.Predictions,
            labelColumnName: ctx.LabelColumn,
            scoreColumnName: ctx.ScoreColumn
        );

        // Simulate tier distribution
        var premiumCustomers = 200;
        var standardCustomers = 600;
        var basicCustomers = 200;

        // Calculate weighted churn loss
        var avgCLV = (premiumCustomers * PREMIUM_CLV +
                     standardCustomers * STANDARD_CLV +
                     basicCustomers * BASIC_CLV) / 1000.0;

        var truePositives = metrics.PositiveRecall * 1000;
        var falsePositives = metrics.FalsePositiveRate * 1000;
        var falseNegatives = (1 - metrics.PositiveRecall) * 1000;

        var campaignCost = (truePositives + falsePositives) * CAMPAIGN_COST;
        var preventedChurns = truePositives * CAMPAIGN_SUCCESS;
        var churnLosses = (falseNegatives + (truePositives * (1 - CAMPAIGN_SUCCESS))) * avgCLV;

        var totalCost = campaignCost + churnLosses;

        ctx.Logger.Info($"💰 Tiered Churn Cost: ${totalCost / 1000:F2} per customer");
        ctx.Logger.Info($"   (Weighted CLV: ${avgCLV:F2})");

        return totalCost / 1000;
    }
}
