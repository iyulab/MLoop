using MLoop.Extensibility.Metrics;

/// <summary>
/// ROI Optimization Metric for Binary Classification
///
/// Business Context:
/// - Evaluating marketing campaign effectiveness
/// - True Positive: Campaign succeeds → Customer converts
/// - False Positive: Campaign fails → Cost wasted
/// - ROI = ((Revenue - Cost) / Cost) × 100
///
/// Formula: ROI = ((TP × ConversionValue) - (AllPositives × CampaignCost)) / (AllPositives × CampaignCost) × 100
///
/// Example Use Case:
/// - Email marketing campaign optimization
/// - Paid advertising targeting
/// - Sales lead prioritization
/// </summary>
public class ROIOptimizationMetric : IMLoopMetric
{
    public string Name => "Marketing ROI";
    public string Description => "Return on investment for marketing campaigns";

    // Configuration: Adjust based on your business model
    private const double CONVERSION_VALUE = 200.0;      // Revenue per conversion
    private const double CAMPAIGN_COST_PER_CONTACT = 5.0;  // Cost per campaign contact

    public Task<MetricResult> CalculateAsync(MetricContext ctx)
    {
        // Validate task type
        if (ctx.TaskType != "BinaryClassification")
        {
            throw new InvalidOperationException(
                $"ROI metric requires binary classification, got: {ctx.TaskType}");
        }

        // Extract predictions and labels
        var predictions = (bool[])ctx.Predictions;  // True = target for campaign
        var labels = (bool[])ctx.Labels;            // True = actually converted

        if (predictions.Length != labels.Length)
        {
            throw new InvalidOperationException(
                $"Prediction count ({predictions.Length}) != Label count ({labels.Length})");
        }

        // Calculate metrics
        int truePositives = 0;   // Successful campaigns (predicted + converted)
        int falsePositives = 0;  // Failed campaigns (predicted but not converted)
        int falseNegatives = 0;  // Missed opportunities (not targeted but would have converted)
        int totalContacts = 0;   // Total campaign contacts (TP + FP)

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i])
            {
                totalContacts++;
                if (labels[i])
                    truePositives++;
                else
                    falsePositives++;
            }
            else if (labels[i])
            {
                falseNegatives++;
            }
        }

        // Calculate financial metrics
        double revenue = truePositives * CONVERSION_VALUE;
        double campaignCost = totalContacts * CAMPAIGN_COST_PER_CONTACT;
        double profit = revenue - campaignCost;
        double roi = campaignCost > 0
            ? ((revenue - campaignCost) / campaignCost) * 100
            : 0;

        // Calculate opportunity cost
        double missedRevenue = falseNegatives * CONVERSION_VALUE;
        double potentialRevenue = revenue + missedRevenue;
        double potentialCampaignCost = (totalContacts + falseNegatives) * CAMPAIGN_COST_PER_CONTACT;
        double potentialROI = potentialCampaignCost > 0
            ? ((potentialRevenue - potentialCampaignCost) / potentialCampaignCost) * 100
            : 0;

        // Calculate conversion rate
        double conversionRate = totalContacts > 0
            ? (truePositives / (double)totalContacts) * 100
            : 0;

        // Calculate efficiency metrics
        double costPerAcquisition = truePositives > 0
            ? campaignCost / truePositives
            : campaignCost;

        // Logging
        ctx.Logger.Info($"📧 Campaign Performance:");
        ctx.Logger.Info($"   Contacts:                {totalContacts,4}");
        ctx.Logger.Info($"   Conversions (TP):        {truePositives,4}");
        ctx.Logger.Info($"   Failed Campaigns (FP):   {falsePositives,4}");
        ctx.Logger.Info($"   Missed Opportunities:    {falseNegatives,4}");
        ctx.Logger.Info($"   Conversion Rate:         {conversionRate,6:F2}%");
        ctx.Logger.Info($"");
        ctx.Logger.Info($"💰 Financial Analysis:");
        ctx.Logger.Info($"   Revenue:                ${revenue,8:F2}");
        ctx.Logger.Info($"   Campaign Cost:          ${campaignCost,8:F2}");
        ctx.Logger.Info($"   Profit:                 ${profit,8:F2}");
        ctx.Logger.Info($"   ROI:                     {roi,7:F1}%");
        ctx.Logger.Info($"   Cost per Acquisition:   ${costPerAcquisition,8:F2}");
        ctx.Logger.Info($"");
        ctx.Logger.Info($"📈 Potential (if FNs targeted):");
        ctx.Logger.Info($"   Potential Revenue:      ${potentialRevenue,8:F2}");
        ctx.Logger.Info($"   Potential ROI:           {potentialROI,7:F1}%");

        return Task.FromResult(new MetricResult
        {
            Name = Name,
            Value = roi,
            Description = $"{roi:F1}% ROI (${profit:F2} profit from {totalContacts} contacts, {truePositives} conversions)",
            Details = new Dictionary<string, object>
            {
                ["TruePositives"] = truePositives,
                ["FalsePositives"] = falsePositives,
                ["FalseNegatives"] = falseNegatives,
                ["TotalContacts"] = totalContacts,
                ["Revenue"] = revenue,
                ["CampaignCost"] = campaignCost,
                ["Profit"] = profit,
                ["ConversionRate"] = conversionRate,
                ["CostPerAcquisition"] = costPerAcquisition,
                ["MissedRevenue"] = missedRevenue,
                ["PotentialRevenue"] = potentialRevenue,
                ["PotentialROI"] = potentialROI,
                ["ConversionValue"] = CONVERSION_VALUE,
                ["CampaignCostPerContact"] = CAMPAIGN_COST_PER_CONTACT
            }
        });
    }
}
