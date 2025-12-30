namespace MLoop.AIAgent.Core.Rules;

/// <summary>
/// Report on rule convergence status.
/// </summary>
public class ConvergenceReport
{
    /// <summary>Overall confidence across all rules.</summary>
    public double OverallConfidence { get; set; }

    /// <summary>Number of consecutive samples without discovering new rules.</summary>
    public int SamplesSinceLastNewRule { get; set; }

    /// <summary>Whether rules have converged (stable).</summary>
    public bool IsStable { get; set; }

    /// <summary>Recommendation for next action.</summary>
    public ConvergenceRecommendation Recommendation { get; set; }

    /// <summary>Detailed status for each rule.</summary>
    public List<RuleConfidenceDetail> RuleDetails { get; set; } = [];

    /// <summary>Summary message.</summary>
    public string Summary { get; set; } = string.Empty;
}

/// <summary>
/// Recommendations based on convergence analysis.
/// </summary>
public enum ConvergenceRecommendation
{
    /// <summary>Need more samples to establish patterns.</summary>
    ContinueSampling,

    /// <summary>Rules are stable, ready for HITL decisions.</summary>
    ProceedToHITL,

    /// <summary>High confidence, ready for bulk processing.</summary>
    ReadyForBulkProcessing,

    /// <summary>Rules are unstable, consider different approach.</summary>
    ReviewStrategy
}

/// <summary>
/// Confidence details for a single rule.
/// </summary>
public class RuleConfidenceDetail
{
    /// <summary>Rule ID.</summary>
    public required string RuleId { get; set; }

    /// <summary>Rule name.</summary>
    public required string RuleName { get; set; }

    /// <summary>Historical confidence values across samples.</summary>
    public List<double> ConfidenceHistory { get; set; } = [];

    /// <summary>Current confidence.</summary>
    public double CurrentConfidence { get; set; }

    /// <summary>Confidence trend (increasing/decreasing/stable).</summary>
    public ConfidenceTrend Trend { get; set; }

    /// <summary>Variance in confidence across samples.</summary>
    public double ConfidenceVariance { get; set; }

    /// <summary>Whether this rule is stable.</summary>
    public bool IsStable { get; set; }
}

/// <summary>
/// Trend in confidence over time.
/// </summary>
public enum ConfidenceTrend
{
    Increasing,
    Stable,
    Decreasing,
    Volatile
}

/// <summary>
/// Calculator for rule confidence and convergence detection.
/// </summary>
public class ConfidenceCalculator
{
    /// <summary>
    /// Minimum confidence threshold for considering a rule stable.
    /// </summary>
    public double StabilityThreshold { get; set; } = 0.98;

    /// <summary>
    /// Number of samples without new rules to consider converged.
    /// </summary>
    public int ConvergenceSampleCount { get; set; } = 500;

    /// <summary>
    /// Maximum allowed variance for stability.
    /// </summary>
    public double MaxVarianceForStability { get; set; } = 0.05;

    private readonly Dictionary<string, List<double>> _confidenceHistory = [];
    private int _samplesSinceLastNewRule;
    private int _lastKnownRuleCount;

    /// <summary>
    /// Updates confidence tracking with new validation results.
    /// </summary>
    /// <param name="validationResults">Results from rule validation.</param>
    /// <param name="newRulesDiscovered">Number of new rules discovered in this sample.</param>
    public void Update(List<RuleValidationResult> validationResults, int newRulesDiscovered)
    {
        foreach (var result in validationResults)
        {
            var ruleId = result.Rule.Id;

            if (!_confidenceHistory.TryGetValue(ruleId, out var history))
            {
                history = [];
                _confidenceHistory[ruleId] = history;
            }

            history.Add(result.UpdatedConfidence);

            // Keep only last 10 confidence values
            if (history.Count > 10)
                history.RemoveAt(0);
        }

        // Track new rule discovery
        if (newRulesDiscovered > 0)
        {
            _samplesSinceLastNewRule = 0;
            _lastKnownRuleCount += newRulesDiscovered;
        }
        else
        {
            _samplesSinceLastNewRule++;
        }
    }

    /// <summary>
    /// Calculates confidence for a single rule across sample history.
    /// </summary>
    public double CalculateRuleConfidence(PreprocessingRule rule)
    {
        if (!_confidenceHistory.TryGetValue(rule.Id, out var history) || history.Count == 0)
            return rule.Confidence;

        // Use exponential moving average for recent bias
        var weights = Enumerable.Range(0, history.Count)
            .Select(i => Math.Exp(i - history.Count + 1))
            .ToList();

        var weightSum = weights.Sum();
        var weightedSum = history.Zip(weights, (c, w) => c * w).Sum();

        return weightedSum / weightSum;
    }

    /// <summary>
    /// Checks if rules have converged (no new patterns, stable confidence).
    /// </summary>
    public bool HasConverged(List<PreprocessingRule> rules)
    {
        if (_samplesSinceLastNewRule < ConvergenceSampleCount)
            return false;

        foreach (var rule in rules)
        {
            var confidence = CalculateRuleConfidence(rule);
            if (confidence < StabilityThreshold)
                return false;

            if (_confidenceHistory.TryGetValue(rule.Id, out var history) && history.Count >= 3)
            {
                var variance = CalculateVariance(history);
                if (variance > MaxVarianceForStability)
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Generates a comprehensive convergence report.
    /// </summary>
    public ConvergenceReport GetConvergenceReport(List<PreprocessingRule> rules)
    {
        var ruleDetails = new List<RuleConfidenceDetail>();
        var confidences = new List<double>();

        foreach (var rule in rules)
        {
            var currentConfidence = CalculateRuleConfidence(rule);
            confidences.Add(currentConfidence);

            var detail = new RuleConfidenceDetail
            {
                RuleId = rule.Id,
                RuleName = rule.Name,
                CurrentConfidence = currentConfidence,
                ConfidenceHistory = _confidenceHistory.GetValueOrDefault(rule.Id, []),
                Trend = CalculateTrend(rule.Id),
                ConfidenceVariance = CalculateRuleVariance(rule.Id),
                IsStable = IsRuleStable(rule)
            };

            ruleDetails.Add(detail);
        }

        var overallConfidence = confidences.Count > 0 ? confidences.Average() : 0;
        var isStable = HasConverged(rules);
        var recommendation = DetermineRecommendation(rules, isStable, overallConfidence);

        return new ConvergenceReport
        {
            OverallConfidence = overallConfidence,
            SamplesSinceLastNewRule = _samplesSinceLastNewRule,
            IsStable = isStable,
            Recommendation = recommendation,
            RuleDetails = ruleDetails,
            Summary = GenerateSummary(rules, isStable, overallConfidence, recommendation)
        };
    }

    /// <summary>
    /// Resets the calculator state.
    /// </summary>
    public void Reset()
    {
        _confidenceHistory.Clear();
        _samplesSinceLastNewRule = 0;
        _lastKnownRuleCount = 0;
    }

    private ConfidenceTrend CalculateTrend(string ruleId)
    {
        if (!_confidenceHistory.TryGetValue(ruleId, out var history) || history.Count < 3)
            return ConfidenceTrend.Stable;

        var recent = history.TakeLast(3).ToList();
        var slope = (recent[2] - recent[0]) / 2;

        if (Math.Abs(slope) < 0.02)
            return ConfidenceTrend.Stable;

        var variance = CalculateVariance(recent);
        if (variance > 0.1)
            return ConfidenceTrend.Volatile;

        return slope > 0 ? ConfidenceTrend.Increasing : ConfidenceTrend.Decreasing;
    }

    private double CalculateRuleVariance(string ruleId)
    {
        if (!_confidenceHistory.TryGetValue(ruleId, out var history) || history.Count < 2)
            return 0;

        return CalculateVariance(history);
    }

    private bool IsRuleStable(PreprocessingRule rule)
    {
        var confidence = CalculateRuleConfidence(rule);
        if (confidence < StabilityThreshold)
            return false;

        var variance = CalculateRuleVariance(rule.Id);
        return variance <= MaxVarianceForStability;
    }

    private ConvergenceRecommendation DetermineRecommendation(
        List<PreprocessingRule> rules,
        bool isStable,
        double overallConfidence)
    {
        var pendingHITL = rules.Count(r => r.RequiresHITL && !r.IsApproved);

        if (!isStable && _samplesSinceLastNewRule < 100)
            return ConvergenceRecommendation.ContinueSampling;

        if (overallConfidence < 0.7)
            return ConvergenceRecommendation.ReviewStrategy;

        if (pendingHITL > 0)
            return ConvergenceRecommendation.ProceedToHITL;

        if (isStable && overallConfidence >= StabilityThreshold)
            return ConvergenceRecommendation.ReadyForBulkProcessing;

        return ConvergenceRecommendation.ContinueSampling;
    }

    private string GenerateSummary(
        List<PreprocessingRule> rules,
        bool isStable,
        double overallConfidence,
        ConvergenceRecommendation recommendation)
    {
        var approvedCount = rules.Count(r => r.IsApproved);
        var pendingHITL = rules.Count(r => r.RequiresHITL && !r.IsApproved);
        var autoRules = rules.Count(r => r.IsAutoResolvable);

        var status = isStable ? "STABLE" : "EVOLVING";
        var summary = $"Rule Status: {status} | Overall Confidence: {overallConfidence:P1}\n";
        summary += $"Rules: {rules.Count} total ({approvedCount} approved, {pendingHITL} pending HITL, {autoRules} auto-resolvable)\n";
        summary += $"Convergence: {_samplesSinceLastNewRule} samples since last new rule\n";
        summary += $"Recommendation: {recommendation}";

        return summary;
    }

    private static double CalculateVariance(List<double> values)
    {
        if (values.Count < 2)
            return 0;

        var mean = values.Average();
        return values.Select(v => Math.Pow(v - mean, 2)).Average();
    }
}
