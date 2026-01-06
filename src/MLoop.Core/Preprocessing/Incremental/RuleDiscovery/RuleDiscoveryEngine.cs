using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery;

/// <summary>
/// Main engine for discovering preprocessing rules from data samples.
/// Orchestrates pattern detection and rule generation.
/// </summary>
public sealed class RuleDiscoveryEngine : IRuleDiscoveryEngine
{
    private readonly ILogger<RuleDiscoveryEngine> _logger;
    private readonly List<IPatternDetector> _detectors;

    public RuleDiscoveryEngine(ILogger<RuleDiscoveryEngine> logger)
    {
        _logger = logger;
        _detectors = InitializeDetectors();
    }

    public async Task<IReadOnlyList<PreprocessingRule>> DiscoverRulesAsync(
        DataFrame sample,
        SampleAnalysis analysis,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation(
            "Starting rule discovery on sample with {Rows} rows, {Cols} columns",
            sample.Rows.Count,
            sample.Columns.Count);

        var allPatterns = new List<DetectedPattern>();

        // Run pattern detection on each column
        foreach (var column in sample.Columns)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var columnPatterns = await DetectPatternsInColumnAsync(
                column,
                cancellationToken);

            allPatterns.AddRange(columnPatterns);
        }

        _logger.LogInformation(
            "Detected {PatternCount} patterns across {ColumnCount} columns",
            allPatterns.Count,
            sample.Columns.Count);

        // Convert patterns to rules
        var rules = GenerateRulesFromPatterns(allPatterns, analysis.StageNumber);

        // Prioritize rules
        var prioritizedRules = PrioritizeRules(rules);

        _logger.LogInformation(
            "Generated {RuleCount} preprocessing rules ({AutoFixable} auto-fixable, {HITL} HITL-required)",
            prioritizedRules.Count,
            prioritizedRules.Count(r => !r.RequiresHITL),
            prioritizedRules.Count(r => r.RequiresHITL));

        return prioritizedRules;
    }

    public Task<ConfidenceScore> CalculateConfidenceAsync(
        PreprocessingRule rule,
        DataFrame previousSample,
        DataFrame currentSample,
        CancellationToken cancellationToken = default)
    {
        // This will be implemented in Phase 4 (ConfidenceCalculator)
        throw new NotImplementedException("Confidence calculation will be implemented in Phase 4");
    }

    public bool HasConverged(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules,
        double threshold = 0.02)
    {
        // This will be implemented in Phase 4 (ConvergenceDetector)
        throw new NotImplementedException("Convergence detection will be implemented in Phase 4");
    }

    /// <summary>
    /// Run all applicable detectors on a single column.
    /// </summary>
    private async Task<List<DetectedPattern>> DetectPatternsInColumnAsync(
        DataFrameColumn column,
        CancellationToken cancellationToken)
    {
        var patterns = new List<DetectedPattern>();

        foreach (var detector in _detectors)
        {
            if (!detector.IsApplicable(column))
                continue;

            try
            {
                var detectedPatterns = await detector.DetectAsync(
                    column,
                    column.Name,
                    cancellationToken);

                patterns.AddRange(detectedPatterns);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(
                    ex,
                    "Pattern detector {Detector} failed on column {Column}",
                    detector.GetType().Name,
                    column.Name);
            }
        }

        return patterns;
    }

    /// <summary>
    /// Convert detected patterns into preprocessing rules.
    /// </summary>
    private List<PreprocessingRule> GenerateRulesFromPatterns(
        List<DetectedPattern> patterns,
        int stageNumber)
    {
        var rules = new List<PreprocessingRule>();

        foreach (var pattern in patterns)
        {
            var rule = ConvertPatternToRule(pattern, stageNumber);
            rules.Add(rule);
        }

        return rules;
    }

    /// <summary>
    /// Convert a single detected pattern into a preprocessing rule.
    /// </summary>
    private PreprocessingRule ConvertPatternToRule(
        DetectedPattern pattern,
        int stageNumber)
    {
        var ruleType = DetermineRuleType(pattern.Type);
        var requiresHITL = RequiresHumanApproval(ruleType);
        var priority = DeterminePriority(pattern.Severity, ruleType);

        var rule = new PreprocessingRule
        {
            Id = GenerateRuleId(ruleType, pattern.ColumnName, pattern.Type),
            Type = ruleType,
            ColumnNames = new[] { pattern.ColumnName },
            Description = pattern.Description,
            PatternType = pattern.Type,
            Confidence = pattern.Confidence,
            RequiresHITL = requiresHITL,
            Priority = priority,
            AffectedRows = pattern.Occurrences,
            DiscoveredInStage = stageNumber,
            SuggestedAction = pattern.SuggestedFix,
            Examples = pattern.Examples
        };

        // Add pattern-specific parameters
        PopulateRuleParameters(rule, pattern);

        return rule;
    }

    /// <summary>
    /// Determine the preprocessing rule type from pattern type.
    /// </summary>
    private static PreprocessingRuleType DetermineRuleType(PatternType patternType)
    {
        return patternType switch
        {
            PatternType.MissingValue => PreprocessingRuleType.MissingValueStrategy,
            PatternType.TypeInconsistency => PreprocessingRuleType.TypeConversion,
            PatternType.FormatVariation => PreprocessingRuleType.DateFormatStandardization,
            PatternType.OutlierAnomaly => PreprocessingRuleType.OutlierHandling,
            PatternType.CategoryVariation => PreprocessingRuleType.CategoryMapping,
            PatternType.EncodingIssue => PreprocessingRuleType.EncodingNormalization,
            PatternType.WhitespaceIssue => PreprocessingRuleType.WhitespaceNormalization,
            PatternType.BusinessRule => PreprocessingRuleType.BusinessLogicDecision,
            _ => PreprocessingRuleType.BusinessLogicDecision
        };
    }

    /// <summary>
    /// Determine if a rule type requires human-in-the-loop approval.
    /// </summary>
    private static bool RequiresHumanApproval(PreprocessingRuleType ruleType)
    {
        return ruleType switch
        {
            // Auto-resolvable rules
            PreprocessingRuleType.DateFormatStandardization => false,
            PreprocessingRuleType.EncodingNormalization => false,
            PreprocessingRuleType.WhitespaceNormalization => false,
            PreprocessingRuleType.NumericFormatStandardization => false,

            // HITL-required rules
            PreprocessingRuleType.MissingValueStrategy => true,
            PreprocessingRuleType.OutlierHandling => true,
            PreprocessingRuleType.CategoryMapping => true,
            PreprocessingRuleType.TypeConversion => true,
            PreprocessingRuleType.BusinessLogicDecision => true,

            _ => true // Default to HITL for safety
        };
    }

    /// <summary>
    /// Determine rule priority based on severity and type.
    /// Priority: 1-10 (10 = Critical, 5 = Normal, 1 = Low)
    /// </summary>
    private static int DeterminePriority(Severity severity, PreprocessingRuleType ruleType)
    {
        var basePriority = severity switch
        {
            Severity.Critical => 10,
            Severity.High => 7,
            Severity.Medium => 5,
            Severity.Low => 3,
            _ => 1
        };

        // Adjust for rule type criticality
        var adjustment = ruleType switch
        {
            PreprocessingRuleType.MissingValueStrategy => 2,  // Data integrity
            PreprocessingRuleType.TypeConversion => 2,        // Data integrity
            PreprocessingRuleType.OutlierHandling => 1,       // Data quality
            PreprocessingRuleType.EncodingNormalization => 1, // Data quality
            _ => 0
        };

        return Math.Clamp(basePriority + adjustment, 1, 10);
    }

    /// <summary>
    /// Generate unique rule ID.
    /// Format: {RuleType}_{ColumnName}_{PatternType}
    /// </summary>
    private static string GenerateRuleId(
        PreprocessingRuleType ruleType,
        string columnName,
        PatternType patternType)
    {
        return $"{ruleType}_{columnName}_{patternType}";
    }

    /// <summary>
    /// Populate rule-specific parameters from pattern.
    /// </summary>
    private static void PopulateRuleParameters(
        PreprocessingRule rule,
        DetectedPattern pattern)
    {
        rule.Parameters["affected_percentage"] = pattern.AffectedPercentage;
        rule.Parameters["severity"] = pattern.Severity.ToString();
        rule.Parameters["pattern_confidence"] = pattern.Confidence;

        // Pattern-specific parameters
        switch (pattern.Type)
        {
            case PatternType.MissingValue:
                rule.Parameters["strategy"] = "impute_median"; // Default strategy
                break;

            case PatternType.FormatVariation:
                rule.Parameters["target_format"] = "ISO-8601"; // Standard format
                break;

            case PatternType.EncodingIssue:
                rule.Parameters["target_encoding"] = "UTF-8"; // Standard encoding
                break;

            case PatternType.WhitespaceIssue:
                rule.Parameters["trim"] = true;
                rule.Parameters["collapse_spaces"] = true;
                break;
        }
    }

    /// <summary>
    /// Prioritize rules by priority (descending), then severity.
    /// </summary>
    private static List<PreprocessingRule> PrioritizeRules(
        List<PreprocessingRule> rules)
    {
        return rules
            .OrderByDescending(r => r.Priority)
            .ThenByDescending(r => r.AffectedRows)
            .ToList();
    }

    /// <summary>
    /// Initialize all pattern detectors.
    /// </summary>
    private static List<IPatternDetector> InitializeDetectors()
    {
        return new List<IPatternDetector>
        {
            new MissingValueDetector(),
            new TypeInconsistencyDetector(),
            new FormatVariationDetector(),
            new OutlierDetector(),
            new CategoryVariationDetector(),
            new EncodingIssueDetector(),
            new WhitespaceDetector()
        };
    }
}
