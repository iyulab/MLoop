using System.Diagnostics;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleApplication;

/// <summary>
/// Applies preprocessing rules to DataFrame instances.
/// Uses strategy pattern for different rule types.
/// </summary>
public sealed class RuleApplier : IRuleApplier
{
    private readonly ILogger<RuleApplier> _logger;
    private readonly bool _continueOnFailure;

    public RuleApplier(
        ILogger<RuleApplier> logger,
        bool continueOnFailure = true)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _continueOnFailure = continueOnFailure;
    }

    /// <inheritdoc />
    public async Task<RuleApplicationResult> ApplyRuleAsync(
        DataFrame data,
        PreprocessingRule rule,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            _logger.LogInformation("Applying rule {RuleId}: {Description}", rule.Id, rule.Description);

            // Validate rule can be applied
            if (!ValidateRule(data, rule))
            {
                return new RuleApplicationResult
                {
                    Rule = rule,
                    Status = RuleApplicationStatus.Failed,
                    RowsAffected = 0,
                    RowsSkipped = (int)data.Rows.Count,
                    Duration = stopwatch.Elapsed,
                    Success = false,
                    ErrorMessage = "Rule validation failed: columns not found or incompatible types"
                };
            }

            // Apply rule based on type
            var outcome = rule.Type switch
            {
                PreprocessingRuleType.MissingValueStrategy => ApplyMissingValueStrategy(data, rule),
                PreprocessingRuleType.OutlierHandling => ApplyOutlierHandling(data, rule),
                PreprocessingRuleType.WhitespaceNormalization => ApplyWhitespaceNormalization(data, rule),
                PreprocessingRuleType.DateFormatStandardization => ApplyDateFormatStandardization(data, rule),
                PreprocessingRuleType.CategoryMapping => ApplyCategoryMapping(data, rule),
                PreprocessingRuleType.TypeConversion => ApplyTypeConversion(data, rule),
                PreprocessingRuleType.EncodingNormalization => ApplyEncodingNormalization(data, rule),
                PreprocessingRuleType.NumericFormatStandardization => ApplyNumericFormatStandardization(data, rule),
                PreprocessingRuleType.BusinessLogicDecision => ApplyBusinessLogicDecision(data, rule),
                _ => throw new NotSupportedException($"Rule type {rule.Type} is not supported")
            };

            stopwatch.Stop();

            if (outcome.Status == RuleApplicationStatus.NotImplemented)
            {
                // Do NOT report success: the data was left untouched. Surfacing this prevents
                // callers from presenting unmodified data as "cleaned".
                _logger.LogWarning(
                    "Rule {RuleId} ({RuleType}) has no application strategy implemented yet; data left unchanged.",
                    rule.Id, rule.Type);

                return new RuleApplicationResult
                {
                    Rule = rule,
                    Status = RuleApplicationStatus.NotImplemented,
                    RowsAffected = 0,
                    RowsSkipped = (int)data.Rows.Count,
                    Duration = stopwatch.Elapsed,
                    Success = false,
                    ErrorMessage = $"Rule type '{rule.Type}' application is not yet implemented"
                };
            }

            _logger.LogInformation(
                "Rule {RuleId} applied successfully. Rows affected: {RowsAffected}, Duration: {Duration}ms",
                rule.Id, outcome.RowsAffected, stopwatch.ElapsedMilliseconds);

            return new RuleApplicationResult
            {
                Rule = rule,
                Status = RuleApplicationStatus.Applied,
                RowsAffected = outcome.RowsAffected,
                RowsSkipped = (int)data.Rows.Count - outcome.RowsAffected,
                Duration = stopwatch.Elapsed,
                Success = true
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            _logger.LogError(ex, "Failed to apply rule {RuleId}: {Message}", rule.Id, ex.Message);

            return new RuleApplicationResult
            {
                Rule = rule,
                Status = RuleApplicationStatus.Failed,
                RowsAffected = 0,
                RowsSkipped = (int)data.Rows.Count,
                Duration = stopwatch.Elapsed,
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <inheritdoc />
    public async Task<BulkApplicationResult> ApplyRulesAsync(
        DataFrame data,
        IReadOnlyList<PreprocessingRule> rules,
        IProgress<RuleApplicationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var results = new List<RuleApplicationResult>();

        _logger.LogInformation("Starting bulk rule application for {RuleCount} rules", rules.Count);

        for (var i = 0; i < rules.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var rule = rules[i];

            // Report progress
            progress?.Report(new RuleApplicationProgress
            {
                CurrentRule = rule,
                RuleIndex = i,
                TotalRules = rules.Count,
                Message = $"Applying rule {i + 1}/{rules.Count}: {rule.Description}"
            });

            // Apply rule
            var result = await ApplyRuleAsync(data, rule, cancellationToken).ConfigureAwait(false);
            results.Add(result);

            // Stop on failure if not continuing
            if (!result.Success && !_continueOnFailure)
            {
                _logger.LogWarning("Rule application failed and ContinueOnFailure is false. Stopping.");
                break;
            }
        }

        stopwatch.Stop();

        var bulkResult = new BulkApplicationResult
        {
            TotalRules = rules.Count,
            SuccessfulRules = results.Count(r => r.Success),
            FailedRules = results.Count(r => !r.Success),
            Results = results,
            TotalDuration = stopwatch.Elapsed
        };

        _logger.LogInformation(
            "Bulk application complete. Success: {Success}/{Total}, Failed: {Failed}, Duration: {Duration}s",
            bulkResult.SuccessfulRules, bulkResult.TotalRules, bulkResult.FailedRules,
            bulkResult.TotalDuration.TotalSeconds);

        return bulkResult;
    }

    /// <inheritdoc />
    public bool ValidateRule(DataFrame data, PreprocessingRule rule)
    {
        // Check all required columns exist
        var existingColumns = new HashSet<string>(
            data.Columns.Select(c => c.Name),
            StringComparer.OrdinalIgnoreCase);

        foreach (var columnName in rule.ColumnNames)
        {
            if (!existingColumns.Contains(columnName))
            {
                _logger.LogWarning("Validation failed: Column {ColumnName} not found in DataFrame", columnName);
                return false;
            }
        }

        return true;
    }

    // ===== Rule Application Strategies =====
    //
    // Each strategy returns a StrategyOutcome. Until the Rule Application Engine is built
    // (see claudedocs/plans/2026-06-19-rule-application-engine.md and
    // ISSUE-mloop-20260619-ruleapplier-noop-apply), every strategy returns NotImplemented so
    // the caller cannot mistake an untouched DataFrame for a successful transform. As each
    // strategy is implemented, return StrategyOutcome.Applied(rowsAffected) instead.

    /// <summary>
    /// Outcome of a single rule-application strategy: how many rows it changed and whether
    /// a real strategy ran at all.
    /// </summary>
    private readonly record struct StrategyOutcome(int RowsAffected, RuleApplicationStatus Status)
    {
        public static StrategyOutcome NotImplemented { get; } = new(0, RuleApplicationStatus.NotImplemented);
        public static StrategyOutcome Applied(int rowsAffected) => new(rowsAffected, RuleApplicationStatus.Applied);
    }

    // MissingValue: identify missing values, apply strategy (delete / impute mean·median·mode / default).
    private StrategyOutcome ApplyMissingValueStrategy(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // Outlier: detect via IQR / Z-score, then remove / cap / flag.
    private StrategyOutcome ApplyOutlierHandling(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // Whitespace: trim and normalize internal whitespace.
    private StrategyOutcome ApplyWhitespaceNormalization(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // DateFormat: parse various formats, convert to ISO-8601.
    private StrategyOutcome ApplyDateFormatStandardization(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // CategoryMapping: map variations to canonical values, merge similar categories.
    private StrategyOutcome ApplyCategoryMapping(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // TypeConversion: parse and convert to target type, handling errors gracefully.
    private StrategyOutcome ApplyTypeConversion(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // Encoding: detect mojibake / corruption, normalize to UTF-8.
    private StrategyOutcome ApplyEncodingNormalization(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // NumericFormat: remove thousand separators, standardize decimal separators.
    private StrategyOutcome ApplyNumericFormatStandardization(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;

    // BusinessLogic: domain-specific rules. For custom-detector rules this will dispatch to the
    // detector's paired applier (IPatternApplier) rather than embedding domain logic in core.
    private StrategyOutcome ApplyBusinessLogicDecision(DataFrame data, PreprocessingRule rule)
        => StrategyOutcome.NotImplemented;
}
