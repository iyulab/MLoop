namespace MLoop.Ops.Interfaces;

/// <summary>
/// Manages automated model promotion with safety checks.
/// </summary>
public interface IPromotionManager
{
    /// <summary>
    /// Evaluates whether a model should be auto-promoted.
    /// </summary>
    /// <param name="modelName">Model name</param>
    /// <param name="candidateExpId">Candidate experiment ID</param>
    /// <param name="policy">Promotion policy to apply</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<PromotionDecision> EvaluatePromotionAsync(
        string modelName,
        string candidateExpId,
        PromotionPolicy policy,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes promotion if approved.
    /// </summary>
    Task<PromotionOutcome> PromoteAsync(
        string modelName,
        string experimentId,
        bool createBackup = true,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Rolls back to a previous production model.
    /// </summary>
    Task<RollbackOutcome> RollbackAsync(
        string modelName,
        string? targetExpId = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets promotion history for a model.
    /// </summary>
    Task<IReadOnlyList<PromotionRecord>> GetHistoryAsync(
        string modelName,
        int limit = 10,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Policy governing auto-promotion behavior.
/// </summary>
public record PromotionPolicy(
    double MinimumImprovement,
    bool RequireComparisonWithProduction = true,
    bool RequireTestDataValidation = true,
    IReadOnlyList<string>? RequiredMetrics = null);

/// <summary>
/// Decision on whether to promote a model.
/// </summary>
public record PromotionDecision(
    bool ShouldPromote,
    string Reason,
    IReadOnlyList<string> ChecksPassed,
    IReadOnlyList<string> ChecksFailed);

/// <summary>
/// Outcome of a promotion operation.
/// </summary>
public record PromotionOutcome(
    bool Success,
    string ModelName,
    string PromotedExpId,
    string? PreviousExpId,
    string? BackupPath,
    DateTimeOffset PromotedAt);

/// <summary>
/// Outcome of a rollback operation.
/// </summary>
public record RollbackOutcome(
    bool Success,
    string ModelName,
    string RolledBackToExpId,
    string? RolledBackFromExpId,
    DateTimeOffset RolledBackAt);

/// <summary>
/// Historical record of a promotion.
/// </summary>
public record PromotionRecord(
    string ModelName,
    string ExperimentId,
    string? PreviousExperimentId,
    string Action,
    string? Reason,
    DateTimeOffset Timestamp);
