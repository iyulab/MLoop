namespace MLoop.Ops.Interfaces;

/// <summary>
/// Records promotion history and backs up the production model.
/// Promotion itself is performed by the authoritative <c>ModelRegistry</c> (CLI/REST);
/// this service provides the surrounding safety/audit operations.
/// </summary>
public interface IPromotionManager
{
    /// <summary>
    /// Records a promotion event in the model's promotion history.
    /// </summary>
    Task RecordPromotionAsync(
        string modelName,
        string experimentId,
        string? previousExpId,
        string action,
        string? reason = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates a backup of the current production model directory.
    /// Returns the backup path, or null if no production model exists.
    /// </summary>
    Task<string?> BackupProductionAsync(
        string modelName,
        CancellationToken cancellationToken = default);
}

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
