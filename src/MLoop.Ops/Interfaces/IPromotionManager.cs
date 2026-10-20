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

    /// <summary>
    /// The model's promotion history, oldest first. Empty when nothing was ever promoted.
    /// </summary>
    Task<IReadOnlyList<PromotionRecord>> GetHistoryAsync(
        string modelName,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Which experiment a rollback returns production to — decided from the promotion history alone, so
/// the CLI and the REST API cannot disagree about it.
/// </summary>
public static class RollbackTarget
{
    /// <summary>
    /// The experiment production held before <paramref name="currentProduction"/> took its place, or,
    /// with <paramref name="requested"/>, that experiment when it has been production before. Returns
    /// the experiment id, or an error that says why there is none.
    /// </summary>
    /// <remarks>
    /// Only a former production experiment qualifies: going back to one is undoing a decision, while
    /// promoting an experiment that never served is a new decision, which is <c>promote</c>'s — with
    /// the quality gate that comes with it.
    /// </remarks>
    public static (string? ExperimentId, string? Error) Resolve(
        IReadOnlyList<PromotionRecord> history, string? currentProduction, string? requested = null)
    {
        if (currentProduction is null)
            return (null, "There is no production model to roll back.");

        if (requested is not null)
        {
            if (requested.Equals(currentProduction, StringComparison.OrdinalIgnoreCase))
                return (null, $"'{requested}' is already in production.");

            return history.Any(r => r.ExperimentId.Equals(requested, StringComparison.OrdinalIgnoreCase))
                ? (requested, null)
                : (null, $"'{requested}' has never been in production, so there is nothing to roll back to. "
                         + $"Use 'mloop promote {requested}' to make it production.");
        }

        var arrival = history.LastOrDefault(r => r.ExperimentId.Equals(currentProduction, StringComparison.OrdinalIgnoreCase));
        return arrival?.PreviousExperimentId is { } previous
            ? (previous, null)
            : (null, $"The promotion history does not record what production held before '{currentProduction}'. "
                     + "Name the experiment with --to.");
    }
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
