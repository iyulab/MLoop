namespace MLoop.Ops.Interfaces;

/// <summary>
/// Records promotion history and backs up the production model.
/// Promotion itself is performed by the authoritative <c>ModelRegistry</c> (CLI/REST);
/// this service provides the surrounding safety/audit operations.
/// </summary>
public interface IPromotionManager
{
    /// <summary>Appends <paramref name="record"/> to its model's promotion history.</summary>
    Task RecordAsync(PromotionRecord record, CancellationToken cancellationToken = default);

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
/// One change of a model's production slot: what took it, what it replaced, how, and what the quality
/// gate said about the newcomer at that moment.
/// </summary>
/// <param name="ModelName">The model whose production slot changed.</param>
/// <param name="ExperimentId">The experiment that took the slot.</param>
/// <param name="PreviousExperimentId">The experiment it replaced; <c>null</c> for a first production model.</param>
/// <param name="Timestamp">When the slot changed.</param>
/// <param name="Action">How production changed — <c>promote</c>, <c>auto-promote</c> (a training run), <c>rollback</c>.</param>
/// <param name="Reason">A sentence for a person reading the history.</param>
/// <param name="Gate">
/// The quality gate's outcome (<see cref="PromotionGate"/>): <c>passed</c>, <c>forced</c> (it refused
/// and was overridden), or <c>not-applied</c> (a rollback returns a model that already served).
/// <c>null</c> in records written before the gate was recorded.
/// </param>
/// <param name="GateDetail">Why the gate refused (when forced), or the warning it passed with; else <c>null</c>.</param>
/// <param name="Metric">The project metric the gate judged by.</param>
/// <param name="MetricValue">The newcomer's value of <paramref name="Metric"/>.</param>
/// <param name="By">
/// Who changed the slot: the <c>sub</c> of the token that called <c>POST /promote</c> or <c>POST /rollback</c>, or what
/// <c>--by</c> said on the command line. <c>null</c> when nobody was named (a command run without <c>--by</c>) and in
/// records written before this was recorded. The login name of the machine is never written on its own: a project
/// folder goes into git, and a person's name must be given, not picked up.
/// </param>
/// <param name="Via">The way in — <see cref="PromotionVia"/>; <c>null</c> in older records.</param>
public record PromotionRecord(
    string ModelName,
    string ExperimentId,
    string? PreviousExperimentId,
    string Action,
    string? Reason,
    DateTimeOffset Timestamp,
    string? Gate = null,
    string? GateDetail = null,
    string? Metric = null,
    double? MetricValue = null,
    string? By = null,
    string? Via = null);

/// <summary>Who changed the production slot and through which door (<see cref="PromotionRecord.By"/> · <see cref="PromotionRecord.Via"/>).</summary>
/// <param name="Via">One of <see cref="PromotionVia"/>.</param>
/// <param name="By">The person or token, when known.</param>
public sealed record PromotionOrigin(string Via, string? By = null)
{
    /// <summary>The value of <see cref="By"/> with surrounding white space gone; <c>null</c> when nothing is left.</summary>
    public static string? Clean(string? by) => string.IsNullOrWhiteSpace(by) ? null : by.Trim();
}

/// <summary>The values of <see cref="PromotionRecord.Via"/>.</summary>
public static class PromotionVia
{
    /// <summary>A command of the <c>mloop</c> tool (<c>promote</c>, <c>rollback</c>, or the promotion at the end of <c>train</c>).</summary>
    public const string Cli = "cli";

    /// <summary>The HTTP server (<c>POST /promote</c>, <c>POST /rollback</c>, or the promotion at the end of a <c>POST /train</c> job).</summary>
    public const string Http = "http";
}

/// <summary>
/// The promotion history as a client reads it — <c>mloop promote --history --json</c> and <c>GET /promotions</c> list the
/// same rows, made here so a field added to a record reaches both (each used to build its own copy of the row).
/// </summary>
public static class PromotionHistoryView
{
    /// <summary>The rows of <paramref name="history"/>, oldest first as given.</summary>
    public static IReadOnlyList<object> Rows(IEnumerable<PromotionRecord> history)
        => [.. history.Select(r => (object)new
        {
            experimentId = r.ExperimentId,
            previousExperimentId = r.PreviousExperimentId,
            action = r.Action,
            timestamp = r.Timestamp.ToUniversalTime(),
            gate = r.Gate,
            gateDetail = r.GateDetail,
            metric = r.Metric,
            metricValue = r.MetricValue,
            by = r.By,
            via = r.Via,
        })];
}

/// <summary>The values of <see cref="PromotionRecord.Gate"/>.</summary>
public static class PromotionGate
{
    /// <summary>The gate passed the model.</summary>
    public const string Passed = "passed";

    /// <summary>The gate refused the model and the promotion was forced.</summary>
    public const string Forced = "forced";

    /// <summary>The gate was not applied: a rollback returns a model that already served.</summary>
    public const string NotApplied = "not-applied";
}

/// <summary>The values of <see cref="PromotionRecord.Action"/>.</summary>
public static class PromotionActions
{
    /// <summary>An explicit promotion (<c>mloop promote</c>, <c>POST /promote</c>).</summary>
    public const string Promote = "promote";

    /// <summary>A training run promoted its own result.</summary>
    public const string AutoPromote = "auto-promote";

    /// <summary>Production returned to a model that served before (<c>mloop rollback</c>, <c>POST /rollback</c>).</summary>
    public const string Rollback = "rollback";
}
