using System.Text.Json;
using MLoop.Core.Monitoring;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>Whether drift could be measured, and if not, why.</summary>
public enum DriftCheckStatus
{
    /// <summary>The window was compared with the production model's training profile.</summary>
    Measured = 0,

    /// <summary>No model is in production.</summary>
    NoProduction = 1,

    /// <summary>The production model has no drift profile — trained before profiles existed, or a task that reads no feature columns.</summary>
    NoProfile = 2,

    /// <summary>Too few logged predictions of the production model to compare.</summary>
    TooFewRows = 3,
}

/// <summary>A drift check: the overall report, and one per slice when a slice column was named.</summary>
public sealed record DriftCheck(
    DriftCheckStatus Status,
    string? ExperimentId,
    int Rows,
    DriftReport? Report,
    IReadOnlyList<SliceDrift>? Slices,
    string Message);

/// <summary>One slice's drift — the rows whose slice column holds <paramref name="Value"/>.</summary>
public sealed record SliceDrift(string Value, DriftReport Report);

/// <summary>
/// Compares the production model's recent inputs — its own logged predictions, newest first — with
/// the distribution it was trained on (<see cref="ExperimentLayout.DriftProfileFileName"/>). Computed
/// when asked; nothing runs in between.
/// </summary>
public static class DriftEvaluator
{
    /// <summary>Logged predictions compared by default: the most recent this many.</summary>
    public const int DefaultWindowRows = 5_000;

    /// <summary>Fewer logged predictions than this, and a PSI says more about chance than about drift.</summary>
    public const int MinimumRows = 100;

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
    };

    /// <summary>Checks <paramref name="modelName"/>'s production model.</summary>
    /// <param name="projectRoot">The project directory.</param>
    /// <param name="logs">The prediction logs to read the recent inputs from.</param>
    /// <param name="modelName">The model.</param>
    /// <param name="windowRows">How many of the most recent logged predictions to compare.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <param name="sliceColumn">An input column to report drift per value of (values with at least <see cref="MinimumRows"/> rows).</param>
    public static async Task<DriftCheck> CheckAsync(
        string projectRoot, IPredictionLogger logs, string modelName,
        int windowRows = DefaultWindowRows, string? sliceColumn = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(logs);
        var production = Path.Combine(projectRoot, ExperimentLayout.ModelsDirectory, modelName, ExperimentLayout.ProductionDirectory);
        var metadata = await ProductionMetadata.ReadAsync(production, cancellationToken).ConfigureAwait(false);
        if (metadata?.ExperimentId is not { } experimentId)
            return new DriftCheck(DriftCheckStatus.NoProduction, null, 0, null, null,
                $"No model is in production for '{modelName}'.");

        var profilePath = Path.Combine(production, ExperimentLayout.DriftProfileFileName);
        if (!File.Exists(profilePath))
            return new DriftCheck(DriftCheckStatus.NoProfile, experimentId, 0, null, null,
                $"The production model ({experimentId}) has no drift profile — it was trained before profiles were recorded, "
                + "or its task reads no feature columns. Retrain and promote to start measuring drift.");
        DriftProfile profile;
        await using (var stream = File.OpenRead(profilePath))
            profile = await JsonSerializer.DeserializeAsync<DriftProfile>(stream, Options, cancellationToken).ConfigureAwait(false)
                      ?? throw new JsonException($"{profilePath} holds no profile.");

        // Only what this model answered: another experiment's predictions describe another model's traffic.
        var entries = (await logs.GetLogsAsync(modelName, limit: Math.Max(windowRows, 1) * 2, cancellationToken: cancellationToken).ConfigureAwait(false))
            .Where(e => string.Equals(e.ExperimentId, experimentId, StringComparison.OrdinalIgnoreCase))
            .Take(windowRows)
            .Select(e => new Logged(
                e.Input.ToDictionary(kv => kv.Key, kv => (object?)kv.Value, StringComparer.OrdinalIgnoreCase),
                e.Output))
            .ToList();
        if (entries.Count < MinimumRows)
            return new DriftCheck(DriftCheckStatus.TooFewRows, experimentId, entries.Count, null, null,
                $"{entries.Count} logged predictions of {experimentId}; at least {MinimumRows} are needed to compare. "
                + "Predictions are logged with `mloop predict --log` and by `mloop serve`.");

        var report = Compare(profile, entries);
        IReadOnlyList<SliceDrift>? slices = null;
        if (!string.IsNullOrWhiteSpace(sliceColumn))
        {
            slices = entries
                .GroupBy(r => r.Input.TryGetValue(sliceColumn, out var v) ? Text(v) : "(missing)", StringComparer.Ordinal)
                .Where(g => g.Count() >= MinimumRows)
                .Select(g => new SliceDrift(g.Key, Compare(profile, g.ToList())))
                .OrderByDescending(s => s.Report.MaxPsi)
                .ToList();
        }

        var top = report.Features.FirstOrDefault();
        var message = top is null
            ? $"No profiled feature to compare for {experimentId}."
            : $"{Describe(report.Level)} over {entries.Count} predictions of {experimentId}: most moved '{top.Column}' "
              + $"(PSI {top.Psi:F3}; {top.MostChangedBin}: {top.ReferenceShare:P0} → {top.CurrentShare:P0}).";
        if (report.Prediction is { } answers)
            message += $" Answers: {Describe(answers.Level).ToLowerInvariant()} (PSI {answers.Psi:F3}; "
                       + $"{answers.MostChangedBin}: {answers.ReferenceShare:P0} → {answers.CurrentShare:P0}).";
        return new DriftCheck(DriftCheckStatus.Measured, experimentId, entries.Count, report, slices, message);
    }

    private sealed record Logged(IReadOnlyDictionary<string, object?> Input, object? Output);

    private static DriftReport Compare(DriftProfile profile, IReadOnlyCollection<Logged> entries)
        => DriftMeasure.Compare(profile, entries.Select(e => e.Input).ToList(), entries.Select(e => e.Output).ToList());

    /// <summary>
    /// The drift signal for machines — one shape for <c>mloop trigger check --json</c>, <c>mloop drift
    /// --json</c> and the REST API: the window size, the inputs' level, every column's PSI (most moved
    /// first), and the answers' drift — <c>null</c> when the model was trained without an answer profile.
    /// </summary>
    public static object Signal(DriftReport report) => new
    {
        rows = report.Rows,
        level = report.Level.ToString(),
        maxPsi = report.MaxPsi,
        features = report.Features.Select(Signal),
        prediction = report.Prediction is { } p ? Signal(p) : null,
    };

    private static object Signal(FeatureDrift f) => new
    {
        column = f.Column,
        psi = f.Psi,
        level = f.Level.ToString(),
        mostChangedBin = f.MostChangedBin,
        referenceShare = f.ReferenceShare,
        currentShare = f.CurrentShare,
    };

    private static string Describe(DriftLevel level) => level switch
    {
        DriftLevel.Significant => "Significant drift",
        DriftLevel.Moderate => "Moderate drift",
        _ => "No meaningful drift",
    };

    private static string Text(object? value) => value switch
    {
        null => "(missing)",
        JsonElement { ValueKind: JsonValueKind.String } j => j.GetString() ?? "(missing)",
        JsonElement j => j.GetRawText(),
        _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "(missing)",
    };
}
