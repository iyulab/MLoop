using System.Text.Json;
using System.Text.Json.Serialization;
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

    /// <summary>
    /// Fewer logged predictions than this, and a PSI says more about chance than about drift. A column is cut in ten bins, and with no change at
    /// all the PSI of a window of n rows is about 9 ÷ n — 0.09 at a hundred rows, so ten columns read "moderate" in nearly every window and
    /// "significant" (the level a retraining trigger acts on) in one window of twelve, thirty columns in one of four. At three hundred rows
    /// the level is read against what chance alone gives across the profile's columns (<see cref="DriftMeasure.NoDriftBound"/>): read against 0.1
    /// flat, fifty columns reached "moderate" one window in five and two hundred one in two; read this way, 2–10 % at any width (simulated: two hundred columns, a thousand windows over
    /// two hundred profiles - about 10 % against a 5,000-row profile, about 4 % against a 500-row one).
    /// </summary>
    public const int MinimumRows = 300;

    /// <summary>
    /// A training profile of fewer rows than this is itself a noisy sample: with nothing changed, a profile of 100 rows read a window as
    /// moved in up to a fifth of the windows (38 of 200 at two hundred columns with 1 % categories; simulated), from 150 rows at most 15
    /// of 200 in any shape tried. The reading is not refused — a small table is the common case — but the answer says how much weight it bears.
    /// </summary>
    public const int SmallProfileRows = 150;

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
                // A task whose model reads no feature columns never has a profile, so retraining would not start anything.
                metadata.Task is { } task && !DriftProfile.IsRecordedFor(task)
                    ? $"The production model ({experimentId}) is {(IsVowel(task[0]) ? "an" : "a")} {task} model: it reads no feature columns, so there is no input to profile and drift is not measured for it."
                    : $"The production model ({experimentId}) has no drift profile — it was trained before profiles were recorded. "
                      + "Retrain and promote to start measuring drift.");
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
        if (top is not null && profile.Rows < SmallProfileRows)
            message += $" The training profile holds only {profile.Rows} rows, so chance alone moves a PSI more than usual: read a \"moderate\" as weak evidence.";
        if (report.Prediction is { } answers)
            message += $" Answers: {Describe(answers.Level).ToLowerInvariant()} (PSI {answers.Psi:F3}; "
                       + $"{answers.MostChangedBin}: {answers.ReferenceShare:P0} → {answers.CurrentShare:P0}).";
        if (DescribeRange(report) is { } range)
            message += " " + range;
        return new DriftCheck(DriftCheckStatus.Measured, experimentId, entries.Count, report, slices, message);
    }

    private sealed record Logged(IReadOnlyDictionary<string, object?> Input, object? Output);

    private static DriftReport Compare(DriftProfile profile, IReadOnlyCollection<Logged> entries)
        => DriftMeasure.Compare(profile, entries.Select(e => e.Input).ToList(), entries.Select(e => e.Output).ToList());

    /// <summary>
    /// The drift signal for machines — one shape for <c>mloop trigger check --json</c>, <c>mloop drift
    /// --json</c> and the REST API: the window size, the inputs' level, every column's PSI (most moved
    /// first) with the values it saw past the training range, the columns where that count is more than chance
    /// gives (<c>rangeNotable</c>, apart from <c>level</c>), and the answers' drift — <c>null</c> when the model
    /// was trained without an answer profile.
    /// </summary>
    public static DriftSignal Signal(DriftReport report) => new(
        report.Rows,
        report.Level.ToString(),
        report.MaxPsi,
        [.. report.RangeNotable.Select(f => f.Column)],
        [.. report.Features.Select(Signal)],
        report.Prediction is { } p ? Signal(p) : null);

    /// <summary>
    /// A sentence naming the columns whose values left the training range by more than chance allows, or <c>null</c> when none did. It is read apart
    /// from the PSI level: the PSI barely sees a share of the values leaving the range.
    /// </summary>
    public static string? DescribeRange(DriftReport report)
    {
        const int Shown = 5;
        var notable = report.RangeNotable;
        if (notable.Count == 0)
            return null;
        var parts = notable.Take(Shown).Select(f =>
        {
            var r = f.Range!;
            var sides = new List<string>();
            if (r.Above > 0)
                sides.Add($"{r.Above:N0} of {r.WindowValues:N0} above the training maximum");
            if (r.Below > 0)
                sides.Add($"{r.Below:N0} of {r.WindowValues:N0} below the training minimum");
            return $"'{f.Column}' ({string.Join(", ", sides)})";
        });
        var more = notable.Count > Shown ? $" and {notable.Count - Shown} more" : "";
        return $"Values past the training range, counted apart from the PSI: {string.Join(", ", parts)}{more}.";
    }

    private static FeatureDriftSignal Signal(FeatureDrift f) => new(
        f.Column,
        f.Psi,
        f.Level.ToString(),
        f.MostChangedBin,
        f.ReferenceShare,
        f.CurrentShare,
        f.Bound,
        f.Range is { } r ? new RangeCountsSignal(r.Below, r.Above, r.WindowValues, r.ReferenceValues, r.P, r.Notable) : null);

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

    private static bool IsVowel(char c) => "aeiouAEIOU".Contains(c);
}

// The drift signal as a client reads it (mloop drift --json, trigger check --json, GET /drift and GET /trigger). The property names
// are fixed on the records, so each host writes them the same way whatever naming policy its serializer has; a null property is
// written or left out as the host's options say.

/// <summary>The whole-window drift: its size, its level, the worst column and what each column and the answers did.</summary>
public sealed record DriftSignal(
    [property: JsonPropertyName("rows")] int Rows,
    [property: JsonPropertyName("level")] string Level,
    [property: JsonPropertyName("maxPsi")] double MaxPsi,
    [property: JsonPropertyName("rangeNotable")] IReadOnlyList<string> RangeNotable,
    [property: JsonPropertyName("features")] IReadOnlyList<FeatureDriftSignal> Features,
    [property: JsonPropertyName("prediction")] FeatureDriftSignal? Prediction);

/// <summary>One column's (or the answers') drift.</summary>
public sealed record FeatureDriftSignal(
    [property: JsonPropertyName("column")] string Column,
    [property: JsonPropertyName("psi")] double Psi,
    [property: JsonPropertyName("level")] string Level,
    [property: JsonPropertyName("mostChangedBin")] string MostChangedBin,
    [property: JsonPropertyName("referenceShare")] double ReferenceShare,
    [property: JsonPropertyName("currentShare")] double CurrentShare,
    [property: JsonPropertyName("noDriftBound")] double NoDriftBound,
    [property: JsonPropertyName("range")] RangeCountsSignal? Range);

/// <summary>How many of a numeric column's values left the range it was trained on.</summary>
public sealed record RangeCountsSignal(
    [property: JsonPropertyName("below")] int Below,
    [property: JsonPropertyName("above")] int Above,
    [property: JsonPropertyName("windowValues")] int WindowValues,
    [property: JsonPropertyName("referenceValues")] int ReferenceValues,
    [property: JsonPropertyName("p")] double P,
    [property: JsonPropertyName("notable")] bool Notable);
