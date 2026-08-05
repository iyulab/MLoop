using System.Text.Json.Serialization;

namespace MLoop.Core.Models;

/// <summary>
/// What a run trained, in parts: the trainer's name, the hyperparameters that identify this
/// particular configuration of it, and why it is a fallback if it is one.
/// </summary>
/// <remarks>
/// <para>
/// MLoop reported all of that as one string — <c>KMeans (k=3)</c>,
/// <c>SsaForecasting (window=7, horizon=3)</c>,
/// <c>SdcaLogisticRegression [manual fallback: AutoML AUC failure]</c> — built by concatenation at
/// each task path. Consumers read that field as a trainer identifier (a downstream app stamps it
/// into provenance), so every one of them had to parse a format that was never specified and
/// differed per path.
/// </para>
/// <para>
/// The parts exist at the moment the string is built; folding them together is the lossy step, and
/// it cannot be undone downstream. So the parts are what a run carries, and
/// <see cref="Display"/> composes them in one place instead of sixteen. The composition reproduces
/// every string MLoop already emitted, so nothing a user or a stored experiment sees changes —
/// what changes is that the parts are also available without parsing.
/// </para>
/// </remarks>
public sealed record TrainerDescriptor
{
    /// <summary>
    /// The trainer, or the pipeline AutoML assembled (<c>…=&gt;LightGbmRegression</c>). Nothing but
    /// the name — no parameters, no fallback note.
    /// </summary>
    /// <remarks>
    /// A blank name becomes <c>(unknown)</c> here rather than at each call site: AutoML can report a
    /// trial with no trainer name, and a blank rendered into a progress line or a leaderboard row
    /// reads as missing data instead of as an unnamed trainer.
    /// </remarks>
    public required string Name
    {
        get => _name;
        init => _name = string.IsNullOrWhiteSpace(value) ? "(unknown)" : value;
    }

    private readonly string _name = "(unknown)";

    /// <summary>
    /// The hyperparameters that distinguish this configuration from another run of the same
    /// trainer, in the order they should be shown. Null when the path has none to name — not an
    /// empty map, which would claim the trainer was configured with nothing.
    /// </summary>
    [JsonIgnore]
    public IReadOnlyList<KeyValuePair<string, string>>? Parameters { get; init; }

    /// <summary>
    /// <see cref="Parameters"/> as it is written and read: one JSON object, under the same name in
    /// every artifact that carries it (the <c>result</c> event, <c>metadata.json</c>).
    /// </summary>
    /// <remarks>
    /// The in-memory form is an ordered list because <see cref="Display"/> renders the parameters in
    /// the order the task path names them (<c>window=7, horizon=3</c>), and a dictionary does not
    /// promise that order. The wire form is an object because that is what a consumer reads a
    /// parameter map as. Projecting between them here is what keeps the two artifacts from
    /// describing the same fact in two shapes.
    /// </remarks>
    [JsonPropertyName("params")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public IReadOnlyDictionary<string, string>? Params
    {
        get => Parameters is { Count: > 0 }
            ? Parameters.ToDictionary(p => p.Key, p => p.Value)
            : null;
        init => Parameters = value is null ? null : [.. value];
    }

    /// <summary>
    /// Why this trainer stood in for the one the run intended, e.g.
    /// <c>manual fallback: AutoML AUC failure</c>. Null on the ordinary path — a run that did what
    /// it meant to do has no reason to give.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FallbackReason { get; init; }

    /// <summary>
    /// The human-facing form: <c>Name (k=v, k=v) [reason]</c>, with each part omitted when absent.
    /// The only place these parts are joined.
    /// </summary>
    /// <remarks>
    /// Never serialized. Where a stored artifact needs the rendering it already has one — an
    /// experiment's <c>result.bestTrainer</c>, the <c>result</c> event's <c>bestTrainer</c> — and
    /// writing it here too would put the same string in a file twice, which is how a display form
    /// and its parts drift apart.
    /// </remarks>
    [JsonIgnore]
    public string Display
    {
        get
        {
            var text = Name;
            if (Parameters is { Count: > 0 })
                text += $" ({string.Join(", ", Parameters.Select(p => $"{p.Key}={p.Value}"))})";
            if (!string.IsNullOrEmpty(FallbackReason))
                text += $" [{FallbackReason}]";
            return text;
        }
    }

    /// <summary>A trainer with nothing to qualify it.</summary>
    public static TrainerDescriptor Of(string name) => new() { Name = name };

    /// <summary>A trainer identified by its hyperparameters.</summary>
    public static TrainerDescriptor Of(string name, params (string Key, object Value)[] parameters)
        => new()
        {
            Name = name,
            Parameters = [.. parameters.Select(p =>
                new KeyValuePair<string, string>(p.Key, Convert.ToString(p.Value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty))]
        };
}
