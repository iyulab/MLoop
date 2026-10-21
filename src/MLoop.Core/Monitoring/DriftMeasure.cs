using System.Globalization;
using System.Text.Json;

namespace MLoop.Core.Monitoring;

/// <summary>How far a column has moved, by the usual reading of the population stability index.</summary>
public enum DriftLevel
{
    /// <summary>PSI below 0.1 — no meaningful shift.</summary>
    None = 0,

    /// <summary>PSI from 0.1 to 0.25 — a shift worth watching.</summary>
    Moderate = 1,

    /// <summary>PSI of 0.25 or more — the inputs are not the ones the model was trained on.</summary>
    Significant = 2,
}

/// <summary>One column's drift: its PSI against the training profile, and the bin that moved most.</summary>
public sealed record FeatureDrift(
    string Column,
    double Psi,
    DriftLevel Level,
    string MostChangedBin,
    double ReferenceShare,
    double CurrentShare);

/// <summary>
/// A window of inputs compared against a training profile, and — when the profile carries one — the
/// model's answers over the same window compared against its answers at training.
/// </summary>
/// <param name="Rows">Rows in the window.</param>
/// <param name="Features">
/// One entry per profiled input column, most moved first — a free-text column as two
/// (<see cref="TextProfile.LengthColumn"/>, <see cref="TextProfile.UnseenColumn"/>).
/// </param>
/// <param name="Prediction">
/// The answers' drift (<see cref="DriftProfile.PredictionColumn"/>); <c>null</c> when the profile has
/// no answer profile. Not part of <see cref="MaxPsi"/> and <see cref="Level"/>: those describe the
/// inputs, and an answer shift with steady inputs is a different finding — the model's world changed
/// where it cannot see, or the mix of what it is asked moved inside its inputs' ranges.
/// </param>
public sealed record DriftReport(int Rows, IReadOnlyList<FeatureDrift> Features, FeatureDrift? Prediction = null)
{
    /// <summary>The largest PSI of any input column; 0 with none.</summary>
    public double MaxPsi => Features.Count == 0 ? 0 : Features.Max(f => f.Psi);

    /// <summary>The level of the most moved input column.</summary>
    public DriftLevel Level => DriftMeasure.LevelOf(MaxPsi);
}

/// <summary>
/// The population stability index: Σ (current − reference) · ln(current ÷ reference) over the
/// profile's bins. Symmetric, zero for an unchanged distribution, and read by convention — under 0.1
/// stable, 0.1–0.25 moderate, 0.25 and above significant. A bin empty on either side is floored at
/// <see cref="Floor"/> so a category that appears or vanishes counts large but finite.
/// </summary>
public static class DriftMeasure
{
    /// <summary>The share an empty bin is counted at.</summary>
    public const double Floor = 1e-4;

    /// <summary>PSI from 0.1 is a moderate shift.</summary>
    public const double ModerateThreshold = 0.1;

    /// <summary>PSI from 0.25 is a significant shift.</summary>
    public const double SignificantThreshold = 0.25;

    /// <summary>The level PSI <paramref name="psi"/> reads as.</summary>
    public static DriftLevel LevelOf(double psi)
        => psi >= SignificantThreshold ? DriftLevel.Significant
            : psi >= ModerateThreshold ? DriftLevel.Moderate
            : DriftLevel.None;

    /// <summary>PSI between two share vectors over the same bins.</summary>
    public static double Psi(IReadOnlyList<double> reference, IReadOnlyList<double> current)
    {
        if (reference.Count != current.Count)
            throw new ArgumentException("Both distributions must be over the same bins.", nameof(current));
        double psi = 0;
        for (var i = 0; i < reference.Count; i++)
        {
            var e = Math.Max(reference[i], Floor);
            var a = Math.Max(current[i], Floor);
            psi += (a - e) * Math.Log(a / e);
        }
        return psi;
    }

    /// <summary>
    /// Compares <paramref name="rows"/> (column → value, as a prediction log records inputs) with
    /// <paramref name="profile"/>, column by column. A column a row does not carry counts as missing in it.
    /// </summary>
    /// <param name="profile">The training profile.</param>
    /// <param name="rows">The window's inputs.</param>
    /// <param name="answers">
    /// The answer logged for each row, in the same order (<see cref="Prediction.PredictionLogOutput"/>);
    /// compared with <see cref="DriftProfile.Prediction"/> when both are present.
    /// </param>
    public static DriftReport Compare(
        DriftProfile profile, IReadOnlyCollection<IReadOnlyDictionary<string, object?>> rows, IReadOnlyCollection<object?>? answers = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(rows);
        var features = profile.Features
            .Select(feature => Measure(feature, rows.Select(r => ValueOf(r, feature.Column)).ToList()))
            .Concat((profile.Texts ?? []).SelectMany(text =>
            {
                var (lengths, unseen) = text.Describe(rows.Select(r => ValueOf(r, text.Column)).ToList());
                return new[] { Measure(text.Length, lengths), Measure(text.Unseen, unseen) };
            }))
            .OrderByDescending(f => f.Psi)
            .ToList();
        var prediction = profile.Prediction is { } reference && answers is not null
            ? Measure(reference, answers.Select(Text).ToList())
            : null;
        return new DriftReport(rows.Count, features, prediction);
    }

    private static FeatureDrift Measure(FeatureProfile feature, IReadOnlyCollection<string?> values)
    {
        var current = feature.SharesOf(values);
        var psi = Psi(feature.Shares, current);
        var moved = Enumerable.Range(0, feature.BinCount).MaxBy(i => Math.Abs(current[i] - feature.Shares[i]));
        return new FeatureDrift(feature.Column, psi, LevelOf(psi), BinName(feature, moved), feature.Shares[moved], current[moved]);
    }

    /// <summary>A readable name for bin <paramref name="bin"/> of <paramref name="feature"/>: a category, a range, "other" or "missing".</summary>
    public static string BinName(FeatureProfile feature, int bin)
    {
        if (bin == feature.BinCount - 1)
            return "(missing)";
        if (!feature.IsNumeric)
            return bin < (feature.Categories?.Count ?? 0) ? feature.Categories![bin] : "(other)";
        var edges = feature.Edges ?? [];
        static string F(double x) => x.ToString("G6", CultureInfo.InvariantCulture);
        return edges.Count == 0 ? "(all values)"
            : bin == 0 ? $"≤ {F(edges[0])}"
            : bin == edges.Count ? $"> {F(edges[^1])}"
            : $"({F(edges[bin - 1])}, {F(edges[bin])}]";
    }

    private static string? ValueOf(IReadOnlyDictionary<string, object?> row, string column)
    {
        if (!row.TryGetValue(column, out var value))
        {
            var match = row.Keys.FirstOrDefault(k => string.Equals(k, column, StringComparison.OrdinalIgnoreCase));
            if (match is null)
                return null;
            value = row[match];
        }
        return Text(value);
    }

    /// <summary>A raw value as the text a profile bins — the same for a CSV field, a JSON log value and a CLR value.</summary>
    internal static string? Text(object? value)
        => value switch
        {
            null => null,
            string s => s,
            JsonElement { ValueKind: JsonValueKind.String } j => j.GetString(),
            JsonElement { ValueKind: JsonValueKind.Null or JsonValueKind.Undefined } => null,
            JsonElement { ValueKind: JsonValueKind.True } => "True",
            JsonElement { ValueKind: JsonValueKind.False } => "False",
            JsonElement j => j.GetRawText(),
            bool b => b ? "True" : "False",
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString(),
        };
}
