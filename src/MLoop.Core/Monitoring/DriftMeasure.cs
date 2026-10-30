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
/// <param name="Column">The column.</param>
/// <param name="Psi">Its PSI against the training profile (the window's shares counted with half a row more in every bin).</param>
/// <param name="Level">How far it has moved, read against <paramref name="Bound"/> as well as 0.1 and 0.25.</param>
/// <param name="MostChangedBin">The bin that moved most.</param>
/// <param name="ReferenceShare">That bin's share of the training rows.</param>
/// <param name="CurrentShare">That bin's share of the window.</param>
/// <param name="Bound">
/// The PSI a window of this size would stay under, in the column's family of columns, when nothing had changed (<see cref="DriftMeasure.NoDriftBound"/>) —
/// the level is read against it as well as against the usual 0.1 and 0.25. 0 when none was worked out.
/// </param>
public sealed record FeatureDrift(
    string Column,
    double Psi,
    DriftLevel Level,
    string MostChangedBin,
    double ReferenceShare,
    double CurrentShare,
    double Bound = 0);

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
    public DriftLevel Level => Features.Count == 0 ? DriftLevel.None : Features.Max(f => f.Level);
}

/// <summary>
/// The population stability index: Σ (current − reference) · ln(current ÷ reference) over the
/// profile's bins. Symmetric, zero for an unchanged distribution, and read by convention — under 0.1
/// stable, 0.1–0.25 moderate, 0.25 and above significant. A bin empty on either side is floored at
/// <see cref="Floor"/> so a category that appears or vanishes counts large but finite.
/// </summary>
/// <remarks>
/// Those readings are for large samples. A window of n rows against a profile of N rows, with nothing changed, has a PSI of about
/// χ²<sub>bins − 1</sub> × (1 ÷ n + 1 ÷ N) — 0.03 at three hundred rows — and a column holds a bin that expects a handful of rows
/// (the missing values') where a window with none of them reads 0.1 on that bin alone. Read against 0.1 flat, a window of three hundred rows
/// with nothing changed reads "moderate" one time in five at fifty columns and nearly always at a thousand. So the level is read against the
/// usual 0.1 and 0.25 <em>and</em> against what chance alone gives across the profile's columns (<see cref="NoDriftBound"/>), the window's
/// shares are counted with half a row more in every bin, and the result is the same for any width of table.
/// </remarks>
public static class DriftMeasure
{
    /// <summary>The share an empty bin is counted at.</summary>
    public const double Floor = 1e-4;

    /// <summary>PSI from 0.1 is a moderate shift.</summary>
    public const double ModerateThreshold = 0.1;

    /// <summary>PSI from 0.25 is a significant shift.</summary>
    public const double SignificantThreshold = 0.25;

    /// <summary>The chance, over all the profile's columns together, that a window with nothing changed reads over its <see cref="NoDriftBound"/> in some column.</summary>
    public const double FamilyErrorRate = 0.05;

    /// <summary>The level PSI <paramref name="psi"/> reads as, against the usual thresholds alone.</summary>
    public static DriftLevel LevelOf(double psi) => LevelOf(psi, 0);

    /// <summary>
    /// The level PSI <paramref name="psi"/> reads as: "moderate" from the larger of 0.1 and <paramref name="bound"/>, "significant" from the
    /// larger of 0.25 and <paramref name="bound"/>.
    /// </summary>
    public static DriftLevel LevelOf(double psi, double bound)
        => psi >= Math.Max(SignificantThreshold, bound) ? DriftLevel.Significant
            : psi >= Math.Max(ModerateThreshold, bound) ? DriftLevel.Moderate
            : DriftLevel.None;

    /// <summary>
    /// The PSI a column stays under, with probability 1 − <see cref="FamilyErrorRate"/> across <paramref name="columns"/> columns all
    /// unchanged: a window of <paramref name="windowRows"/> rows against a profile of <paramref name="referenceRows"/> rows has a PSI of about
    /// χ²(k − 1) × (1 ÷ n + 1 ÷ N), k being the bins the reference occupies; the bound is the quantile of that at
    /// <see cref="FamilyErrorRate"/> ÷ <paramref name="columns"/> (the Bonferroni share of one column).
    /// </summary>
    public static double NoDriftBound(IReadOnlyList<double> reference, int windowRows, int referenceRows, int columns)
    {
        ArgumentNullException.ThrowIfNull(reference);
        if (windowRows < 1)
            return 0;
        var degrees = Math.Max(1, reference.Count(share => share > 0) - 1);
        var tail = FamilyErrorRate / Math.Max(columns, 1);
        var quantile = Quantiles.GetOrAdd((degrees, tail), key => Evaluation.SpecialFunctions.ChiSquareUpperQuantile(key.Item1, key.Item2));
        return quantile * (1.0 / windowRows + 1.0 / Math.Max(referenceRows, 1));
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(int, double), double> Quantiles = new();

    // The window's shares with half a row added to every bin: a bin that expects a handful of rows and gets none is not a
    // distribution that left, and the floor alone counts it as one.
    private static List<double> Smoothed(IReadOnlyList<double> shares, int rows)
        => [.. shares.Select(share => (share * rows + 0.5) / (rows + 0.5 * shares.Count))];

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
        var family = profile.Features.Count + 2 * (profile.Texts?.Count ?? 0);
        var features = profile.Features
            .Select(feature => Measure(feature, rows.Select(r => ValueOf(r, feature.Column)).ToList(), family, profile.Rows))
            .Concat((profile.Texts ?? []).SelectMany(text =>
            {
                var (lengths, unseen) = text.Describe(rows.Select(r => ValueOf(r, text.Column)).ToList());
                return new[] { Measure(text.Length, lengths, family, profile.Rows), Measure(text.Unseen, unseen, family, profile.Rows) };
            }))
            .OrderByDescending(f => f.Psi)
            .ToList();
        var prediction = profile.Prediction is { } reference && answers is not null
            ? Measure(reference, answers.Select(Text).ToList(), 1, profile.Rows)
            : null;
        return new DriftReport(rows.Count, features, prediction);
    }

    private static FeatureDrift Measure(FeatureProfile feature, IReadOnlyCollection<string?> values, int family, int referenceRows)
    {
        var current = feature.SharesOf(values);
        var reference = feature.Shares;
        // A numeric column's end bins are open: rows past the training range land beside the rows that were there. Counted apart — against a
        // reference share of nothing — a window that left the range reads as it should, whatever the training rows did at the edge.
        var (below, above) = feature.OutsideRange(values);
        if (below + above > 0)
        {
            var top = (feature.Edges?.Count ?? 0);
            var shares = current.ToList();
            var expected = reference.ToList();
            shares[0] -= below;
            shares[top] -= above;
            shares.Add(below);
            expected.Add(0);
            shares.Add(above);
            expected.Add(0);
            var extended = Psi(expected, Smoothed(shares, values.Count));
            var extendedBound = NoDriftBound(expected, values.Count, referenceRows, family);
            var movedBin = Enumerable.Range(0, shares.Count).MaxBy(i => Math.Abs(shares[i] - expected[i]));
            if (movedBin < feature.BinCount)
                return new FeatureDrift(feature.Column, extended, LevelOf(extended, extendedBound), BinName(feature, movedBin), expected[movedBin], shares[movedBin], extendedBound);
            static string F(double x) => x.ToString("G6", CultureInfo.InvariantCulture);
            var name = movedBin == feature.BinCount ? $"< {F(feature.Min!.Value)}" : $"> {F(feature.Max!.Value)}";
            return new FeatureDrift(feature.Column, extended, LevelOf(extended, extendedBound), name, 0, shares[movedBin], extendedBound);
        }
        var psi = Psi(reference, Smoothed(current, values.Count));
        var bound = NoDriftBound(reference, values.Count, referenceRows, family);
        var moved = Enumerable.Range(0, feature.BinCount).MaxBy(i => Math.Abs(current[i] - reference[i]));
        return new FeatureDrift(feature.Column, psi, LevelOf(psi, bound), BinName(feature, moved), reference[moved], current[moved], bound);
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
