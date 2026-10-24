using System.Globalization;

namespace MLoop.Core.Training;

/// <summary>
/// The class-balancing rule behind <c>--balance</c>: which classes to oversample and by how many examples, for any
/// kind of example — a CSV row (<c>DataBalancer</c>) or an image path (image classification, which holds its own
/// examples). One rule, so a ratio means the same thing whatever the task reads.
/// </summary>
/// <remarks>
/// Balancing applies to a training partition only. Copies of a minority example on both sides of a split score the
/// model on what it trained on.
/// </remarks>
public static class ClassBalance
{
    /// <summary>Target ratio for <c>auto</c> (10:1).</summary>
    public const double DefaultTargetRatio = 10.0;

    /// <summary>Imbalance ratio above which <c>auto</c> balances.</summary>
    public const double AutoTriggerThreshold = 10.0;

    /// <summary>The seed every balance draws its copies with, so a run repeats.</summary>
    public const int Seed = 42;

    /// <summary>
    /// Why <paramref name="balanceOption"/> is not a balance option — <c>auto</c>, <c>none</c> or a positive ratio — or
    /// null when it is (absent included). Checked before anything runs, so a misspelt option refuses rather than
    /// training unbalanced.
    /// </summary>
    public static string? OptionError(string? balanceOption) =>
        string.IsNullOrWhiteSpace(balanceOption)
        || balanceOption.Equals("none", StringComparison.OrdinalIgnoreCase)
        || balanceOption.Equals("auto", StringComparison.OrdinalIgnoreCase)
        || (double.TryParse(balanceOption, NumberStyles.Float, CultureInfo.InvariantCulture, out var ratio) && ratio > 0)
            ? null
            : $"Invalid balance option: '{balanceOption}'. Use 'auto', 'none', or a number (e.g., '5' for 5:1)";

    /// <summary>Whether <paramref name="balanceOption"/> asks for balancing at all.</summary>
    public static bool Requested(string? balanceOption) =>
        !string.IsNullOrWhiteSpace(balanceOption) && !balanceOption.Equals("none", StringComparison.OrdinalIgnoreCase);

    /// <summary>What balancing did, or why it did nothing.</summary>
    /// <param name="Copies">The added examples, drawn from the classes below the target; empty when nothing applied.</param>
    /// <param name="Message">One line for the user — what was oversampled, or why nothing was.</param>
    /// <param name="OriginalMinorityCount">Examples of the smallest class before balancing.</param>
    /// <param name="NewMinorityCount">Examples of the smallest class after balancing (the target count when applied).</param>
    /// <param name="OriginalRatio">Largest class ÷ smallest class before balancing.</param>
    /// <param name="NewRatio">The same ratio after balancing.</param>
    public sealed record Outcome<T>(
        IReadOnlyList<T> Copies, string Message, int OriginalMinorityCount, int NewMinorityCount,
        double OriginalRatio, double NewRatio)
    {
        /// <summary>Whether any example was added.</summary>
        public bool Applied => Copies.Count > 0;
    }

    /// <summary>
    /// Oversamples the classes of <paramref name="byClass"/> (class → its examples, in reading order) that fall below
    /// the target: every class short of <c>⌈largest ÷ ratio⌉</c> gains copies drawn at random from its own examples
    /// (<see cref="Seed"/>). <c>auto</c> balances to <see cref="DefaultTargetRatio"/> only above
    /// <see cref="AutoTriggerThreshold"/>.
    /// </summary>
    public static Outcome<T> Oversample<T>(IReadOnlyList<KeyValuePair<string, IReadOnlyList<T>>> byClass, string balanceOption)
    {
        ArgumentNullException.ThrowIfNull(byClass);
        if (byClass.Count < 2)
            return Nothing<T>("Only one class found, no balancing needed");

        // Stable: classes of equal size keep their reading order.
        var sorted = byClass.OrderByDescending(c => c.Value.Count).ToList();
        var majority = sorted[0];
        var minority = sorted[^1];
        var originalRatio = (double)majority.Value.Count / minority.Value.Count;

        double targetRatio;
        if (balanceOption.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            if (originalRatio <= AutoTriggerThreshold)
                return Nothing<T>($"Class ratio {originalRatio:F1}:1 is within acceptable range (threshold: {AutoTriggerThreshold}:1)",
                    minority.Value.Count, originalRatio);
            targetRatio = DefaultTargetRatio;
        }
        else if (double.TryParse(balanceOption, NumberStyles.Float, CultureInfo.InvariantCulture, out var customRatio) && customRatio > 0)
        {
            targetRatio = customRatio;
        }
        else
        {
            return Nothing<T>(OptionError(balanceOption) ?? $"Invalid balance option: '{balanceOption}'", minority.Value.Count, originalRatio);
        }

        if (originalRatio <= targetRatio)
            return Nothing<T>($"Class ratio {originalRatio:F1}:1 already meets target ratio {targetRatio}:1",
                minority.Value.Count, originalRatio);

        var targetMinorityCount = (int)Math.Ceiling(majority.Value.Count / targetRatio);
        var random = new Random(Seed);
        var copies = new List<T>();
        var oversampled = new List<string>();
        foreach (var cls in sorted)
        {
            if (cls.Key == majority.Key) continue;
            var toAdd = targetMinorityCount - cls.Value.Count;
            if (toAdd <= 0) continue;
            for (var i = 0; i < toAdd; i++)
                copies.Add(cls.Value[random.Next(cls.Value.Count)]);
            oversampled.Add($"'{cls.Key}' ({cls.Value.Count} → {cls.Value.Count + toAdd})");
        }

        if (copies.Count == 0)
            return Nothing<T>("No oversampling needed", minority.Value.Count, originalRatio);

        var newRatio = (double)majority.Value.Count / targetMinorityCount;
        return new Outcome<T>(copies,
            $"Oversampled {oversampled.Count} class(es): {string.Join(", ", oversampled)} (ratio: {originalRatio:F1}:1 → {newRatio:F1}:1)",
            minority.Value.Count, targetMinorityCount, originalRatio, newRatio);
    }

    private static Outcome<T> Nothing<T>(string message, int minorityCount = 0, double ratio = 0) =>
        new([], message, minorityCount, minorityCount, ratio, ratio);
}
