using MLoop.Search;

namespace MLoop.Core.Training;

/// <summary>
/// How a run asked for a blend (<see cref="Models.TrainingConfig.Ensemble"/>) keeps to its time budget: the trainer search is given
/// a share of it, and the members are fitted on the whole partition only if the estimated time of fitting them all fits in
/// what is left. A run never spends time it was not given on a blend — the single best model is the answer instead.
/// </summary>
internal static class EnsembleBudget
{
    /// <summary>The share of the time budget the trainer search is given when a blend is asked for.</summary>
    public const double SearchShare = 0.7;

    /// <summary>
    /// Why the members cannot be fitted in the time left, or <see langword="null"/> when they can. The estimates are the
    /// search's own for refitting a trial on the whole partition (<see cref="RetainedTrial.FitSeconds"/>), summed — the
    /// members are fitted one after another.
    /// </summary>
    public static string? Shortfall(IReadOnlyList<RetainedTrial> members, TimeSpan left)
    {
        var needed = TimeSpan.FromSeconds(members.Sum(m => m.FitSeconds));
        return needed > left
            ? $"fitting the {members.Count} members needs about {needed.TotalSeconds:0} s and {Math.Max(left.TotalSeconds, 0):0} s of the time budget are left"
            : null;
    }
}
