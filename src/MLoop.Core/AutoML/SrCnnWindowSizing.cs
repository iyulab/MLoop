namespace MLoop.Core.AutoML;

/// <summary>
/// Window parameters for ML.NET's SR-CNN time-series anomaly detector.
///
/// This exists as its own type because the sizes are not independent — ML.NET
/// rejects the estimator outright unless the judgement window fits inside the
/// analysis window — and the relation was previously expressed by one computed
/// value sitting next to four literals. The judgement size was a constant 21
/// while the window was derived from the row count, so for any series shorter
/// than 84 rows the detector threw
/// "Must be non-negative and not larger than window size (Parameter
/// 'JudgementWindowSize')" on every single call.
///
/// The consequence was quiet rather than loud: the caller catches a failing
/// detector and moves to the next one in its list, so SR-CNN — the detector the
/// caller itself describes as the best general-purpose choice — simply never
/// ran below 84 points, and the run reported success with a different
/// algorithm. Nothing failed; the better answer was just never available.
/// </summary>
public readonly record struct SrCnnWindowSizes(
    int WindowSize,
    int BackAddWindowSize,
    int LookaheadWindowSize,
    int AveragingWindowSize,
    int JudgementWindowSize)
{
    /// <summary>ML.NET's own default judgement window, used whenever it fits.</summary>
    public const int PreferredJudgementWindowSize = 21;

    /// <summary>Smallest analysis window worth running the detector over.</summary>
    public const int MinimumWindowSize = 8;

    /// <summary>Largest analysis window; beyond this the detector gains nothing.</summary>
    public const int MaximumWindowSize = 64;

    /// <summary>
    /// Derive a self-consistent set of windows for a series of
    /// <paramref name="totalRows"/> points.
    ///
    /// The analysis window is a quarter of the series, clamped; every other
    /// window is then held inside it. The judgement window keeps ML.NET's
    /// preferred 21 whenever the analysis window is at least that wide, and
    /// shrinks to fit rather than throwing when it is not — a smaller judgement
    /// window is a slightly less confident verdict, which is strictly better
    /// than no verdict from this detector at all.
    /// </summary>
    public static SrCnnWindowSizes ForSeries(int totalRows)
    {
        var window = Math.Min(MaximumWindowSize, Math.Max(MinimumWindowSize, totalRows / 4));
        return new SrCnnWindowSizes(
            WindowSize: window,
            // ML.NET's defaults are 5 / 5 / 3; each must also fit the window,
            // which only bites at the smallest sizes.
            BackAddWindowSize: Math.Min(5, window),
            LookaheadWindowSize: Math.Min(5, window),
            AveragingWindowSize: Math.Min(3, window),
            JudgementWindowSize: Math.Min(PreferredJudgementWindowSize, window));
    }
}
