namespace MLoop.Core.Prediction;

/// <summary>
/// The score a detected box needs to be reported, and the lowest score a detector is trained to keep.
/// </summary>
/// <remarks>
/// The trainer drops every box below its score threshold when it scores, and the threshold used to be its default, 0.5, for
/// everything: the model a user got, the mAP the quality gate judged, the boxes <c>predict</c> printed. A detector can rank boxes
/// well and never be 0.5 confident (a focal-loss detector is trained so that hard objects score low): on a noisy 60-image set one
/// scored <c>map_50</c> 0.06 after 10 epochs and 0.27 after 30 at a threshold of 0.05, and exactly 0 at 0.5 after both. A zero
/// that comes from the threshold looks the same as a zero that comes from learning nothing.
/// So a detector is now trained to keep every box down to <see cref="Floor"/>, which gives mAP over the whole ranking
/// (<c>map_50_ranked</c>), and the boxes a user is shown are cut at an operating threshold, <see cref="Default"/> unless asked
/// otherwise (<c>map_50</c> is scored at it). A model trained before this keeps only boxes of 0.5 and over, so a lower
/// threshold shows no more of it.
/// </remarks>
public static class DetectionScoreThreshold
{
    /// <summary>The confidence a box needs to be reported unless the caller asks for another.</summary>
    public const double Default = 0.5;

    /// <summary>The lowest confidence a detector is trained to keep; a lower operating threshold shows nothing more.</summary>
    public const double Floor = 0.05;

    /// <summary>True for a threshold that can select anything: above 0 and at most 1.</summary>
    public static bool IsValid(double threshold) => threshold > 0 && threshold <= 1 && double.IsFinite(threshold);
}
