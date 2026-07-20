namespace MLoop.Core.AutoML;

/// <summary>
/// Thrown when an AutoML experiment ends without a single completed trial — ML.NET reports this as a
/// <see cref="TimeoutException"/> ("Training time finished without completing a successful trial"),
/// wrapped in an <see cref="AggregateException"/> by its internal threading.
/// </summary>
/// <remarks>
/// A dedicated type exists because the wrapped message is an <b>ML.NET</b> string, not ours: matching it
/// textually would break on localization or an ML.NET wording change, exactly the fragility that made
/// keyword-matching "bad allocation" the wrong fix for this failure class (cycle-173). Callers that need
/// to recognise the condition — the CLI's suggestion layer above all — test the type instead.
/// <para>
/// The message deliberately names <b>two</b> possible causes. In this code path MLoop genuinely cannot
/// tell them apart: when no trial completes, AutoML's <c>progressHandler</c> reports nothing at all
/// (measured, cycle-173), so a budget that was merely too small and trials that all died on resources
/// look identical from here. Naming only one would be a guess presented as a diagnosis.
/// </para>
/// Derives from <see cref="InvalidOperationException"/> so existing handlers keep working — the same
/// reasoning as <c>DegeneratePredictionException</c>.
/// </remarks>
public sealed class NoSuccessfulTrialException : InvalidOperationException
{
    /// <summary>The experiment's time budget in seconds, so callers can echo it back to the user.</summary>
    public int TimeLimitSeconds { get; }

    public NoSuccessfulTrialException(int timeLimitSeconds, Exception? innerException)
        : base(BuildMessage(timeLimitSeconds), innerException)
    {
        TimeLimitSeconds = timeLimitSeconds;
    }

    private static string BuildMessage(int timeLimitSeconds) =>
        $"AutoML finished without completing a single successful trial (time budget: {timeLimitSeconds}s). " +
        "Either the budget was too small for this dataset, or every trial failed before finishing " +
        "(most often memory exhaustion on wide or large data).";
}
