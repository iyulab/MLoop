namespace MLoop.Core.Training;

/// <summary>
/// Thrown when a trainer search ends without a single completed trial — the search reports it as
/// <c>MLoop.Search.SearchFailedException</c>, which this wraps with the run's time budget.
/// </summary>
/// <remarks>
/// A dedicated type, so callers that need to recognise the condition — the CLI's suggestion layer above
/// all — test the type rather than match a message.
/// <para>
/// The message names <b>two</b> possible causes — a budget too small for one trial, and trials that all
/// failed — because the run here does not tell them apart. Naming only one would be a guess presented as
/// a diagnosis.
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
        $"The trainer search finished without completing a single successful trial (time budget: {timeLimitSeconds}s). " +
        "Either the budget was too small for this dataset, or every trial failed before finishing " +
        "(most often memory exhaustion on wide or large data).";
}
