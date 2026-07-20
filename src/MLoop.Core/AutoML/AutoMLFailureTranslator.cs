namespace MLoop.Core.AutoML;

/// <summary>
/// Turns ML.NET AutoML's terminal exceptions into failures MLoop can diagnose.
/// </summary>
/// <remarks>
/// AutoML runs trials on its own threads, so whatever ended the experiment reaches us wrapped in an
/// <see cref="AggregateException"/> — and a wrapper is opaque to every rule that inspects an exception:
/// its <see cref="Exception.Message"/> reads "One or more errors occurred. (…)" and its type says nothing.
/// Two real causes were hiding behind it (cycle-173, measured against Microsoft.ML.AutoML 0.23.0):
/// <list type="bullet">
/// <item><description><see cref="TimeoutException"/> — no trial completed. Promoted to
/// <see cref="NoSuccessfulTrialException"/> so callers can recognise it without matching ML.NET's
/// English message.</description></item>
/// <item><description><see cref="OutOfMemoryException"/> — unwrapped as-is; the diagnosis for it already
/// existed downstream and only ever needed to *see* the exception.</description></item>
/// </list>
/// Everything else is deliberately left untouched: <c>AutoMLRunner</c> recovers from the AUC-undefined
/// family (BUG-22/24/36) by matching the aggregate itself, and translating it would break that recovery.
/// This is the single place AutoML terminal failures are interpreted — adding a second interpretation
/// site is how metric/layout knowledge drifted across assemblies before (see CLAUDE.md,
/// Single-Source Authorities).
/// </remarks>
public static class AutoMLFailureTranslator
{
    /// <summary>
    /// Recognises an AutoML terminal failure and produces the exception MLoop should surface.
    /// </summary>
    /// <param name="exception">The exception AutoML's <c>Execute</c> threw.</param>
    /// <param name="timeLimitSeconds">The experiment's time budget, echoed back to the user.</param>
    /// <param name="translated">The failure to surface — the input itself when nothing is recognised.</param>
    /// <returns><c>true</c> when <paramref name="translated"/> differs from the input.</returns>
    public static bool TryTranslate(Exception exception, int timeLimitSeconds, out Exception translated)
    {
        ArgumentNullException.ThrowIfNull(exception);

        foreach (var candidate in Unwrap(exception))
        {
            switch (candidate)
            {
                case TimeoutException:
                    translated = new NoSuccessfulTrialException(timeLimitSeconds, exception);
                    return true;

                case OutOfMemoryException:
                    translated = candidate;
                    return true;
            }
        }

        translated = exception;
        return false;
    }

    /// <summary>
    /// The exception itself plus everything nested inside it — AutoML nests both ways (an
    /// <see cref="AggregateException"/> holding siblings, and ordinary inner-exception chains).
    /// </summary>
    private static IEnumerable<Exception> Unwrap(Exception exception)
    {
        yield return exception;

        if (exception is AggregateException aggregate)
        {
            foreach (var inner in aggregate.Flatten().InnerExceptions)
            {
                foreach (var nested in Unwrap(inner))
                    yield return nested;
            }
        }
        else if (exception.InnerException is { } inner)
        {
            foreach (var nested in Unwrap(inner))
                yield return nested;
        }
    }
}
