namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// What a failed run says went wrong: its message and, when a library wrapped the real cause, the cause. ML.NET reports a
/// row that throws while its data is read as <c>Splitter/consolidator worker encountered exception while consuming source data</c> —
/// true of every such failure, and the exception that says which row and why sits one or two levels down. A caller of the
/// API sees only the job's message, so the run's failure carries the cause in its own text instead of leaving it in a chain
/// nobody reads. One place, so the CLI, the job record and the experiment record say the same thing.
/// </summary>
public static class FailureCause
{
    /// <summary>The exception's message, with the message of the innermost exception after it when that says something the message does not.</summary>
    public static string Describe(Exception ex)
    {
        var message = ex.Message;
        var root = Innermost(ex);
        if (ReferenceEquals(root, ex) || string.IsNullOrWhiteSpace(root.Message))
            return message;
        return message.Contains(root.Message.Trim(), StringComparison.Ordinal)
            ? message
            : $"{message} — cause: {root.Message.Trim()}";
    }

    private static Exception Innermost(Exception ex)
    {
        var current = ex;
        while (true)
        {
            var next = current is AggregateException aggregate
                ? aggregate.Flatten().InnerExceptions.FirstOrDefault() ?? aggregate.InnerException
                : current.InnerException;
            if (next is null || ReferenceEquals(next, current))
                return current;
            current = next;
        }
    }
}
