namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The one place a failure written to stderr is announced to whichever machine-readable stream the
/// running command owns.
/// </summary>
/// <remarks>
/// <para>
/// There are two such streams and they are chosen per command: <see cref="MachineOutputScope"/> for a
/// command that emits an event stream (<c>train</c>), and <see cref="JsonOutputScope"/> for one that
/// emits a single document. A diagnostics sink has no business knowing which is active, and — the
/// reason this type exists — no business remembering to tell both.
/// </para>
/// <para>
/// It was written after exactly that was forgotten. <see cref="JsonOutputScope"/> was wired into
/// <see cref="ErrorConsole"/> only, while <see cref="ErrorSuggestions"/> announced to the event scope
/// directly from two of its own call sites — the two that every command's outermost <c>catch</c> funnels
/// through. So an unexpected exception in <c>--json</c> mode, the most ordinary failure a command has,
/// reported to one stream and left the other with an empty stdout: the precise defect the guarantee was
/// added to close. Pairing the two calls by hand at each site is what failed; there is one call now.
/// </para>
/// </remarks>
public static class FailureReport
{
    /// <summary>
    /// Announces <paramref name="message"/> to every machine-readable stream. Markup is the caller's
    /// natural form and each scope strips it for its own stream. Inactive scopes ignore the report, so
    /// this is a no-op in ordinary human mode.
    /// </summary>
    public static void Report(string message)
    {
        MachineOutputScope.ReportError(message);
        JsonOutputScope.ReportError(message);
    }
}
