using System.CommandLine;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The failure report for a command line that never reached a command — an unknown option, an
/// unknown command, a value the parser rejected.
/// </summary>
/// <remarks>
/// <para>
/// Every other structured-output guarantee in this assembly lives inside a command's action:
/// <see cref="JsonOutputScope"/> owns stdout for the scope's lifetime, and
/// <see cref="FailureReport"/> fans a reported failure out to whichever machine streams are open.
/// A parse failure happens <b>before</b> any of that exists — the action is never entered, so no
/// scope is ever constructed — which left it as the one <b>failure</b> exit where a consumer asking
/// for structured output got prose instead. Help and version requests are deliberately not that:
/// they clear the parse errors and never reach here, because output a caller asked for by name is
/// the answer rather than a diagnostic about a failure.
/// </para>
/// <para>
/// Two things were wrong at that exit, and they are separable. The usage text was written to
/// <b>stdout</b>, where a caller redirecting stdout collects it as if it were the command's answer;
/// usage is diagnostic output and belongs on stderr, while stdout stays reserved for what was asked
/// for. And a caller that asked for structured output got an empty stdout once that was corrected,
/// which fails a parser exactly as prose does — the same reasoning <see cref="JsonError"/> records
/// for action-level exits, applied one layer out.
/// </para>
/// <para>
/// The payload is the single-document envelope, never a command's own stream dialect, and that is
/// deliberate: this layer frequently does not know which command was meant (an unknown command name
/// is itself one of the failures it reports), so keying the shape on a command would mean guessing.
/// One envelope at one layer cannot drift; a per-command dialect here would reintroduce exactly the
/// call-site coupling the scopes were built to remove.
/// </para>
/// </remarks>
public static class ParseFailureReport
{
    /// <summary>
    /// Renders the parse failure: usage and the parser's own messages to stderr, and — when the
    /// caller asked for structured output — one JSON document to stdout.
    /// </summary>
    public static int Report(ParseResult parseResult, IReadOnlyList<string> args)
    {
        // The library's own rendering, redirected wholesale. Reusing it keeps the human-facing text
        // (usage, the suggestion list, localization) identical to every other help path instead of
        // reimplementing a second, poorer renderer that would drift from it.
        var exitCode = parseResult.Invoke(new InvocationConfiguration
        {
            Output = Console.Error,
            Error = Console.Error
        });

        if (JsonError.WasRequested(args))
            JsonError.Emit(Describe(parseResult));

        // A parse failure is a failure even if the renderer reports otherwise.
        return exitCode == 0 ? 1 : exitCode;
    }

    /// <summary>
    /// Every parser message joined into one sentence. A single command line can fail several ways at
    /// once — an unknown command plus each of the options that command would have owned — and
    /// reporting only the first would name a cause the caller cannot act on alone.
    /// </summary>
    internal static string Describe(ParseResult parseResult) =>
        string.Join(" ", parseResult.Errors.Select(e => e.Message));
}
