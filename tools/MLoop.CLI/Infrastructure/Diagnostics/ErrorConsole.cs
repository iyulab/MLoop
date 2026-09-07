using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The stderr sink for failure diagnostics — the single place that owns the
/// "<c>exit != 0</c> ⇒ stderr is not empty" contract.
/// <para>
/// mloop already established this rule for the update notice (machine-readable stdout such as
/// <c>mloop token -q</c> or <c>predict --json</c> must never be polluted by diagnostics), but the
/// rule had only ever been applied to that one notice and to <c>predict --json</c>. Every *domain*
/// error still went to stdout via the default <see cref="AnsiConsole"/>, so a subprocess consumer
/// following the POSIX convention — read stderr when the exit code is non-zero — got an empty
/// reason string while a perfectly good diagnostic sat in stdout. System.CommandLine's own parse
/// errors go to stderr, so the channel silently depended on *which kind* of error occurred.
/// </para>
/// <para>
/// Only the failure lines move here. Human-facing rich output (progress, class-distribution
/// visualizations, tables) stays on stdout: the contract is that a cause exists on stderr, not that
/// stdout goes quiet.
/// </para>
/// </summary>
public static class ErrorConsole
{
    /// <summary>
    /// The stderr console. Prefer <see cref="Error"/>/<see cref="Tip"/> for the common shapes so the
    /// "Error:"/"Tip:" prefixes stay uniform across commands.
    /// <para>
    /// Built per access rather than cached so it always binds to the current
    /// <see cref="Console.Error"/> — a cached instance would pin whichever writer happened to be
    /// installed at first use, silently bypassing any later redirection. Callers that emit several
    /// lines should hold the returned instance in a local. Error paths are rare, so the construction
    /// cost never lands on a hot path.
    /// </para>
    /// </summary>
    public static IAnsiConsole Out
    {
        get
        {
            var console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(Console.Error)
            });
            console.Profile.Width = DiagnosticLineWidth;
            return console;
        }
    }

    /// <summary>
    /// Width this console lays lines out against. Spectre assumes 80 columns when the stream is not
    /// a terminal, and folds a long line at that point — inside a token if the line has no space
    /// there. A diagnostic that names a command to run (<c>Run 'mloop train --name default' first.</c>)
    /// then splits the command across two lines in a log or a captured stream, and a user who copies
    /// it gets a broken command. Nothing written here is a table or a panel that needs a width to lay
    /// out — only lines and exception traces — so the console gets a width no line reaches. A real
    /// terminal still wraps visually at its own edge; the text itself stays one line. Large but
    /// finite: Spectre does arithmetic on the width, and <see cref="int.MaxValue"/> would overflow it.
    /// </summary>
    private const int DiagnosticLineWidth = 4096;

    /// <summary>
    /// Writes an <c>Error:</c> line to stderr. <paramref name="markup"/> is Spectre markup — escape
    /// interpolated user data with <see cref="Markup.Escape"/>.
    /// </summary>
    public static void Error(string markup)
    {
        Out.MarkupLine($"[red]Error:[/] {markup}");

        // A command in machine-readable mode reports the same cause on its own stream, so a consumer
        // reading only that stream is not left with an empty stdout and a non-zero exit.
        FailureReport.Report(markup);
    }

    /// <summary>
    /// Writes an <c>Error:</c> line followed by its <c>Tip:</c> line, and reports the two together as
    /// a <b>single</b> event.
    /// <para>
    /// This is the shape to reach for whenever the tip is part of the cause rather than an aside. A
    /// tip written on its own reaches the terminal only (see <see cref="Tip"/>), so an error whose
    /// guidance lives in a separate <see cref="Tip"/> call tells a machine consumer *what* failed
    /// while withholding *what to do* — the half of the message that resolves the failure. Reporting
    /// one combined event rather than two keeps "one failure ⇒ one error event" intact.
    /// </para>
    /// </summary>
    public static void Error(string markup, string tip)
    {
        var console = Out;
        console.MarkupLine($"[red]Error:[/] {markup}");
        console.MarkupLine($"[yellow]Tip:[/] {tip}");

        FailureReport.Report($"{markup} {tip}");
    }

    /// <summary>
    /// Writes a <c>Tip:</c> line to stderr, for guidance that stands on its own — a second tip after
    /// <see cref="Error(string, string)"/>, or advice not attached to a failure at all.
    /// <para>
    /// <b>Terminal only.</b> A tip does not become an event: the sink reports failures, and a bare
    /// tip has no failure to attach to. When the guidance belongs to an error, pass it to
    /// <see cref="Error(string, string)"/> instead of writing it here, or a <c>--json</c> consumer
    /// silently loses it.
    /// </para>
    /// </summary>
    public static void Tip(string markup) => Out.MarkupLine($"[yellow]Tip:[/] {markup}");

    /// <summary>
    /// The tip for a user-supplied path that resolved to something that doesn't exist, where the
    /// resolution base was the project root (the common case: every explicit-path option is joined
    /// against <paramref name="projectRoot"/> when it isn't already absolute). Centralized so the
    /// wording — and the resolution rule it states — stays one fact instead of drifting per call site.
    /// </summary>
    public static string PathNotFoundTip(string projectRoot) =>
        $"A relative path resolves against the project root ([cyan]{Markup.Escape(projectRoot)}[/]) — pass an absolute path if the file lives elsewhere.";

    /// <summary>
    /// Same as <see cref="PathNotFoundTip(string)"/>, for the one command (<c>sample create --from</c>)
    /// that resolves its explicit path against the current directory instead of the project root.
    /// </summary>
    public static string PathNotFoundTipCwd() =>
        $"A relative path resolves against the current directory ([cyan]{Markup.Escape(Directory.GetCurrentDirectory())}[/]) — pass an absolute path if the file lives elsewhere.";
}
