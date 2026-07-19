using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The stderr sink for failure diagnostics — the single place that owns the
/// "<c>exit != 0</c> ⇒ stderr is not empty" contract.
/// <para>
/// mloop already established this rule for the update notice (D19: machine-readable stdout such as
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
    public static IAnsiConsole Out => AnsiConsole.Create(new AnsiConsoleSettings
    {
        Out = new AnsiConsoleOutput(Console.Error)
    });

    /// <summary>
    /// Writes an <c>Error:</c> line to stderr. <paramref name="markup"/> is Spectre markup — escape
    /// interpolated user data with <see cref="Markup.Escape"/>.
    /// </summary>
    public static void Error(string markup) => Out.MarkupLine($"[red]Error:[/] {markup}");

    /// <summary>
    /// Writes a <c>Tip:</c> line to stderr. Tips accompany an error as part of the cause a machine
    /// consumer persists, so they belong on the same channel as the error itself.
    /// </summary>
    public static void Tip(string markup) => Out.MarkupLine($"[yellow]Tip:[/] {markup}");
}
