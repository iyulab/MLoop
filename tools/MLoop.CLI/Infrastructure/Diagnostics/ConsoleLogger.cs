using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The logger this CLI hands to <c>MLoop.Core</c> and to a project's scripts — hooks, metric scripts,
/// preprocessing: information through <see cref="CoreNarration"/> (the console a <c>--json</c> command's
/// scope moves to stderr), warnings through the warning seam (so a machine stream gets its
/// <c>warning</c> event), errors to stderr, debug detail nowhere.
/// </summary>
/// <remarks>
/// <para>
/// Seven commands had a logger of their own, and they disagreed: two printed straight to the console, so
/// their warnings never became <c>warning</c> events; two passed messages into Spectre markup unescaped,
/// so a message with a bracket in it threw instead of printing; one showed debug detail that the others
/// hid. One logger, one set of rules.
/// </para>
/// <para>
/// Errors go to stderr whatever the mode: they surface while a command may be streaming to stdout, and a
/// subprocess consumer reading stderr on a non-zero exit must find the cause there.
/// </para>
/// </remarks>
public sealed class ConsoleLogger : ILogger
{
    public void Debug(string message) { }
    public void Info(string message) => CoreNarration.Write(message);
    public void Warning(string message) => WarningConsole.Warn(Markup.Escape(message));
    public void Error(string message) => ErrorConsole.Out.MarkupLine($"[red]{Markup.Escape(message)}[/]");
    public void Error(string message, Exception exception)
    {
        Error(message);
        ErrorConsole.Out.WriteException(exception);
    }
}

/// <summary>A logger that says nothing — for a step whose output the command reports another way.</summary>
public sealed class SilentLogger : ILogger
{
    public static SilentLogger Instance { get; } = new();

    private SilentLogger() { }

    public void Debug(string message) { }
    public void Info(string message) { }
    public void Warning(string message) { }
    public void Error(string message) { }
    public void Error(string message, Exception exception) { }
}
