using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// What a project's scripts — hooks, metric scripts — say, on the console: their warnings through the
/// warning seam, the rest through <see cref="CoreNarration"/>, the console a <c>--json</c> command's
/// scope moves to stderr.
/// </summary>
/// <remarks>
/// It wrote with <c>Console.WriteLine</c> before, which no scope moves: a hook's or a metric script's
/// <c>ctx.Logger.Info</c> landed on stdout ahead of the document, and <c>mloop evaluate --json</c> could
/// not be parsed.
/// </remarks>
public sealed class ScriptConsoleLogger : ILogger
{
    public void Debug(string message) { }
    public void Info(string message) => CoreNarration.Write(message);
    public void Warning(string message) => WarningConsole.Warn(Markup.Escape(message));
    public void Error(string message) => CoreNarration.Write($"[Error] {message}");
    public void Error(string message, Exception exception)
    {
        CoreNarration.Write($"[Error] {message}");
        CoreNarration.Write($"  {exception.Message}");
    }
}
