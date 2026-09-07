using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The channel this CLI hands to <c>MLoop.Core</c> for the lines it narrates — which columns a load
/// dropped, which file it flattened, which trainer it fell back to.
/// </summary>
/// <remarks>
/// <para>
/// Core takes a sink and writes nowhere without one, so this is what decides where its narration
/// lands. It goes through the ambient <see cref="AnsiConsole"/> for one reason: that is the console
/// a command's scope replaces. <see cref="JsonOutputScope"/> points it at stderr for the life of a
/// <c>--json</c> command, and <see cref="MachineOutputScope"/> discards writes while an event stream
/// owns stdout — so routing Core's narration here puts it under the same rule as every line the CLI
/// writes itself, without Core needing to know either scope exists.
/// </para>
/// <para>
/// It reached stdout directly before this. <c>mloop info --json</c> printed
/// <c>[Info] Removed index column(s): Unnamed: 0</c> ahead of its document and exited 0, so a
/// consumer's <c>JSON.parse</c> failed on a command that had reported success; <c>analyze --json</c>
/// did the same. The CLI-side <c>--json</c> audit could not have caught it — the offending write came
/// from a layer underneath the scope, through a console the scope does not own.
/// </para>
/// <para>
/// <see cref="IAnsiConsole.WriteLine(string)"/>, not <c>MarkupLine</c>: these messages carry literal
/// brackets (<c>[Info]</c>, <c>[Warning]</c>) that markup would try to parse.
/// </para>
/// </remarks>
public static class CoreNarration
{
    /// <summary>The sink to pass wherever <c>MLoop.Core</c> takes an <c>Action&lt;string&gt; log</c>.</summary>
    public static void Write(string message) => AnsiConsole.WriteLine(message);

    /// <summary>The same sink as a delegate, for the call sites that pass one along.</summary>
    public static Action<string> Sink => Write;
}
