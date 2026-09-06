using System.Text.Json;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The <c>--json</c> failure payload — the stdout half of what <see cref="ErrorConsole"/> writes to
/// stderr.
/// <para>
/// A command in <c>--json</c> mode owes its consumer a parseable document at <b>every</b> exit point,
/// error exits included: an empty stdout fails <c>JSON.parse</c> exactly as prose does, so "exit code
/// plus a human sentence on stderr" is not a machine contract. Several commands already emitted
/// <c>{"error": "..."}</c> by hand; this is that same one-liner, named once, so the shape cannot drift
/// per command.
/// </para>
/// <para>
/// <b>For the stderr-rebind pattern only.</b> A command that reserves stdout with
/// <see cref="MachineOutputScope"/> (currently <c>train</c>) has its <see cref="Console.Out"/>
/// replaced by <see cref="TextWriter.Null"/>, and reports failures through the scope's
/// <see cref="MachineOutputScope.ErrorSink"/> instead — calling this inside a scope would write the
/// payload nowhere.
/// </para>
/// </summary>
public static class JsonError
{
    /// <summary>
    /// Writes <c>{"error": "&lt;message&gt;"}</c> to stdout, verbatim.
    /// <para>
    /// The message is <b>plain text</b>, not markup: whatever is passed reaches the consumer
    /// unaltered, so a path or column name containing brackets survives intact. Use
    /// <see cref="EmitMarkup(string)"/> for a call site whose string is Spectre markup.
    /// </para>
    /// </summary>
    public static void Emit(string message) =>
        Console.WriteLine(JsonSerializer.Serialize(new { error = message }));

    /// <summary>
    /// Emits a message written as Spectre markup, with the tags removed — for the common case where
    /// one string literal serves both <see cref="ErrorConsole.Error(string)"/> and this payload, so
    /// the two channels cannot describe the same failure differently. The payload carries the
    /// sentence a consumer would print, not the rendering instructions.
    /// </summary>
    public static void EmitMarkup(string markup) => Emit(Plain(markup));

    /// <summary>
    /// The error and its tip as one payload, mirroring <see cref="ErrorConsole.Error(string, string)"/>:
    /// a tip that resolves the failure belongs with the failure, not in a second document the consumer
    /// has to correlate. Both parts are markup.
    /// </summary>
    public static void EmitMarkup(string markup, string tip) => Emit($"{Plain(markup)} {Plain(tip)}");

    /// <summary>
    /// Strips markup the way <see cref="MachineOutputScope"/> does for its event stream — including
    /// unescaping <c>[[</c>/<c>]]</c> back to literal brackets — and falls back to the string itself
    /// when it does not parse as markup, so a malformed call site degrades to verbatim rather than
    /// throwing on an error path.
    /// </summary>
    private static string Plain(string markup)
    {
        try { return Markup.Remove(markup); }
        catch (InvalidOperationException) { return markup; }
    }
}
