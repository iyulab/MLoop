using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// A line that ends in a value the reader is going to copy — a path, a command, an id. The label is
/// rendered; the value is written whole.
/// </summary>
/// <remarks>
/// <para>
/// Spectre lays out every line against the console's width and folds what does not fit, inserting a
/// real newline at the break — including inside a token, when the token has no space to break at.
/// A backup path came out as <c>…\backups\exp-001-2</c> on one line and <c>0260908-064321</c> on the
/// next, and a user who copies what they see gets a path that does not exist. The width it folds
/// against is 80 whenever the stream is not a terminal, so the defect is worst exactly where output
/// is captured and pasted from.
/// </para>
/// <para>
/// The fix is not a wider console. Width is a property of the whole console, and raising it would
/// draw every rule and progress bar four thousand columns wide. What has to be true is narrower
/// than that: <b>a value is written whole</b>. So the label goes through the console — markup,
/// styling, and whatever stream the active output scope installed — and the value goes to that same
/// console's own writer, past the layout that would fold it.
/// </para>
/// <para>
/// This is the stdout counterpart of the rule <see cref="ErrorConsole"/> states for diagnostics, and
/// it is deliberately not "warnings must not fold". Prose reflows without harm, and a machine
/// consumer never sees the folded form anyway — <see cref="WarningConsole"/> reports the unfolded
/// text as a <c>warning</c> event. Only a value a person retypes or pastes is broken by a fold.
/// </para>
/// </remarks>
public static class ValueLine
{
    /// <summary>
    /// Writes <paramref name="labelMarkup"/> (Spectre markup, escaped by the caller) followed by
    /// <paramref name="value"/> verbatim and a newline.
    /// </summary>
    public static void Write(string labelMarkup, string value)
    {
        var console = AnsiConsole.Console;

        console.Markup(labelMarkup);
        console.Profile.Out.Writer.WriteLine(value);
    }
}
