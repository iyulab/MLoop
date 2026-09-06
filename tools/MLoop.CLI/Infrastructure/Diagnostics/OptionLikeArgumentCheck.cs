using System.CommandLine;
using System.CommandLine.Parsing;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// Catches the command line that parses <b>successfully</b> and still means something the caller did
/// not write: a mistyped option absorbed as a positional value.
/// </summary>
/// <remarks>
/// <para>
/// A command that takes a positional argument will accept an unrecognized <c>--</c> token as that
/// argument's value rather than rejecting it, so a single mistyped character turns into a confident
/// report about the wrong thing — a mistyped selection flag is reported as a missing experiment, and
/// a mistyped output flag as a missing production model. The exit code is non-zero either way, so
/// the failure is not silent; what is wrong is the <i>cause</i>, and a cause that names the wrong
/// subject is worse than none: it sends the reader to look at their models when the defect is a
/// typo in front of them.
/// </para>
/// <para>
/// Detected on the resolved token stream, once, rather than by constraining each argument. Fifteen
/// positional arguments across a dozen commands would each need the same rule, a new command would
/// silently omit it, and the rule is not a property of any one argument — it is a property of how a
/// command line is read. The parser has already told us how it read every token by the time this
/// runs.
/// </para>
/// <para>
/// Only <c>--</c>-prefixed tokens are refused. A single dash is a value in its own right by long
/// convention — this CLI already reads <c>-</c> as "the standard input stream" for one option — and
/// refusing it would break that while catching nothing a typo produces.
/// </para>
/// <para>
/// The escape hatch is the standard one, and it is honored rather than reimplemented: everything
/// after a bare <c>--</c> is a value by definition, so the scan stops there. That is the answer for
/// the caller who genuinely has a file whose name begins with two dashes.
/// </para>
/// </remarks>
public static class OptionLikeArgumentCheck
{
    /// <summary>
    /// The first option-like value the parser bound to an argument, or null when the line is clean.
    /// </summary>
    public static string? Find(ParseResult parseResult)
    {
        foreach (var token in parseResult.Tokens)
        {
            // Everything past the separator is a value by definition — the caller said so.
            if (token.Type == TokenType.DoubleDash)
                return null;

            if (token.Type == TokenType.Argument && token.Value.StartsWith("--", StringComparison.Ordinal))
                return token.Value;
        }

        return null;
    }

    /// <summary>
    /// Reports the token on both channels a failure owes: the human sentence on stderr, and — when
    /// the caller asked for structured output — one document on stdout. The parse itself succeeded,
    /// so there is no parser rendering to redirect here; the report is the whole output.
    /// </summary>
    public static int Report(ParseResult parseResult, IReadOnlyList<string> args, string token)
    {
        var command = parseResult.CommandResult.Command.Name;
        var error = $"'{Markup.Escape(token)}' is not an option of '{Markup.Escape(command)}', so it was read as a value.";
        var tip = $"Run 'mloop {Markup.Escape(command)} --help' for the options it does take. To pass it as a value, put it after '--'.";

        ErrorConsole.Error(error, tip);

        if (JsonError.WasRequested(args))
            JsonError.EmitMarkup(error, tip);

        return 1;
    }
}
