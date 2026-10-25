using System.Runtime.CompilerServices;
using System.Text;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// An interpolated string written as console markup: the literal parts are markup (<c>[cyan]…[/]</c> styles them), and
/// every interpolated value is text — escaped, so a path, a name or an exception message with brackets
/// (<c>data[1].csv</c>) is shown as it is instead of being read as a style tag. <see cref="ErrorConsole.Error(MarkupText)"/>
/// and <see cref="WarningConsole.Warn(MarkupText)"/> take one wherever they are called with <c>$"…"</c>, so no call site
/// has to remember <see cref="Markup.Escape"/>.
/// </summary>
/// <remarks>
/// Values format as string interpolation formats them — in the current culture, with the format and alignment given.
/// A value that is markup on purpose is passed as <see cref="Markup"/> text in the literal, not as a value.
/// </remarks>
[InterpolatedStringHandler]
public struct MarkupText
{
    private readonly StringBuilder _markup;

    public MarkupText(int literalLength, int formattedCount) => _markup = new StringBuilder(literalLength + formattedCount * 16);

    public readonly void AppendLiteral(string literal) => _markup.Append(literal);

    public readonly void AppendFormatted<T>(T value) => AppendFormatted(value, alignment: 0, format: null);

    public readonly void AppendFormatted<T>(T value, string? format) => AppendFormatted(value, alignment: 0, format);

    public readonly void AppendFormatted<T>(T value, int alignment) => AppendFormatted(value, alignment, format: null);

    public readonly void AppendFormatted<T>(T value, int alignment, string? format)
    {
        var text = value is IFormattable formattable ? formattable.ToString(format, null) : value?.ToString() ?? "";
        if (alignment != 0)
            text = alignment > 0 ? text.PadLeft(alignment) : text.PadRight(-alignment);
        _markup.Append(Markup.Escape(text));
    }

    /// <summary>The markup: the literal parts as written, the values escaped.</summary>
    public override readonly string ToString() => _markup?.ToString() ?? "";
}
