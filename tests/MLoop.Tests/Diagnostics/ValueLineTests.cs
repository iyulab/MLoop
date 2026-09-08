using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.Tests.Diagnostics;

/// <summary>
/// A value the reader copies is written whole, at any console width.
/// </summary>
/// <remarks>
/// Measured before this: <c>mloop promote</c> printed a backup path that Spectre folded at column
/// 80 — the width it assumes whenever the stream is not a terminal — splitting the timestamp in the
/// directory name across two lines. What a user saw and copied was not a path that existed. The
/// width in these tests is set deliberately narrow so a fold, if one returned, happens well inside
/// the value.
/// </remarks>
public class ValueLineTests
{
    private const string LongPath =
        @"C:\Users\someone\AppData\Local\Temp\a-project\models\default\backups\exp-001-20260908-064321";

    private static string Render(string labelMarkup, string value, int width)
    {
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            var console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Ansi = AnsiSupport.No,
                ColorSystem = ColorSystemSupport.NoColors,
                Out = new AnsiConsoleOutput(buffer),
            });
            console.Profile.Width = width;
            AnsiConsole.Console = console;

            ValueLine.Write(labelMarkup, value);
            return buffer.ToString();
        }
        finally
        {
            AnsiConsole.Console = original;
        }
    }

    [Theory]
    [InlineData(20)]
    [InlineData(80)]
    [InlineData(200)]
    public void TheValueSurvivesOnOneLineAtAnyWidth(int width)
    {
        var rendered = Render("[green]>[/] Backup: ", LongPath, width);

        Assert.Contains(LongPath, rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLabelIsStillRenderedAsMarkup()
    {
        var rendered = Render("[green]>[/] Backup: ", LongPath, 80);

        Assert.StartsWith("> Backup: ", rendered, StringComparison.Ordinal);
        Assert.DoesNotContain("[green]", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLineEndsAfterTheValue()
    {
        var rendered = Render("[grey]Output: [/]", LongPath, 40);

        Assert.Equal($"Output: {LongPath}", rendered.TrimEnd('\r', '\n'));
    }

    /// <summary>
    /// The guard is only meaningful if the plain path it replaces would actually have folded at
    /// these widths — otherwise it would pass on a console that never folds anything.
    /// </summary>
    [Fact]
    public void TheOrdinaryRenderingWouldHaveFoldedTheSameValue()
    {
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            var console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Ansi = AnsiSupport.No,
                ColorSystem = ColorSystemSupport.NoColors,
                Out = new AnsiConsoleOutput(buffer),
            });
            console.Profile.Width = 40;
            AnsiConsole.Console = console;

            AnsiConsole.MarkupLine($"[green]>[/] Backup: [grey]{LongPath}[/]");
        }
        finally
        {
            AnsiConsole.Console = original;
        }

        Assert.DoesNotContain(LongPath, buffer.ToString(), StringComparison.Ordinal);
    }
}
