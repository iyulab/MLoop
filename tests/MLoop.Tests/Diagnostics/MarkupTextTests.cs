using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Tests.Documentation;
using Spectre.Console;

namespace MLoop.Tests.Diagnostics;

/// <summary>
/// <c>$"…"</c> passed to an error or warning line is markup in its literal and text in its values: a value with
/// brackets is shown as given. Before, every call site had to remember <see cref="Markup.Escape"/>, and fifteen did
/// not — a file named <c>data[1].csv</c> turned an error into "Unbalanced markup stack".
/// </summary>
[Collection("FileSystem")]
public class MarkupTextTests
{
    private static string Of(MarkupText text) => text.ToString();

    [Fact]
    public void TheLiteralIsMarkup_AndEveryValueIsText()
    {
        var path = "data[1].csv";
        var model = "[red]m[/]";

        Assert.Equal("File [cyan]data[[1]].csv[/] for [[red]]m[[/]]", Of($"File [cyan]{path}[/] for {model}"));
        Assert.Equal("data[1].csv", Markup.Remove(Of($"{path}")));
    }

    [Fact]
    public void ValuesFormatAsInterpolationFormatsThem()
    {
        var value = 0.12345;

        Assert.Equal($"{value:F2} |{"ab",4}|{"ab",-4}|", Of($"{value:F2} |{"ab",4}|{"ab",-4}|"));
    }

    // Two interpolated strings joined by + are one interpolated string to the compiler: the handler takes the whole.
    [Fact]
    public void AConcatenationOfInterpolatedStrings_IsEscapedAsOne()
    {
        var refusal = "rule [x]";
        var id = "exp-001";

        var written = CaptureWarning(() => WarningConsole.Warn($"Kept: {refusal}. " + $"[blue]mloop promote {id}[/]"));

        Assert.Contains("Kept: rule [x].", written);
        Assert.Contains("mloop promote exp-001", written);
    }

    [Fact]
    public void AnErrorWithAPathOfBrackets_IsWrittenAsTheMessage()
    {
        var original = Console.Error;
        var buffer = new StringWriter();
        Console.SetError(buffer);
        try
        {
            var file = "rows[1].csv";
            ErrorConsole.Error($"Data file not found: {file}", $"Tip for {file}");
        }
        finally
        {
            Console.SetError(original);
        }
        Assert.Contains("Data file not found: rows[1].csv", buffer.ToString());
        Assert.Contains("Tip for rows[1].csv", buffer.ToString());
    }

    private static string CaptureWarning(Action warn)
    {
        var original = AnsiConsole.Console;
        var buffer = new StringWriter();
        AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer) });
        try
        {
            warn();
        }
        finally
        {
            AnsiConsole.Console = original;
        }
        return buffer.ToString();
    }

    // A value escaped by hand inside $"…" is escaped twice now: data[1] would show as data[[1]].
    private static readonly Regex DoubleEscape = new(
        @"(ErrorConsole\.Error|WarningConsole\.Warn)\(\s*\$""[^""\n]*\{Markup\.Escape\(", RegexOptions.Compiled);

    private static bool Offends(string source) => DoubleEscape.IsMatch(source);

    [Fact]
    public void NoErrorOrWarningEscapesAnInterpolatedValueByHand()
    {
        var offenders = RepoSourceTree.ProductionSourceFiles("tools")
            .Where(f => Offends(File.ReadAllText(f)))
            .Select(Path.GetFileName)
            .ToList();

        Assert.Empty(offenders);
        // The scan sees the calls it polices: interpolated errors exist in the tree it reads.
        Assert.Contains(RepoSourceTree.ProductionSourceFiles("tools"), f => File.ReadAllText(f).Contains("ErrorConsole.Error($\""));
    }

    [Fact]
    public void TheScan_RefusesAHandEscapedValue_AndPassesAnEscapedWholeString()
    {
        Assert.True(Offends("ErrorConsole.Error($\"File not found: {Markup.Escape(path)}\");"));
        Assert.True(Offends("WarningConsole.Warn(\n    $\"x {Markup.Escape(e.Message)}\");"));
        Assert.False(Offends("ErrorConsole.Error(Markup.Escape($\"File not found: {path}\"));"));
        Assert.False(Offends("ErrorConsole.Error($\"File not found: {path}\");"));
    }
}
