using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A command's failure goes through <see cref="ErrorConsole"/> and nowhere else: the cause reaches stderr (exit ≠ 0 ⇒
/// stderr has a cause) and, in <c>--json</c> mode, <see cref="JsonOutputScope"/> turns that report into the
/// document when the command wrote none of its own.
/// </summary>
/// <remarks>
/// Before this guard, thirteen error lines went to stdout as red markup (an exit 1 with an empty stderr — and a model
/// name containing <c>[</c> crashed the markup parser), six commands kept a private copy of the error writer
/// (<c>Console.Error</c> + an explicit JSON document), and seven exits emitted the JSON document by hand after
/// reporting through <see cref="ErrorConsole"/>, so the scope's rule ran twice and the hand-built one was poorer.
/// </remarks>
public class CommandErrorChannelContractTests
{
    /// <summary>A red line written to stdout — the shape an error line takes when it misses the error channel.</summary>
    private static readonly Regex StdoutRedLine =
        new(@"AnsiConsole\.(MarkupLine|Markup|MarkupLineInterpolated|WriteLine)\(\$?@?""\[(bold )?red\]", RegexOptions.Compiled);

    /// <summary>stderr written past the error channel.</summary>
    private static readonly Regex DirectStderr = new(@"Console\.Error\.Write", RegexOptions.Compiled);

    /// <summary>A JSON error document built by a command instead of by its scope.</summary>
    private static readonly Regex HandBuiltJsonError = new(@"JsonError\.Emit", RegexOptions.Compiled);

    private static List<string> CommandFiles() =>
        RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI", "Commands"))
            .Where(f => Path.GetFileName(f).EndsWith("Command.cs", StringComparison.Ordinal))
            .ToList();

    private static IEnumerable<string> Offenders(Regex shape, IEnumerable<string> files) =>
        from file in files
        from line in File.ReadAllLines(file).Select((text, i) => (text, number: i + 1))
        where shape.IsMatch(line.text)
        select $"{Path.GetFileName(file)}:{line.number}: {line.text.Trim()}";

    [Fact]
    public void TheScanSeesTheCommands()
    {
        var files = CommandFiles();

        Assert.True(files.Count > 30, $"expected the CLI's commands, found {files.Count}");
        Assert.Contains(files, f => Path.GetFileName(f) == "PredictCommand.cs");
    }

    [Theory]
    [InlineData(@"AnsiConsole.MarkupLine(""[red]File is empty[/]"");")]
    [InlineData(@"AnsiConsole.MarkupLine($""[red]Model '{name}' not found.[/]"");")]
    [InlineData(@"Console.Error.WriteLine(message);")]
    [InlineData(@"JsonError.Emit(message);")]
    [InlineData(@"    JsonError.EmitMarkup(cause, tip);")]
    public void TheShapesItForbidsAreSeen(string line)
    {
        Assert.True(StdoutRedLine.IsMatch(line) || DirectStderr.IsMatch(line) || HandBuiltJsonError.IsMatch(line), line);
    }

    [Theory]
    [InlineData(@"ErrorConsole.Error($""Model '{Markup.Escape(name)}' not found."");")]
    [InlineData(@"err.MarkupLine($""  [red]✗[/] {Markup.Escape(error)}"");")]
    [InlineData(@"AnsiConsole.MarkupLine(""[green]>[/] Schema validation passed"");")]
    public void TheErrorChannelItselfIsNotRefused(string line)
    {
        Assert.False(StdoutRedLine.IsMatch(line) || DirectStderr.IsMatch(line) || HandBuiltJsonError.IsMatch(line), line);
    }

    [Fact]
    public void NoCommandWritesAnErrorLineToStdout() =>
        Assert.Empty(Offenders(StdoutRedLine, CommandFiles()));

    [Fact]
    public void NoCommandWritesStderrPastTheErrorChannel() =>
        Assert.Empty(Offenders(DirectStderr, CommandFiles()));

    [Fact]
    public void NoCommandBuildsItsOwnJsonErrorDocument() =>
        Assert.Empty(Offenders(HandBuiltJsonError, CommandFiles()));
}
