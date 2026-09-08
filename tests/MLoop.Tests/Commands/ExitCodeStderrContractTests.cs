using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// The two rules <c>docs/GUIDE.md</c> § Exit Codes states to anyone scripting mloop: a non-zero
/// exit always leaves a cause on stderr, and stdout is never where the diagnostic goes.
/// </summary>
/// <remarks>
/// <para>
/// Both halves have been broken before, in opposite directions. Diagnostics used to land on stdout
/// while the exit code stayed 0, so a caller reading stderr on failure got an empty string and a
/// caller redirecting stdout collected an error message as its answer. The guards that closed those
/// hold their own axes — <c>ParseFailureContractTests</c> the pre-command exit,
/// <c>LibraryOutputChannelContractTests</c> the library layer, <c>CommandExitCodeContractTests</c>
/// the handler signature. This one asserts the rule the documentation actually promises, end to
/// end, on real command lines a user can type.
/// </para>
/// <para>
/// Only failures that need no project state are exercised — a missing file, an unknown model, a
/// directory that is not an mloop project. That keeps the guard fast and free of fixtures while
/// still crossing the seam that matters: the command's own action, not the parser.
/// </para>
/// </remarks>
public class ExitCodeStderrContractTests
{
    private static (int ExitCode, string Stdout, string Stderr) RunOutsideAProject(params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var originalDirectory = Directory.GetCurrentDirectory();
        var scratch = Directory.CreateTempSubdirectory("mloop-exitcode-");

        var stdout = new StringWriter();
        var stderr = new StringWriter();
        try
        {
            Directory.SetCurrentDirectory(scratch.FullName);
            Console.SetOut(stdout);
            Console.SetError(stderr);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(stderr) });

            var exitCode = Program.Execute(Program.BuildRootCommand(), args);
            return (exitCode, stdout.ToString(), stderr.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
            Directory.SetCurrentDirectory(originalDirectory);
            try { scratch.Delete(recursive: true); } catch (IOException) { }
        }
    }

    /// <summary>
    /// Command lines that fail inside the command's own action — past the parser — and need nothing
    /// on disk to get there.
    /// </summary>
    private static readonly string[][] FailingCommandLines =
    [
        ["pipeline", "no-such-pipeline.yaml"],
        ["validate"],
        ["status"],
        ["list"],
        ["info", "no-such-data.csv"],
    ];

    public static TheoryData<string[]> FailingInvocations => [.. FailingCommandLines];

    [Theory]
    [MemberData(nameof(FailingInvocations))]
    public void AFailureExitsNonZeroAndSaysWhyOnStderr(string[] args)
    {
        var (exitCode, _, stderr) = RunOutsideAProject(args);

        Assert.NotEqual(0, exitCode);
        Assert.False(
            string.IsNullOrWhiteSpace(stderr),
            $"'mloop {string.Join(' ', args)}' exited {exitCode} with nothing on stderr — a caller "
            + "following the documented contract has no cause to report.");
    }

    /// <summary>
    /// The same failing lines, restricted to the commands that actually declare <c>--json</c>. The
    /// restriction is derived from the command tree rather than listed here: passing <c>--json</c>
    /// to a command that has no such option fails in the parser, which is a different seam with its
    /// own guard, and hardcoding the list is how it would silently stop matching the real one.
    /// </summary>
    public static TheoryData<string[]> FailingJsonInvocations
    {
        get
        {
            var root = Program.BuildRootCommand();
            var data = new TheoryData<string[]>();

            foreach (var args in FailingCommandLines)
            {
                var command = root.Subcommands.FirstOrDefault(c => c.Name == args[0]);
                if (command?.Options.Any(o => o.Name == "--json") == true)
                {
                    data.Add([.. args, "--json"]);
                }
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(FailingJsonInvocations))]
    public void UnderJsonAFailureLeavesAParseableDocumentOnStdout(string[] args)
    {
        var (exitCode, stdout, _) = RunOutsideAProject(args);

        Assert.NotEqual(0, exitCode);

        // Not merely "parseable or empty": a --json consumer that gets nothing on a non-zero exit
        // has to fall back to scraping stderr, which is the state --json exists to remove.
        var text = stdout.Trim();
        Assert.False(
            text.Length == 0,
            $"'mloop {string.Join(' ', args)}' exited {exitCode} and wrote no document at all.");

        var exception = Record.Exception(() => System.Text.Json.JsonDocument.Parse(text));
        Assert.True(
            exception is null,
            $"'mloop {string.Join(' ', args)}' exited {exitCode} and left unparseable text on "
            + $"stdout: {text[..Math.Min(200, text.Length)]}");
    }
}
