using System.Text.Json;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// The structured-output guarantee has to hold at the one exit that happens before any command runs:
/// a command line that fails to parse. No command action is entered there, so none of the in-action
/// machinery — the stdout-owning scope, the failure fan-out — is ever constructed, and the exit was
/// writing usage text to stdout where a caller redirecting stdout collects it as the answer.
/// </summary>
public class ParseFailureContractTests
{
    private static (int ExitCode, string Stdout, string Stderr) Run(params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        try
        {
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
        }
    }

    // Commands with no positional argument, deliberately: where one exists, an unknown option binds
    // to it as a value and the line parses, so it never reaches this layer at all.
    [Theory]
    [InlineData("list")]
    [InlineData("status")]
    [InlineData("validate")]
    [InlineData("logs")]
    public void UnknownOption_WithStructuredOutput_PutsOneDocumentOnStdout(string command)
    {
        const string bad = "--bogus";
        var (exitCode, stdout, stderr) = Run(command, bad, "--json");

        Assert.Equal(1, exitCode);

        var document = JsonDocument.Parse(stdout);
        Assert.True(document.RootElement.TryGetProperty("error", out var error));
        Assert.Contains(bad, error.GetString());

        // Usage belongs on the diagnostic stream, never mixed into the answer.
        Assert.DoesNotContain("Usage:", stdout);
        Assert.Contains("Usage:", stderr);
    }

    [Fact]
    public void UnknownCommand_WithStructuredOutput_StillYieldsADocument()
    {
        // The layer cannot consult a command's options here: the command name is itself the token the
        // parser rejected. The raw-argument scan is what keeps the guarantee reachable.
        var (exitCode, stdout, _) = Run("no-such-command", "--json");

        Assert.Equal(1, exitCode);
        var document = JsonDocument.Parse(stdout);
        Assert.Contains("no-such-command", document.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public void ParseFailure_ReportsEveryCause_NotOnlyTheFirst()
    {
        var (_, stdout, _) = Run("no-such-command", "--json");

        var message = JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()!;
        Assert.Contains("no-such-command", message);
        Assert.Contains("--json", message);
    }

    [Fact]
    public void ParseFailure_WithoutStructuredOutput_LeavesStdoutUntouched()
    {
        var (exitCode, stdout, stderr) = Run("list", "--bogus");

        Assert.Equal(1, exitCode);
        Assert.Equal(string.Empty, stdout);
        Assert.Contains("--bogus", stderr);
        Assert.Contains("Usage:", stderr);
    }

    [Fact]
    public void RequestedHelp_IsTheAnswer_AndStaysOnStdout()
    {
        // Help asked for explicitly parses cleanly, so it never reaches this path — it is the
        // command's output, not a diagnostic about a failure.
        var (exitCode, stdout, _) = Run("list", "--help");

        Assert.Equal(0, exitCode);
        Assert.Contains("Usage:", stdout);
    }

    [Theory]
    [InlineData(new[] { "list", "--json" }, true)]
    [InlineData(new[] { "list", "--jsonx" }, false)]
    [InlineData(new[] { "list", "--no-json" }, false)]
    [InlineData(new[] { "list" }, false)]
    public void StructuredOutputRequest_IsMatchedByExactToken(string[] args, bool expected) =>
        Assert.Equal(expected, JsonError.WasRequested(args));
}
