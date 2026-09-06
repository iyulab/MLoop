using System.Text.Json;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// A command that takes a positional argument accepts an unrecognized <c>--</c> token as that
/// argument's value, so one mistyped character produced a confident report about the wrong subject —
/// a mistyped selection flag read as a missing experiment, a mistyped output flag as a missing
/// production model. The exit code was already non-zero; what was wrong was the cause.
/// </summary>
public class OptionLikeArgumentContractTests
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

    [Theory]
    [InlineData("promote", "--lastest")]
    [InlineData("predict", "--jsn")]
    [InlineData("evaluate", "--modle")]
    public void MistypedOption_IsNamedAsTheCause_NotAbsorbedAsAValue(string command, string typo)
    {
        var (exitCode, _, stderr) = Run(command, typo);

        Assert.Equal(1, exitCode);
        Assert.Contains(typo, stderr);
        Assert.Contains("is not an option", stderr);
    }

    [Fact]
    public void MistypedOption_WithStructuredOutput_PutsTheSameCauseOnStdout()
    {
        var (exitCode, stdout, _) = Run("promote", "--lastest", "--json");

        Assert.Equal(1, exitCode);
        var message = JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()!;
        Assert.Contains("--lastest", message);
        Assert.Contains("is not an option", message);
    }

    [Fact]
    public void AfterTheSeparator_ATokenIsAValue_AndTheCheckStandsDown()
    {
        // The standard escape hatch, honored rather than reimplemented: a caller whose value really
        // does begin with two dashes says so with a bare separator.
        var (_, _, stderr) = Run("promote", "--", "--lastest");

        Assert.DoesNotContain("is not an option", stderr);
    }

    [Fact]
    public void ASingleDashIsAValue_AndIsLeftAlone()
    {
        // One dash is a value by long convention — this CLI already reads it as the standard input
        // stream for one option — so refusing it would break that while catching no typo.
        var (_, _, stderr) = Run("promote", "-weird");

        Assert.DoesNotContain("is not an option", stderr);
    }

    [Fact]
    public void ARecognizedOption_IsUnaffected()
    {
        var (_, _, stderr) = Run("promote", "--latest");

        Assert.DoesNotContain("is not an option", stderr);
    }
}
