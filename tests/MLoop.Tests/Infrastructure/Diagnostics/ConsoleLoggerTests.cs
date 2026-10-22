using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.Tests.Infrastructure.Diagnostics;

/// <summary>
/// The one logger the CLI hands out: a message is printed as written — brackets included, where markup
/// would have thrown — errors land on stderr, warnings come through the warning seam, debug detail
/// nowhere.
/// </summary>
public class ConsoleLoggerTests
{
    [Fact]
    public void Messages_WithBrackets_ArePrinted_ErrorsOnStderr_WarningsThroughTheSeam()
    {
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        try
        {
            Console.SetError(stderr);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(stdout), Ansi = AnsiSupport.No, ColorSystem = ColorSystemSupport.NoColors });
            var logger = new ConsoleLogger();

            logger.Info("loaded [train.csv]");
            logger.Warning("column [a] is constant");
            logger.Error("script [hook.cs] failed");
            logger.Debug("internal detail");
        }
        finally
        {
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
        }

        Assert.Contains("loaded [train.csv]", stdout.ToString());
        Assert.Contains("Warning: column [a] is constant", stdout.ToString());
        Assert.Contains("script [hook.cs] failed", stderr.ToString());
        Assert.DoesNotContain("script [hook.cs] failed", stdout.ToString());
        Assert.DoesNotContain("internal detail", stdout.ToString() + stderr.ToString());
    }
}
