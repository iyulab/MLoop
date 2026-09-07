using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Diagnostics;

/// <summary>
/// Guards the CLI's error-channel contract: when a command fails, stderr carries the cause.
/// <para>
/// Subprocess consumers follow the POSIX convention of reading stderr on a non-zero exit. Domain
/// errors used to be written to stdout only, so such a consumer received an empty reason while a
/// fully-formed diagnostic sat in stdout — and the channel silently depended on the *kind* of
/// error, since System.CommandLine's own parse errors did go to stderr.
/// </para>
/// </summary>
/// <remarks>
/// <see cref="ErrorConsole.Error(string)"/> unconditionally reports into
/// <see cref="MachineOutputScope.Current"/> — a process-global static — so a test here that runs
/// concurrently with a <see cref="MachineOutputScope"/>-owning test (a different default xUnit
/// collection, absent this attribute) can report into that *other* test's <c>ErrorSink</c> instead
/// of nobody's. Shares the "FileSystem" collection with <c>MachineOutputScopeTests</c> so xUnit
/// serializes them against each other; the collection name is otherwise unrelated to this concern.
/// </remarks>
[Collection("FileSystem")]
public class ErrorConsoleTests
{
    /// <summary>
    /// Captures stderr for the duration of <paramref name="action"/>. Spectre strips markup when
    /// writing to a non-terminal writer, so assertions compare plain text.
    /// </summary>
    private static string CaptureStdErr(Action action)
    {
        var original = Console.Error;
        var buffer = new StringWriter();
        try
        {
            Console.SetError(buffer);
            action();
        }
        finally
        {
            Console.SetError(original);
        }
        return buffer.ToString();
    }

    [Fact]
    public void Error_WritesToStdErr()
    {
        var output = CaptureStdErr(() => ErrorConsole.Error("Cannot train a classifier with only one class."));

        Assert.Contains("Cannot train a classifier with only one class.", output);
        Assert.Contains("Error:", output);
    }

    [Fact]
    public void Error_LongLineIsNotFoldedMidToken()
    {
        // Spectre lays a redirected stream out at 80 columns and folds a long line there — inside a
        // token when the line has no space at that point. A guidance line names a command to copy,
        // so it must survive capture as one line: a user who copies a folded command gets a broken one.
        var command = "mloop train --name a-model-name-long-enough-to-cross-the-eightieth-column --time 60";
        var output = CaptureStdErr(() => ErrorConsole.Error(
            $"No completed experiments found for model 'default'. Run '{command}' first."));

        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Single(lines);
        Assert.Contains(command, output);
    }

    [Fact]
    public void Tip_WritesToStdErr()
    {
        // Tips are part of the cause a machine consumer persists, so they share the error channel.
        var output = CaptureStdErr(() => ErrorConsole.Tip("Check if the correct label column is specified."));

        Assert.Contains("Check if the correct label column is specified.", output);
    }

    [Fact]
    public void DisplayError_WritesMessageAndSuggestionsToStdErr()
    {
        // Every command's top-level catch funnels through DisplayError before returning non-zero,
        // so this one path is what makes the contract hold CLI-wide rather than command-by-command.
        var ex = new InvalidOperationException("Cannot train a classifier with only one class.");

        var output = CaptureStdErr(() => ErrorSuggestions.DisplayError(ex, "training"));

        Assert.Contains("Cannot train a classifier with only one class.", output);
        Assert.Contains("Suggestions:", output);
    }

    [Fact]
    public void DisplayTrainingError_WritesToStdErr()
    {
        var ex = new InvalidOperationException("AUC is not defined when there is no positive class");

        var output = CaptureStdErr(() => ErrorSuggestions.DisplayTrainingError(ex, "obj-42"));

        Assert.Contains("AUC is not defined when there is no positive class", output);
        Assert.Contains("obj-42", output);
    }

    [Fact]
    public void PathNotFoundTip_NamesTheProjectRoot()
    {
        // Every explicit-path command option resolves a relative path against the project root
        // before checking existence — the tip states that rule so a consumer under a different
        // working directory isn't left guessing which base was actually used.
        var tip = ErrorConsole.PathNotFoundTip(@"D:\projects\my-model");

        Assert.Contains(@"D:\projects\my-model", tip);
        Assert.Contains("project root", tip);
    }

    [Fact]
    public void PathNotFoundTipCwd_NamesTheCurrentDirectory()
    {
        // `sample create --from` is the one path option that resolves against the current
        // directory instead of the project root — a distinct fact from PathNotFoundTip's.
        var tip = ErrorConsole.PathNotFoundTipCwd();

        Assert.Contains(Directory.GetCurrentDirectory(), tip);
        Assert.Contains("current directory", tip);
    }

    [Fact]
    public void Out_ReboundsToCurrentStdErr()
    {
        // The console must not cache a writer captured at first use — otherwise a redirection
        // installed later is silently ignored and diagnostics vanish.
        var first = CaptureStdErr(() => ErrorConsole.Error("first"));
        var second = CaptureStdErr(() => ErrorConsole.Error("second"));

        Assert.Contains("first", first);
        Assert.DoesNotContain("second", first);
        Assert.Contains("second", second);
        Assert.DoesNotContain("first", second);
    }
}
