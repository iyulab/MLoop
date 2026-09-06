using System.CommandLine;
using System.Text.Json;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// A <c>--json</c> command owes its consumer a parseable document at every exit, and error exits were
/// reaching <c>return 1</c> with an empty stdout. <c>predict</c> has twelve such exits and the
/// "not inside a project" exit is shared by nearly every command, so neither is reachable by guarding
/// call sites one at a time — <see cref="JsonOutputScope"/> closes both at once by checking, when the
/// command ends, whether anything reached stdout at all.
/// </summary>
[Collection("FileSystem")]
public class JsonOutputScopeContractTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;

    public JsonOutputScopeContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-json-scope-test-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));
        Directory.SetCurrentDirectory(_testProjectRoot);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }

        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    private static async Task<(int ExitCode, string Stdout, string Stderr)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        var errorBuffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            Console.SetError(errorBuffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer) });

            var exitCode = await Program.ExecuteAsync(Program.BuildRootCommand(), args);
            return (exitCode, buffer.ToString(), errorBuffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }

    private static JsonDocument ParseStdout(string stdout, string stderr)
    {
        try
        {
            return JsonDocument.Parse(stdout);
        }
        catch (JsonException ex)
        {
            throw new Xunit.Sdk.XunitException(
                $"stdout was not valid JSON: {ex.Message}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
        }
    }

    [Fact]
    public async Task Predict_WithJsonAndNoProductionModel_EmitsErrorPayload()
    {
        File.WriteAllText(Path.Combine(_testProjectRoot, "tiny.csv"), "X,Y\n1,2\n2,4\n");

        var (exitCode, stdout, stderr) = await RunAsync("predict", "tiny.csv", "--json");

        Assert.Equal(1, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Contains("No production model", doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task ScopedCommand_WithJsonOutsideAProject_EmitsErrorPayload()
    {
        // The exit every command shares, through CommandContext.TryCreate — the one a per-call-site
        // guard cannot reach, since no command owns the code that produces it.
        var outsideAnyProject = Path.Combine(Path.GetTempPath(), "mloop-not-a-project-" + Guid.NewGuid());
        Directory.CreateDirectory(outsideAnyProject);
        Directory.SetCurrentDirectory(outsideAnyProject);
        try
        {
            var (exitCode, stdout, stderr) = await RunAsync("logs", "--json");

            Assert.Equal(1, exitCode);
            using var doc = ParseStdout(stdout, stderr);
            Assert.Contains("project", doc.RootElement.GetProperty("error").GetString()!,
                StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.SetCurrentDirectory(_testProjectRoot);
            try { Directory.Delete(outsideAnyProject, recursive: true); } catch { }
        }
    }

    [Fact]
    public async Task UnexpectedException_WithJson_EmitsErrorPayload()
    {
        // The path every command's outermost catch funnels through. It announces the failure from
        // ErrorSuggestions rather than ErrorConsole, and when the scope was wired into only one of
        // those two the most ordinary failure a command has — an unexpected exception — still exited
        // with an empty stdout. Both now report through one place.
        var experiment = Path.Combine(_testProjectRoot, "models", "default", "staging", "exp-001");
        Directory.CreateDirectory(experiment);
        File.WriteAllText(Path.Combine(experiment, "metadata.json"), "{not json");
        File.WriteAllText(Path.Combine(_testProjectRoot, "tiny.csv"), "X,Y" + Environment.NewLine + "1,2" + Environment.NewLine + "2,4" + Environment.NewLine);

        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "exp-001", "tiny.csv", "--json");

        Assert.Equal(1, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.False(string.IsNullOrWhiteSpace(doc.RootElement.GetProperty("error").GetString()));
    }

    [Theory]
    [InlineData("list")]
    [InlineData("promote", "--best")]
    [InlineData("prep", "plan")]
    [InlineData("features", "select")]
    public async Task DeclaredJsonCommand_OutsideAProject_EmitsErrorPayload(params string[] command)
    {
        // The scope reached the commands that had a stderr-rebind block to replace; these had none and
        // so were left behind — including `list`, which is the command whose partly-covered state
        // started this audit in the first place.
        var outsideAnyProject = Path.Combine(Path.GetTempPath(), "mloop-not-a-project-" + Guid.NewGuid());
        Directory.CreateDirectory(outsideAnyProject);
        Directory.SetCurrentDirectory(outsideAnyProject);
        try
        {
            var args = command.Append("--json").ToArray();
            var (exitCode, stdout, stderr) = await RunAsync(args);

            Assert.Equal(1, exitCode);
            using var doc = ParseStdout(stdout, stderr);
            Assert.Contains("project", doc.RootElement.GetProperty("error").GetString()!,
                StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.SetCurrentDirectory(_testProjectRoot);
            try { Directory.Delete(outsideAnyProject, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Scope_WhenCommandWroteItsOwnPayload_AddsNothing()
    {
        var buffer = new StringWriter();
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        try
        {
            Console.SetOut(buffer);
            using (new JsonOutputScope())
            {
                Console.WriteLine("""{"valid":false,"errors":[]}""");
                JsonOutputScope.ReportError("something went wrong");
            }
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }

        // A command that reported a failure *and* described it in its own richer document keeps that
        // document alone — this is why the scope checks at the end rather than intercepting the report.
        using var doc = JsonDocument.Parse(buffer.ToString());
        Assert.False(doc.RootElement.GetProperty("valid").GetBoolean());
    }

    [Fact]
    public void Scope_WhenNothingWasWritten_EmitsTheReportedCause()
    {
        var buffer = new StringWriter();
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        try
        {
            Console.SetOut(buffer);
            using (new JsonOutputScope())
            {
                // Markup, as every ErrorConsole call site writes it; the payload carries the sentence.
                JsonOutputScope.ReportError("No production model found for '[cyan]default[/]'.");
                JsonOutputScope.ReportError("a later, outer re-report that should not win");
            }
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }

        using var doc = JsonDocument.Parse(buffer.ToString());
        var error = doc.RootElement.GetProperty("error").GetString();
        Assert.Equal("No production model found for 'default'.", error);
    }

    [Fact]
    public void Scope_WhenNothingWasWrittenAndNothingReported_StaysSilent()
    {
        var buffer = new StringWriter();
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        try
        {
            Console.SetOut(buffer);
            using (new JsonOutputScope()) { }
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }

        // Deliberately narrow: with no reported cause the scope cannot tell a silent failure from a
        // silent success, and inventing an error would put one on a zero-exit path.
        Assert.Equal(string.Empty, buffer.ToString());
    }

    [Fact]
    public async Task Validate_ListsFindingsWhosePathContainsBrackets()
    {
        // An indexed path ("models.default.prep[0]") and a note prefixed with a literal "[info] " are
        // data, not markup — interpolated raw they crashed the renderer, so the command whose job is
        // to report configuration problems died while reporting them.
        File.WriteAllText(Path.Combine(_testProjectRoot, "mloop.yaml"),
            "project: p\n" +
            "models:\n" +
            "  default:\n" +
            "    task: regression\n" +
            "    label: Y\n" +
            "    prep:\n" +
            "    - type: no-such-step\n");

        var (_, stdout, stderr) = await RunAsync("validate");

        var output = stdout + stderr;
        Assert.DoesNotContain("Unbalanced markup stack", output);
        Assert.Contains("no-such-step", output);
    }

    [Fact]
    public async Task PrepRun_ReportsStepValidationFailuresWithoutCrashing()
    {
        File.WriteAllText(Path.Combine(_testProjectRoot, "mloop.yaml"),
            "project: p\n" +
            "models:\n" +
            "  default:\n" +
            "    task: regression\n" +
            "    label: Y\n" +
            "    prep:\n" +
            "    - type: no-such-step\n");

        var (_, stdout, stderr) = await RunAsync("prep", "run", "--dry-run");

        var output = stdout + stderr;
        Assert.DoesNotContain("Unbalanced markup stack", output);
        Assert.Contains("no-such-step", output);
    }
}
