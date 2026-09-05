using System.CommandLine;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// BD-7: <c>prep run</c> already has its own <c>PrepStep</c> list from <c>mloop.yaml</c> — a
/// <c>--json</c> branch serializes that list directly (no separate view model) plus the resolved
/// paths and row counts. Exercises the real command tree (<see cref="Program.BuildRootCommand"/>),
/// same approach as the other <c>--json</c> command tests.
/// </summary>
[Collection("FileSystem")]
public class PrepRunCommandJsonTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;

    public PrepRunCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-preprun-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets"));
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

    private void WriteYaml(string content) =>
        File.WriteAllText(Path.Combine(_testProjectRoot, "mloop.yaml"), content);

    private void WriteTrainCsv() =>
        File.WriteAllText(Path.Combine(_testProjectRoot, "datasets", "train.csv"),
            "a,b,Label\n1,,0\n2,4,1\n,6,0\n");

    // Same AnsiConsole.Console rebinding as the other --json tests.
    private static async Task<(int ExitCode, string Stdout)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(buffer)
            });

            var result = Program.BuildRootCommand().Parse(args);
            var exitCode = await result.InvokeAsync();
            return (exitCode, buffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }

    [Fact]
    public async Task PrepRun_Json_UnknownModel_ReportsError()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: regression
                label: b
            """);

        var (exitCode, stdout) = await RunAsync("prep", "run", "--name", "nope", "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetString()!.Contains("nope"));
    }

    [Fact]
    public async Task PrepRun_Json_NoPrepSteps_ReportsEmptySteps()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: regression
                label: b
            """);

        var (exitCode, stdout) = await RunAsync("prep", "run", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Empty(doc.RootElement.GetProperty("steps").EnumerateArray());
    }

    [Fact]
    public async Task PrepRun_Json_DryRun_ListsStepsWithoutExecuting()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: regression
                label: b
                prep:
                  - type: fill-missing
                    columns: [a]
                    method: mean
            """);
        WriteTrainCsv();

        var (exitCode, stdout) = await RunAsync("prep", "run", "--dry-run", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.True(doc.RootElement.GetProperty("dryRun").GetBoolean());
        var steps = doc.RootElement.GetProperty("steps").EnumerateArray().ToList();
        Assert.Single(steps);
        Assert.Equal("fill-missing", steps[0].GetProperty("type").GetString());
        Assert.False(doc.RootElement.TryGetProperty("output", out _)); // null omitted, no execution happened
    }

    [Fact]
    public async Task PrepRun_Json_Execute_ReportsRowCounts()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: regression
                label: b
                prep:
                  - type: fill-missing
                    columns: [a]
                    method: mean
            """);
        WriteTrainCsv();

        var (exitCode, stdout) = await RunAsync("prep", "run", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Equal(3, doc.RootElement.GetProperty("rowsBefore").GetInt32());
        Assert.Equal(3, doc.RootElement.GetProperty("rowsAfter").GetInt32());
        Assert.NotNull(doc.RootElement.GetProperty("output").GetString());
    }
}
