using System.CommandLine;
using System.Text.Json;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// Every <c>--json</c> command owes stdout that parses, in every state it can reach — including the
/// completely ordinary "nothing recorded yet" state of a fresh project. These commands used to take
/// an early return that printed a human hint to stdout and never emitted a payload at all, so a
/// consumer got a parse error rather than an empty result; <c>trigger</c> separately let its
/// progress line land on stdout ahead of the JSON. Driving the real command tree
/// (<see cref="Program.BuildRootCommand"/>) is what surfaces this — none of it is visible to a test
/// that calls the output helpers directly.
/// </summary>
[Collection("FileSystem")]
public class JsonEmptyStateContractTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;

    public JsonEmptyStateContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-json-empty-test-" + Guid.NewGuid());
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

            var exitCode = await Program.BuildRootCommand().Parse(args).InvokeAsync();
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
    public async Task Logs_Json_NoLogs_EmitsEmptyArray()
    {
        var (exitCode, stdout, stderr) = await RunAsync("logs", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Empty(doc.RootElement.EnumerateArray());
    }

    [Fact]
    public async Task FeedbackList_Json_NoFeedback_EmitsEmptyArray()
    {
        var (exitCode, stdout, stderr) = await RunAsync("feedback", "list", "--model", "default", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Empty(doc.RootElement.EnumerateArray());
    }

    [Fact]
    public async Task FeedbackMetrics_Json_NoFeedback_EmitsZeroedPayload()
    {
        var (exitCode, stdout, stderr) = await RunAsync("feedback", "metrics", "--model", "default", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Equal(0, doc.RootElement.GetProperty("totalFeedback").GetInt32());
    }

    [Fact]
    public async Task SampleStats_Json_NoPredictions_EmitsZeroedPayload()
    {
        var (exitCode, stdout, stderr) = await RunAsync("sample", "stats", "--model", "default", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Equal(0, doc.RootElement.GetProperty("totalPredictions").GetInt32());
    }

    [Fact]
    public async Task TriggerCheck_Json_ProgressLineDoesNotReachStdout()
    {
        var (exitCode, stdout, stderr) = await RunAsync("trigger", "check", "--model", "default", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.True(doc.RootElement.TryGetProperty("shouldRetrain", out _));
    }
}
