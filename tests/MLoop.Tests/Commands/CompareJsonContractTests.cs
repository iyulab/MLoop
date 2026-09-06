using System.CommandLine;
using System.Text.Json;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// The local-state path of <c>compare</c> declared <c>--json</c>, documented it as machine-readable,
/// and then never read the flag — it rendered the same Spectre table to stdout either way, so a
/// consumer asking for JSON got a box-drawing diagram. <c>list --trials</c> had the narrower version
/// of the same defect: its error exits wrote a human sentence to stderr and left stdout empty, which
/// fails <c>JSON.parse</c> exactly as prose does.
/// <para>
/// Experiments are written by hand rather than trained: this command reads
/// <c>metadata.json</c>/<c>metrics.json</c>/<c>config.json</c> and never opens a model, so a real
/// search would add minutes and prove nothing this does not.
/// </para>
/// </summary>
[Collection("FileSystem")]
public class CompareJsonContractTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;

    public CompareJsonContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-compare-json-test-" + Guid.NewGuid());
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

    private void WriteExperiment(string experimentId, double rSquared, double meanAbsoluteError)
    {
        var dir = Path.Combine(_testProjectRoot, "models", "default", "staging", experimentId);
        Directory.CreateDirectory(dir);

        File.WriteAllText(Path.Combine(dir, "metadata.json"),
            "{\n" +
            "  \"modelName\": \"default\",\n" +
            "  \"experimentId\": \"" + experimentId + "\",\n" +
            "  \"timestamp\": \"2026-01-02T03:04:05Z\",\n" +
            "  \"status\": \"Completed\",\n" +
            "  \"task\": \"regression\",\n" +
            "  \"result\": { \"bestTrainer\": \"Sdca\", \"trainingTimeSeconds\": 1.5 }\n" +
            "}\n");

        File.WriteAllText(Path.Combine(dir, "metrics.json"),
            $"{{ \"r_squared\": {rSquared}, \"mean_absolute_error\": {meanAbsoluteError} }}\n");

        File.WriteAllText(Path.Combine(dir, "config.json"),
            "{\n" +
            "  \"dataFile\": \"train.csv\",\n" +
            "  \"labelColumn\": \"Y\",\n" +
            "  \"timeLimitSeconds\": 10,\n" +
            "  \"metric\": \"r_squared\",\n" +
            "  \"testSplit\": 0.2\n" +
            "}\n");
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
    public async Task Compare_WithJson_EmitsRankedPayloadInsteadOfTable()
    {
        WriteExperiment("exp-001", rSquared: 0.71, meanAbsoluteError: 0.42);
        WriteExperiment("exp-002", rSquared: 0.88, meanAbsoluteError: 0.19);

        var (exitCode, stdout, stderr) = await RunAsync("compare", "exp-001", "exp-002", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        var root = doc.RootElement;

        Assert.Equal("r_squared", root.GetProperty("metric").GetString());
        Assert.Equal("maximize", root.GetProperty("direction").GetString());
        Assert.Equal("exp-002", root.GetProperty("best").GetString());
        Assert.False(root.GetProperty("tie").GetBoolean());

        // Ranking is best-first, and enrichment carries the state only a local comparison has.
        var ranking = root.GetProperty("ranking").EnumerateArray()
            .Select(e => e.GetProperty("id").GetString()).ToList();
        Assert.Equal(new[] { "exp-002", "exp-001" }, ranking);
        Assert.Equal(2, root.GetProperty("experiments").GetArrayLength());
        Assert.Equal("regression", root.GetProperty("experiments")[0].GetProperty("task").GetString());
    }

    [Fact]
    public async Task Compare_WithJsonAndLowerBetterMetric_RanksByDirectionNotMagnitude()
    {
        WriteExperiment("exp-001", rSquared: 0.71, meanAbsoluteError: 0.42);
        WriteExperiment("exp-002", rSquared: 0.88, meanAbsoluteError: 0.19);

        var (_, stdout, stderr) = await RunAsync(
            "compare", "exp-001", "exp-002", "--metric", "mean_absolute_error", "--json");

        using var doc = ParseStdout(stdout, stderr);
        Assert.Equal("minimize", doc.RootElement.GetProperty("direction").GetString());
        Assert.Equal("exp-002", doc.RootElement.GetProperty("best").GetString());
    }

    [Fact]
    public async Task Compare_WithJsonAndMissingExperiment_ReportsItAsExcludedNotAWarningLine()
    {
        WriteExperiment("exp-001", rSquared: 0.71, meanAbsoluteError: 0.42);

        var (exitCode, stdout, stderr) = await RunAsync("compare", "exp-001", "exp-999", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);

        var excluded = doc.RootElement.GetProperty("excluded");
        Assert.Equal(1, excluded.GetArrayLength());
        Assert.Equal("exp-999", excluded[0].GetProperty("id").GetString());
        Assert.Equal("not-found", excluded[0].GetProperty("reason").GetString());

        // The human warning belongs on stderr, never in the stream the consumer parses.
        Assert.Contains("exp-999", stderr);
    }

    [Fact]
    public async Task Compare_WithJsonAndNoExperimentsFound_EmitsEmptyResultNotProse()
    {
        var (exitCode, stdout, stderr) = await RunAsync("compare", "exp-998", "exp-999", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);

        // "nothing to compare" is a result, not a failure — an empty ranking with no winner.
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("best").ValueKind);
        Assert.Equal(0, doc.RootElement.GetProperty("ranking").GetArrayLength());
        Assert.Equal(0, doc.RootElement.GetProperty("experiments").GetArrayLength());
        Assert.Equal(2, doc.RootElement.GetProperty("excluded").GetArrayLength());
    }

    [Fact]
    public async Task Compare_WithJsonAndNoModelName_EmitsErrorPayload()
    {
        var (exitCode, stdout, stderr) = await RunAsync("compare", "--json");

        Assert.Equal(1, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Contains("Model name is required", doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task List_WithJsonAndNoExperiments_EmitsEmptyPayload()
    {
        // The path this covers is why the audit was re-counted: ListCommand looked covered because
        // its --trials path had tests, while plain `list --json` — the state a fresh project is in —
        // had never been driven through the command tree at all.
        var (exitCode, stdout, stderr) = await RunAsync("list", "--json");

        Assert.Equal(0, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Equal(0, doc.RootElement.GetProperty("experiments").GetArrayLength());
    }

    [Fact]
    public async Task ListTrials_WithJsonAndUnknownExperiment_EmitsErrorPayload()
    {
        var (exitCode, stdout, stderr) = await RunAsync("list", "--trials", "exp-999", "--json");

        Assert.Equal(1, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Contains("exp-999", doc.RootElement.GetProperty("error").GetString());
    }

    [Fact]
    public async Task ListTrials_WithJsonAndNoLeaderboard_EmitsErrorPayload()
    {
        WriteExperiment("exp-001", rSquared: 0.71, meanAbsoluteError: 0.42);

        var (exitCode, stdout, stderr) = await RunAsync("list", "--trials", "exp-001", "--json");

        Assert.Equal(1, exitCode);
        using var doc = ParseStdout(stdout, stderr);
        Assert.Contains("leaderboard", doc.RootElement.GetProperty("error").GetString());
    }
}
