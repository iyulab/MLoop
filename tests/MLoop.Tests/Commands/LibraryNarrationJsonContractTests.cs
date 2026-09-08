using System.Text.Json;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// A <c>--json</c> command's stdout must parse even when the data it reads makes <c>MLoop.Core</c>
/// narrate — the case an audit of the CLI's own streams cannot reach.
/// </summary>
/// <remarks>
/// <para>
/// Core's loaders describe what they did to a file: which index column they dropped, which multi-line
/// header they flattened. Those lines defaulted to <c>Console.WriteLine</c>, which is not the console
/// a command's scope replaces, so they landed on stdout <i>ahead of the document</i>. Measured:
/// <c>mloop info --json</c> and <c>mloop analyze profile --json</c> both began with
/// <c>[Info] Removed index column(s): Unnamed: 0</c> and exited 0 — a consumer's parse failed on a
/// command that reported success.
/// </para>
/// <para>
/// Two separate causes, so two guards' worth of surface in one fixture: Core reaching past the host
/// (fixed by injecting the sink), and <c>analyze</c> opening no scope at all for a flag all five of
/// its sub-commands declare. A fixture whose CSV is clean would pass under both defects, so the one
/// here deliberately carries an index column — the input that makes the loader speak.
/// </para>
/// </remarks>
[Collection("FileSystem")]
public class LibraryNarrationJsonContractTests : IDisposable
{
    private readonly string _projectRoot;
    private readonly string _originalDirectory;
    private readonly string _dataFile;

    public LibraryNarrationJsonContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _projectRoot = Path.Combine(Path.GetTempPath(), "mloop-narration-json-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_projectRoot, ".mloop"));
        Directory.CreateDirectory(Path.Combine(_projectRoot, "datasets"));

        // "Unnamed: 0" is the pandas default index, which RemoveIndexColumns drops and announces.
        var rows = new List<string> { "Unnamed: 0,feature,Label" };
        for (var i = 0; i < 40; i++)
            rows.Add($"{i},{i % 7},{i % 2}");

        _dataFile = Path.Combine(_projectRoot, "datasets", "train.csv");
        File.WriteAllLines(_dataFile, rows);
        Directory.SetCurrentDirectory(_projectRoot);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }

        if (Directory.Exists(_projectRoot))
        {
            try { Directory.Delete(_projectRoot, recursive: true); } catch { }
        }
    }

    [Theory]
    [InlineData("info", "datasets/train.csv", "--json")]
    [InlineData("analyze", "profile", "datasets/train.csv", "--json")]
    [InlineData("analyze", "correlation", "datasets/train.csv", "--json")]
    [InlineData("analyze", "outliers", "datasets/train.csv", "--json")]
    [InlineData("analyze", "distribution", "datasets/train.csv", "--json")]
    [InlineData("detect", "datasets/train.csv", "--column", "feature", "--json")]
    public async Task StdoutParsesWhenTheLoaderNarrates(params string[] args)
    {
        var (exitCode, stdout, stderr) = await RunAsync(args);

        Assert.Equal(0, exitCode);

        try
        {
            using var _ = JsonDocument.Parse(stdout);
        }
        catch (JsonException ex)
        {
            throw new Xunit.Sdk.XunitException(
                $"`mloop {string.Join(' ', args)}` exited 0 but stdout does not parse: {ex.Message}"
                + $"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
        }
    }

    /// <summary>
    /// A <c>--json</c> command owes its consumer a parseable document at <b>every</b> exit, including
    /// the ones it reaches by failing.
    /// </summary>
    /// <remarks>
    /// The cases above all succeed, and a command can be correct on its success path while leaving
    /// stdout empty on the twelve ways it can return non-zero — <c>JSON.parse("")</c> fails exactly
    /// as prose does. These commands are pointed at a project with no trained model and no
    /// experiments, so each one takes a failure exit, and each one still owes a document.
    /// </remarks>
    [Theory]
    [InlineData("predict", "datasets/train.csv", "--json")]
    [InlineData("evaluate", "--json")]
    [InlineData("promote", "--latest", "--json")]
    [InlineData("compare", "exp-001", "exp-002", "--json")]
    [InlineData("status", "--json")]
    [InlineData("list", "--json")]
    [InlineData("validate", "--json")]
    public async Task StdoutParsesAtEveryExitIncludingFailure(params string[] args)
    {
        var (exitCode, stdout, stderr) = await RunAsync(args);

        try
        {
            using var _ = JsonDocument.Parse(stdout);
        }
        catch (JsonException ex)
        {
            throw new Xunit.Sdk.XunitException(
                $"`mloop {string.Join(' ', args)}` exited {exitCode} and stdout does not parse: {ex.Message}"
                + $" | stdout: {stdout} | stderr: {stderr}");
        }
    }

    /// <summary>
    /// A document that parses can still fail to say what happened. <c>evaluate</c> exits 0 without a
    /// production model — skipping is a reported outcome here, not a failure — and the human is told
    /// so in as many words. The document must say it too.
    /// </summary>
    /// <remarks>
    /// Before this, the skip serialized as every field null and exit 0, which is also what an
    /// evaluation that ran and produced nothing would look like. The consumer had no way to tell the
    /// two apart, while the person watching the terminal was told outright. That asymmetry — a fact
    /// on the human channel and not on the machine one — is the same defect class as narration
    /// landing on the document's stream, reached from the other side.
    /// </remarks>
    [Fact]
    public async Task EvaluateSaysInTheDocumentThatItSkipped()
    {
        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "--json");

        Assert.Equal(0, exitCode);

        using var document = JsonDocument.Parse(stdout);
        var skipped = document.RootElement.GetProperty("skipped");

        Assert.Equal(JsonValueKind.String, skipped.ValueKind);
        Assert.Equal("no-production-model", skipped.GetString());
    }

    [Fact]
    public async Task TheFixtureActuallyMakesTheLoaderNarrate()
    {
        // Negative control for the fixture itself: without an index column to drop, the loader says
        // nothing and every assertion above would hold no matter where narration went. Human mode is
        // also where that line still belongs — the fix moved it off the document's stream, not away
        // from the user.
        var (exitCode, stdout, stderr) = await RunAsync("info", "datasets/train.csv");

        Assert.Equal(0, exitCode);
        Assert.Contains("Removed index column(s)", stdout + stderr, StringComparison.Ordinal);
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
}
