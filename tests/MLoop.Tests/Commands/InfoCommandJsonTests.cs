using System.CommandLine;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>info --json</c> reports the same profiling data <c>info</c>'s human report shows —
/// column classification, the DataLens profile, and (when <c>--analyze</c> is passed) the deep
/// analysis result serialized as-is. Exercises the real command tree
/// (<see cref="Program.BuildRootCommand"/>), same approach as the other <c>--json</c> command tests.
/// Also the first test in this repo to run <c>info</c> through the real Spectre-rendering path with
/// <c>--analyze</c> — that run is what surfaced two serialization gaps in this new code (NaN/Infinity
/// doubles, and DataLens's 2D-array report fields) that a string-level unit test never would have.
/// </summary>
[Collection("FileSystem")]
public class InfoCommandJsonTests : IDisposable
{
    private readonly string _testDir;
    private readonly string _originalDirectory;

    public InfoCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-info-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
        Directory.SetCurrentDirectory(_testDir);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }

        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    private string WriteDataCsv() =>
        WriteCsv("data.csv", "a,b,Label\n1,2.5,yes\n2,,no\n3,4.5,yes\n4,5.5,no\n5,,yes\n");

    private string WriteCsv(string name, string content)
    {
        var path = Path.Combine(_testDir, name);
        File.WriteAllText(path, content);
        return path;
    }

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
    public async Task Info_Json_FileNotFound_ReportsError()
    {
        var (exitCode, stdout) = await RunAsync("info", "nope.csv", "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetString()!.Contains("not found"));
    }

    [Fact]
    public async Task Info_Json_EmptyFile_ReportsErrorAndNonZeroExit()
    {
        var path = WriteCsv("empty.csv", "");

        var (exitCode, stdout) = await RunAsync("info", path, "--json");

        Assert.Equal(1, exitCode); // sibling to the file-not-found case above, not a silent success
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetString()!.Contains("empty"));
    }

    [Fact]
    public async Task Info_Json_Profile_ReportsColumnsAndLabelDistribution()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout) = await RunAsync("info", path, "--label", "Label", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.False(doc.RootElement.GetProperty("analyze").GetBoolean());
        Assert.False(doc.RootElement.TryGetProperty("analysis", out _)); // null omitted, --analyze not passed

        var columns = doc.RootElement.GetProperty("columns").EnumerateArray().ToList();
        Assert.Equal(3, columns.Count);
        var labelColumn = columns.Single(c => c.GetProperty("name").GetString() == "Label");
        Assert.Equal("Label", labelColumn.GetProperty("purpose").GetString());

        var labelDistribution = doc.RootElement.GetProperty("labelDistribution");
        Assert.Equal(3, labelDistribution.GetProperty("yes").GetInt32());
        Assert.Equal(2, labelDistribution.GetProperty("no").GetInt32());
    }

    [Fact]
    public async Task Info_Json_Analyze_ReportsDeepAnalysisResult()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout) = await RunAsync("info", path, "--label", "Label", "--analyze", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.True(doc.RootElement.GetProperty("analyze").GetBoolean());

        var analysis = doc.RootElement.GetProperty("analysis");
        // Regression guard: DataLens's CorrelationReport.Matrix is a double[,], which
        // System.Text.Json has no built-in support for — this is what verifies the custom
        // converter actually runs end-to-end rather than only in isolation.
        var matrix = analysis.GetProperty("correlation").GetProperty("matrix");
        Assert.Equal(2, matrix.GetArrayLength());
        Assert.Equal(2, matrix[0].GetArrayLength());
    }

    [Fact]
    public async Task Info_Human_Unaffected_ByJsonChanges()
    {
        var path = WriteDataCsv();

        var (exitCode, stdout) = await RunAsync("info", path, "--label", "Label");

        Assert.Equal(0, exitCode);
        Assert.Contains("Column Information", stdout);
        Assert.Contains("Label", stdout);
        Assert.DoesNotContain("{", stdout); // no JSON leaked into the human report
    }
}
