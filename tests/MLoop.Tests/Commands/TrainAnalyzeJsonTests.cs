using System.Text.Json;
using MLoop.CLI.Commands;
using MLoop.Core.DataQuality;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --analyze-data --json</c> answers on the event stream. The data check is the whole of that run — it
/// trains nothing — and its findings reached the console only, which <c>--json</c> silences: a consumer got an
/// empty stdout and a zero exit, the same reading as a check that found nothing. Now the findings are an
/// <c>analysis</c> event, and a script written by <c>--generate-script</c> a <c>script</c> event.
/// </summary>
public class TrainAnalyzeJsonTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-analyze-json-" + Guid.NewGuid().ToString("N"));

    public TrainAnalyzeJsonTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { /* nothing left to do */ } }

        if (Directory.Exists(_workspace))
        {
            try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }
    }

    private static List<JsonElement> Events(string stdout) =>
        [.. stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(line => JsonDocument.Parse(line).RootElement)];

    private async Task InitProjectAsync(string csv)
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "regression");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", csv);
    }

    private const string Distinct = "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n";
    private const string WithDuplicates = "x,y\n1,2\n1,2\n1,2\n2,4\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n";

    [Fact]
    public async Task The_findings_are_the_streams_one_analysis_event()
    {
        await InitProjectAsync(WithDuplicates);

        var (exit, stdout, _) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--analyze-data", "--json");

        Assert.True(exit == 0, stdout);
        var e = Assert.Single(Events(stdout));
        Assert.Equal("analysis", e.GetProperty("event").GetString());
        var issue = e.GetProperty("issues").EnumerateArray().Single(i => i.GetProperty("type").GetString() == nameof(DataQualityIssueType.DuplicateRows));
        Assert.Equal("high", issue.GetProperty("severity").GetString());
        Assert.Contains("duplicate", issue.GetProperty("description").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(string.IsNullOrWhiteSpace(issue.GetProperty("suggestedFix").GetString()));
    }

    [Fact]
    public async Task Data_with_no_findings_says_so_with_an_empty_list()
    {
        await InitProjectAsync(Distinct);

        var (exit, stdout, _) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--analyze-data", "--json");

        Assert.True(exit == 0, stdout);
        var e = Assert.Single(Events(stdout));
        Assert.Equal("analysis", e.GetProperty("event").GetString());
        Assert.Equal(0, e.GetProperty("issues").GetArrayLength());
    }

    [Fact]
    public async Task A_generated_script_is_a_script_event_with_the_path_it_was_asked_for()
    {
        await InitProjectAsync(WithDuplicates);

        var (exit, stdout, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--analyze-data", "--generate-script", "clean.cs", "--json");

        Assert.True(exit == 0, stdout);
        var events = Events(stdout);
        Assert.Equal(["analysis", "script"], events.Select(e => e.GetProperty("event").GetString()));
        Assert.Equal("clean.cs", events[1].GetProperty("path").GetString());
        Assert.True(File.Exists("clean.cs"));
    }

    [Fact]
    public async Task A_script_asked_for_and_not_needed_is_a_warning_not_silence()
    {
        await InitProjectAsync(Distinct);

        var (exit, stdout, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--analyze-data", "--generate-script", "clean.cs", "--json");

        Assert.True(exit == 0, stdout);
        var events = Events(stdout);
        Assert.Equal(["analysis", "warning"], events.Select(e => e.GetProperty("event").GetString()));
        Assert.Contains("No preprocessing script was written", events[1].GetProperty("message").GetString());
        Assert.False(File.Exists("clean.cs"));
    }

    [Fact]
    public void An_issue_without_a_column_or_a_fix_omits_them_instead_of_writing_nulls()
    {
        var sink = new StringWriter();
        new TrainJsonEmitter(sink).Analysis([new DataQualityIssue
        {
            Type = DataQualityIssueType.DuplicateRows,
            Severity = IssueSeverity.Low,
            Description = "two rows repeat"
        }]);

        var issue = Assert.Single(Events(sink.ToString()).Single().GetProperty("issues").EnumerateArray().ToList());
        Assert.False(issue.TryGetProperty("column", out _));
        Assert.False(issue.TryGetProperty("suggestedFix", out _));
    }
}
