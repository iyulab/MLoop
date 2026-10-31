using System.CommandLine;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>validate</c> collects its findings as <c>List&lt;ValidationError/Warning&gt;</c> before
/// rendering them, so a <c>--json</c> branch just serializes the same data instead of the human
/// report — no separate computation path to drift from it. Exercises the real command tree
/// (<see cref="Program.BuildRootCommand"/>), same approach as <c>ListCommandTrialsTests</c>.
/// </summary>
[Collection("FileSystem")]
public class ValidateCommandJsonTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;

    public ValidateCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-validate-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
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

    private void WriteYaml(string content) =>
        File.WriteAllText(Path.Combine(_testProjectRoot, "mloop.yaml"), content);

    // The training data is a workbook: its header is the table's (a, b, y), not the first line of a zip.
    private void WriteWorkbookProject()
    {
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets"));
        WorkbookFile.Write(Path.Combine(_testProjectRoot, "datasets", "train.xlsx"), ["a", "b", "y"],
            Enumerable.Range(1, 30).Select(i => new[] { i * 1.0, i * 2.0, i * 3.0 }));
        WriteYaml("project: p\nmodels:\n  default:\n    task: regression\n    label: y\ndata:\n  train: datasets/train.xlsx\n");
    }

    [Fact]
    public async Task Validate_Json_ALabelInAWorkbooksHeader_IsFound()
    {
        WriteWorkbookProject();

        var (exitCode, stdout, _) = await CliRunner.RunAsync("validate", "--json");

        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Equal(0, exitCode);
        Assert.DoesNotContain(doc.RootElement.GetProperty("errors").EnumerateArray(),
            e => e.GetProperty("message").GetString()!.Contains("not found in training CSV", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PrepRun_OnAWorkbook_RunsThePipelineOverItsRowsAndWritesCsv()
    {
        WriteWorkbookProject();
        WriteYaml(File.ReadAllText(Path.Combine(_testProjectRoot, "mloop.yaml")).Replace("    label: y\n",
            "    label: y\n    prep:\n      - type: normalize\n        columns: [a, b]\n        method: minmax\n"));

        var (exitCode, stdout, _) = await CliRunner.RunAsync("prep", "run", "--json");

        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Equal(0, exitCode);
        Assert.Equal(30, doc.RootElement.GetProperty("rowsBefore").GetInt32());
        Assert.Equal(30, doc.RootElement.GetProperty("rowsAfter").GetInt32());
        // named for what it holds, not for what was read
        Assert.Equal("train_prep.csv", Path.GetFileName(doc.RootElement.GetProperty("output").GetString()));
        Assert.StartsWith("a,b,y", File.ReadAllText(doc.RootElement.GetProperty("output").GetString()!).TrimStart('\uFEFF'), StringComparison.Ordinal);
    }

    [Fact]
    public async Task FeaturesSelect_DropAndKeep_ReadTheColumnsOfAWorkbook()
    {
        WriteWorkbookProject();

        var (dropExit, dropOut, _) = await CliRunner.RunAsync("features", "select", "--drop", "b", "--json");
        using (var drop = System.Text.Json.JsonDocument.Parse(dropOut))
        {
            Assert.Equal(0, dropExit);
            Assert.Equal(["b"], drop.RootElement.GetProperty("applied").GetProperty("columns").EnumerateArray().Select(c => c.GetString()));
        }

        await CliRunner.RunAsync("features", "select", "--reset", "--json");
        var (keepExit, keepOut, _) = await CliRunner.RunAsync("features", "select", "--keep", "a", "--json");
        using var keep = System.Text.Json.JsonDocument.Parse(keepOut);
        Assert.Equal(0, keepExit);
        // the complement is the table's other column - not the bytes of the zip written into mloop.yaml as a column name
        Assert.Equal(["b"], keep.RootElement.GetProperty("ignored").EnumerateArray().Select(c => c.GetString()));
        Assert.DoesNotContain("PK", File.ReadAllText(Path.Combine(_testProjectRoot, "mloop.yaml")), StringComparison.Ordinal);
    }

    /// <summary>
    /// Loading the old single-model format warns. The warning was written to stdout ahead of the document, so
    /// <c>validate --json</c> printed something no JSON parser reads, with an empty <c>warnings</c>.
    /// </summary>
    [Fact]
    public async Task Validate_Json_OldFormat_IsADocument_CarryingTheFormatWarning()
    {
        WriteYaml("project_name: p\ntask: regression\nlabel_column: y\n");

        var (exitCode, stdout, _) = await CliRunner.RunAsync("validate", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var warning = Assert.Single(doc.RootElement.GetProperty("warnings").EnumerateArray(),
            w => w.GetProperty("message").GetString()!.Contains("old mloop.yaml format", StringComparison.Ordinal));
        Assert.Equal("mloop.yaml", warning.GetProperty("path").GetString());
    }

    [Fact]
    public async Task Validate_Json_ValidConfig_ReportsValidTrue()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: regression
                label: target
            """);

        var (exitCode, stdout, _) = await CliRunner.RunAsync("validate", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.True(doc.RootElement.GetProperty("valid").GetBoolean());
        Assert.Empty(doc.RootElement.GetProperty("errors").EnumerateArray());
    }

    [Fact]
    public async Task Validate_Json_InvalidTaskType_ReportsErrorWithPath()
    {
        WriteYaml("""
            project: test
            models:
              default:
                task: not-a-real-task
                label: target
            """);

        var (exitCode, stdout, _) = await CliRunner.RunAsync("validate", "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.False(doc.RootElement.GetProperty("valid").GetBoolean());
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Single(errors);
        Assert.Equal("models.default.task", errors[0].GetProperty("path").GetString());
    }

    [Fact]
    public async Task Validate_Json_MissingConfigFile_ReportsError()
    {
        var (exitCode, stdout, _) = await CliRunner.RunAsync("validate", "--json");

        Assert.Equal(1, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.False(doc.RootElement.GetProperty("valid").GetBoolean());
        var errors = doc.RootElement.GetProperty("errors").EnumerateArray().ToList();
        Assert.Contains(errors, e => e.GetProperty("path").GetString() == "mloop.yaml");
    }
}
