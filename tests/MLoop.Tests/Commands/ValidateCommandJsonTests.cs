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
