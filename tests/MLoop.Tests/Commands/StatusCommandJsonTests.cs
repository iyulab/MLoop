using System.CommandLine;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>status</c> now collects <c>ModelStatusRow</c>/<c>DataFileRow</c> once before rendering
/// (mirroring the <c>validate</c>/<c>evaluate --json</c> pattern) — a <c>--json</c> branch serializes
/// the same rows instead of re-deriving them. Exercises the real command tree
/// (<see cref="Program.BuildRootCommand"/>), same approach as <c>ListCommandTrialsTests</c>.
/// </summary>
[Collection("FileSystem")]
public class StatusCommandJsonTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;

    public StatusCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-status-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);

        Directory.SetCurrentDirectory(_testProjectRoot);

        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
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

    private async Task CreateExperimentAsync(
        string modelName, string experimentId, string status, double? metric)
    {
        var experiment = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = status,
            Task = "binary-classification",
            Config = new ExperimentConfig
            {
                DataFile = "test-data.csv",
                LabelColumn = "target",
                TimeLimitSeconds = 60,
                Metric = "accuracy",
                TestSplit = 0.2
            },
            Metrics = metric.HasValue
                ? new Dictionary<string, double> { ["accuracy"] = metric.Value }
                : new Dictionary<string, double>()
        };

        await _experimentStore.SaveAsync(modelName, experiment, CancellationToken.None);
    }

    // Same AnsiConsole.Console rebinding as ListCommandTrialsTests.RunAsync — Console.SetOut alone
    // doesn't reach Spectre's ambient renderer.
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

            var exitCode = await Program.ExecuteAsync(Program.BuildRootCommand(), args);
            return (exitCode, buffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }

    [Fact]
    public async Task Status_Json_NoModels_ReportsEmptyStructure()
    {
        var (exitCode, stdout) = await RunAsync("status", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Empty(doc.RootElement.GetProperty("models").EnumerateArray());
        Assert.Equal(0, doc.RootElement.GetProperty("summary").GetProperty("modelsCount").GetInt32());
    }

    [Fact]
    public async Task Status_Json_ReportsModelCountsAndBestMetric()
    {
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-001", "Completed", 0.80);
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-002", "Completed", 0.92);
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-003", "Failed", null);

        var (exitCode, stdout) = await RunAsync("status", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var models = doc.RootElement.GetProperty("models").EnumerateArray().ToList();
        Assert.Single(models);
        var model = models[0];
        Assert.Equal(ConfigDefaults.DefaultModelName, model.GetProperty("modelName").GetString());
        Assert.Equal(3, model.GetProperty("totalExperiments").GetInt32());
        Assert.Equal(2, model.GetProperty("completedExperiments").GetInt32());
        Assert.False(model.GetProperty("hasProduction").GetBoolean());
        // Higher-is-better (binary-classification/accuracy): exp-002's 0.92 must win, not exp-001's 0.80.
        Assert.Equal(0.92, model.GetProperty("bestMetric").GetDouble());

        var summary = doc.RootElement.GetProperty("summary");
        Assert.Equal(1, summary.GetProperty("modelsCount").GetInt32());
        Assert.Equal(3, summary.GetProperty("totalExperiments").GetInt32());
        Assert.Equal(2, summary.GetProperty("completedExperiments").GetInt32());
        Assert.Equal(1, summary.GetProperty("failedExperiments").GetInt32());
    }

    [Fact]
    public async Task Status_Json_WithoutVerbose_OmitsDataFiles()
    {
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-001", "Completed", 0.5);

        var (exitCode, stdout) = await RunAsync("status", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Equal(System.Text.Json.JsonValueKind.Null, doc.RootElement.GetProperty("dataFiles").ValueKind);
    }

    [Fact]
    public async Task Status_Json_Verbose_IncludesDataFiles()
    {
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-001", "Completed", 0.5);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets"));
        File.WriteAllText(Path.Combine(_testProjectRoot, "datasets", "train.csv"), "a,b\n1,2\n");

        var (exitCode, stdout) = await RunAsync("status", "--verbose", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var dataFiles = doc.RootElement.GetProperty("dataFiles").EnumerateArray().ToList();
        Assert.Contains(dataFiles, d => d.GetProperty("type").GetString() == "Train" && d.GetProperty("exists").GetBoolean());
        Assert.Contains(dataFiles, d => d.GetProperty("type").GetString() == "Test" && !d.GetProperty("exists").GetBoolean());
    }

    [Fact]
    public async Task Status_Human_StillRendersTable()
    {
        // Guards against the refactor (collect-then-render) silently changing the human path.
        await CreateExperimentAsync(ConfigDefaults.DefaultModelName, "exp-001", "Completed", 0.5);

        var (exitCode, stdout) = await RunAsync("status");

        Assert.Equal(0, exitCode);
        Assert.Contains("Models Overview", stdout);
        Assert.Contains(ConfigDefaults.DefaultModelName, stdout);
    }
}
