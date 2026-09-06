using System.CommandLine;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>leaderboard.json</c> (a search's ranked trial history) has existed since 0.30.0 with no
/// CLI surface — a consumer had to read the file directly. These exercise <c>mloop list --trials</c>
/// through the real command tree (<see cref="Program.BuildRootCommand"/>), the same path an end-to-end
/// smoke test (`mloop train --task clustering` then `mloop list --trials exp-001`) verified manually.
/// </summary>
[Collection("FileSystem")]
public class ListCommandTrialsTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;

    public ListCommandTrialsTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-list-trials-test-" + Guid.NewGuid());
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

    private async Task CreateExperimentWithTrialsAsync(string modelName, string experimentId)
    {
        var trials = new List<TrialRecord>
        {
            new()
            {
                TrialNumber = 1,
                Trainer = new TrainerDescriptor { Name = "KMeans", Parameters = [new("k", "2")] },
                Metrics = new Dictionary<string, double> { ["davies_bouldin_index"] = 0.30 },
                RuntimeSeconds = 0.3
            },
            new()
            {
                TrialNumber = 2,
                Trainer = new TrainerDescriptor { Name = "KMeans", Parameters = [new("k", "3")] },
                Metrics = new Dictionary<string, double> { ["davies_bouldin_index"] = 0.05 },
                RuntimeSeconds = 0.1
            }
        };

        var experiment = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "clustering",
            Config = new ExperimentConfig
            {
                DataFile = "test-data.csv",
                LabelColumn = "",
                TimeLimitSeconds = 60,
                Metric = "davies_bouldin_index",
                TestSplit = 0.2
            },
            Metrics = new Dictionary<string, double> { ["davies_bouldin_index"] = 0.05 },
            Trials = trials,
            RankingMetric = "davies_bouldin_index"
        };

        await _experimentStore.SaveAsync(modelName, experiment, CancellationToken.None);
    }

    private static async Task<(int ExitCode, string Stdout)> RunAsync(params string[] args)
    {
        // Console.SetOut alone doesn't reach Spectre's rendering: AnsiConsole.Console is a static,
        // lazily-constructed default instance that binds to whichever Console.Out was current at
        // its first use across the whole test run — a later Console.SetOut here doesn't retroactively
        // redirect it. Rebinding AnsiConsole.Console explicitly is what MachineOutputScope itself
        // does for the same reason.
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
    public async Task List_Trials_RanksBestFirst()
    {
        // The lower-is-better trial (k=3, 0.05) must rank above k=2 (0.30) — leaderboard.json is
        // already ranked by ExperimentStore.SaveTrialsAsync; this checks the CLI surface preserves
        // that order rather than re-sorting (or not sorting) it.
        await CreateExperimentWithTrialsAsync(ConfigDefaults.DefaultModelName, "exp-001");

        var (exitCode, stdout) = await RunAsync("list", "--trials", "exp-001");

        Assert.Equal(0, exitCode);
        var kThreeIndex = stdout.IndexOf("KMeans (k=3)", StringComparison.Ordinal);
        var kTwoIndex = stdout.IndexOf("KMeans (k=2)", StringComparison.Ordinal);
        Assert.True(kThreeIndex >= 0 && kTwoIndex >= 0, $"Expected both trainers in output:\n{stdout}");
        Assert.True(kThreeIndex < kTwoIndex, "Lower davies_bouldin_index (k=3) should rank above k=2");
    }

    [Fact]
    public async Task List_Trials_Json_IsWellFormedAndRanked()
    {
        await CreateExperimentWithTrialsAsync(ConfigDefaults.DefaultModelName, "exp-001");

        var (exitCode, stdout) = await RunAsync("list", "--trials", "exp-001", "--json");

        Assert.Equal(0, exitCode);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        var trials = doc.RootElement.GetProperty("trials");
        Assert.Equal(2, trials.GetArrayLength());
        Assert.Equal("KMeans (k=3)", trials[0].GetProperty("trainerName").GetString());
    }

    [Fact]
    public async Task List_Trials_UnknownExperiment_ReportsNotFound()
    {
        var (exitCode, _) = await RunAsync("list", "--trials", "exp-999");

        Assert.Equal(1, exitCode);
    }
}
