using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

/// <summary>
/// A metric the evaluator could not measure is stored as the worst value a double holds, so that it can never win a
/// ranking. <c>train</c> and <c>evaluate</c> say "not measured" for it; <c>list</c>, <c>compare</c>, <c>status</c> and
/// <c>promote</c> printed the number — three hundred digits across the table that was meant to be read at a glance.
/// </summary>
[Collection("FileSystem")]
public sealed class UndefinedMetricDisplayTests : IDisposable
{
    private readonly string _root;
    private readonly string _originalDirectory;

    /// <summary>The first digits of <see cref="double.MaxValue"/> and of its negation, however either is formatted.</summary>
    private const string SentinelDigits = "1797693134862315";

    public UndefinedMetricDisplayTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _root = Path.Combine(Path.GetTempPath(), "mloop-undefined-metric-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalDirectory);
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    private static async Task SaveExperimentAsync(string id, Dictionary<string, double> metrics)
    {
        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var path = store.GetExperimentPath("default", id);
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
        await store.SaveAsync("default", new ExperimentData
        {
            ModelName = "default",
            ExperimentId = id,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig { DataFile = "train.csv", LabelColumn = "y", Metric = "r_squared", TimeLimitSeconds = 60, TestSplit = 0.2 },
            Metrics = metrics,
        }, CancellationToken.None);
    }

    /// <summary>What a terminal shows: colour codes dropped, and a line the table wrapped joined back.</summary>
    private static string Plain(string text) => Regex.Replace(Regex.Replace(text, "\u001b\\[[0-9;]*m", ""), @"\s+", " ");

    private static readonly Dictionary<string, double> Unmeasured = new()
    {
        ["r_squared"] = double.MinValue,
        ["rmse"] = double.MaxValue,
        ["mae"] = double.MaxValue,
    };

    private static readonly Dictionary<string, double> Measured = new()
    {
        ["r_squared"] = 0.8,
        ["rmse"] = 1.5,
        ["mae"] = 1.1,
    };

    private async Task<string> RunAsync(params string[] args)
    {
        await SaveExperimentAsync("exp-001", Unmeasured);
        await SaveExperimentAsync("exp-002", Measured);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync(args);

        Assert.Equal(0, exitCode);
        return Plain(stdout + stderr);
    }

    [Fact]
    public async Task List_SaysNotMeasured_NotTheSentinelNumber()
    {
        var said = await RunAsync("list");

        Assert.Contains("not measured", said);
        Assert.Contains("0.8000", said);
        Assert.DoesNotContain(SentinelDigits, said);
    }

    [Fact]
    public async Task Status_SaysNotMeasured_NotTheSentinelNumber()
    {
        // Only the unmeasured experiment, so that it is the project's best.
        await SaveExperimentAsync("exp-001", Unmeasured);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("status");

        Assert.Equal(0, exitCode);
        var said = Plain(stdout + stderr);
        Assert.Contains("not measured", said);
        Assert.DoesNotContain(SentinelDigits, said);
    }

    [Fact]
    public async Task Compare_SaysNotMeasured_ForEveryMetricItCouldNotMeasure()
    {
        var said = await RunAsync("compare", "exp-001", "exp-002");

        Assert.Contains("not measured", said);
        Assert.DoesNotContain(SentinelDigits, said);
        Assert.Contains("0.8000", said);
    }

    [Fact]
    public async Task Compare_OfExperimentsThatAreAllUnmeasured_ShowsNoNumber()
    {
        await SaveExperimentAsync("exp-001", Unmeasured);
        await SaveExperimentAsync("exp-002", Unmeasured);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("compare", "exp-001", "exp-002");

        Assert.Equal(0, exitCode);
        var said = Plain(stdout + stderr);
        Assert.Contains("not measured", said);
        Assert.DoesNotContain(SentinelDigits, said);
    }

    [Fact]
    public async Task Promote_ShowsTheExperimentItIsAboutToPromote_WithoutTheSentinelNumber()
    {
        await SaveExperimentAsync("exp-001", Unmeasured);

        // The gate refuses an unmeasured model, so the force is what lets the details table be reached.
        var (_, stdout, stderr) = await CliRunner.RunAsync("promote", "exp-001", "--force");

        var said = Plain(stdout + stderr);
        Assert.Contains("exp-001", said);
        Assert.Contains("not measured", said);
        Assert.DoesNotContain(SentinelDigits, said);
    }

    [Fact]
    public async Task TheReportFile_SaysNotMeasured_NotTheSentinelNumber()
    {
        await SaveExperimentAsync("exp-001", Unmeasured);

        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var experiment = await store.LoadAsync("default", "exp-001", CancellationToken.None);
        Assert.NotNull(experiment);

        var report = ExperimentReport.Render(experiment!, [], _root, "0.0.0");

        Assert.Contains("not measured", report);
        Assert.DoesNotContain(SentinelDigits, report);
    }
}
