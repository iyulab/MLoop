using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>promote</c>, <c>rollback</c> and <c>promote --history</c> through the real command tree: every
/// change of the production slot is in the history, with how it happened and the gate's outcome.
/// </summary>
[Collection("FileSystem")]
public sealed class PromotionHistoryCommandTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _root = Path.Combine(Path.GetTempPath(), "mloop-history-" + Guid.NewGuid().ToString("N"));

    public PromotionHistoryCommandTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_root, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static async Task SeedAsync(string experimentId, double rSquared)
    {
        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var path = store.GetExperimentPath("default", experimentId);
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
        await store.SaveAsync("default", new ExperimentData
        {
            ModelName = "default", ExperimentId = experimentId, Timestamp = DateTime.UtcNow, Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig { DataFile = "train.csv", LabelColumn = "y", Metric = "r_squared", TimeLimitSeconds = 10, TestSplit = 0.2 },
            Metrics = new Dictionary<string, double> { ["r_squared"] = rSquared },
        }, CancellationToken.None);
    }

    [Fact]
    public async Task EveryChangeOfProduction_IsListed_WithHowAndTheGate()
    {
        await SeedAsync("exp-001", 0.7);
        await SeedAsync("exp-002", 0.9);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001", "--json")).ExitCode);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-002", "--json")).ExitCode);
        var (rollbackExit, rollbackOut, rollbackErr) = await CliRunner.RunAsync("rollback", "--json");
        Assert.True(rollbackExit == 0, rollbackOut + rollbackErr);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("promote", "--history", "--json");

        Assert.True(exit == 0, stdout + stderr);
        var promotions = JsonDocument.Parse(stdout).RootElement.GetProperty("promotions").EnumerateArray().ToList();
        Assert.Equal(["exp-001", "exp-002", "exp-001"], promotions.Select(p => p.GetProperty("experimentId").GetString()));
        Assert.Equal(["promote", "promote", "rollback"], promotions.Select(p => p.GetProperty("action").GetString()));
        Assert.Equal(["passed", "passed", "not-applied"], promotions.Select(p => p.GetProperty("gate").GetString()));
        Assert.Equal("exp-002", promotions[2].GetProperty("previousExperimentId").GetString());
        Assert.Equal(0.9, promotions[1].GetProperty("metricValue").GetDouble());
    }

    [Fact]
    public async Task History_ChangesNothing_SoItTakesNoSelector()
    {
        await SeedAsync("exp-001", 0.7);

        var (exit, stdout, _) = await CliRunner.RunAsync("promote", "exp-001", "--history", "--json");

        Assert.Equal(1, exit);
        Assert.Contains("--history", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());
        Assert.False(Directory.Exists(Path.Combine(_root, "models", "default", "production")));
    }

    [Fact]
    public async Task WithNothingPromoted_TheHistoryIsEmpty()
    {
        var (exit, stdout, _) = await CliRunner.RunAsync("promote", "--history", "--json");

        Assert.Equal(0, exit);
        Assert.Equal(0, JsonDocument.Parse(stdout).RootElement.GetProperty("promotions").GetArrayLength());
    }
}
