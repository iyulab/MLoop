using System.Globalization;
using System.Text;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Hooks;
using MLoop.Extensibility.Hooks;

namespace MLoop.Tests.Commands;

/// <summary>
/// Every stage a hook can be written for is run by a command. <c>mloop new hook</c> scaffolds four stages and the hook
/// interface documents them, but only the two training stages ever ran: a gate written for <c>post-evaluate</c> blocked nothing,
/// and nothing said so. The guard is behavioural — a hook that leaves a marker is written for every value of
/// <see cref="HookType"/>, the commands run, and each marker must exist — so a fifth stage added without a command that runs
/// it fails here, whatever its name.
/// </summary>
[Collection("FileSystem")]
public sealed class HookStagesRunContractTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-hook-stages-" + Guid.NewGuid().ToString("N"));

    public HookStagesRunContractTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Rows(int rows, bool labelled)
    {
        var csv = new StringBuilder(labelled ? "x,z,y\n" : "x,z\n");
        var rng = new Random(7);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var z = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{z:F4}");
            if (labelled) csv.Append(CultureInfo.InvariantCulture, $",{3 * x + 2 * z:F4}");
            csv.Append('\n');
        }
        return csv.ToString();
    }

    private static string StageFolder(string projectRoot, HookType stage)
        => new HookEngine(projectRoot, SilentLogger.Instance).GetHookDirectory(stage);

    /// <summary>A hook that records that it ran (and with what it was given), then lets the command go on — or refuses.</summary>
    private static void WriteHook(string projectRoot, HookType stage, bool refuse = false)
    {
        var folder = StageFolder(projectRoot, stage);
        Directory.CreateDirectory(folder);
        var result = refuse ? "HookResult.Abort(\"refused on purpose\")" : "HookResult.Continue()";
        File.WriteAllText(Path.Combine(folder, "marker.cs"), $$"""
            using System.IO;
            using System.Threading.Tasks;
            using MLoop.Extensibility.Hooks;

            public class MarkerHook : IMLoopHook
            {
                public string Name => "Marker";

                public Task<HookResult> ExecuteAsync(HookContext ctx)
                {
                    var seen = ctx.GetMetadata<string>("DataFile", "") is { Length: > 0 } file && File.Exists(file) ? "file" : "nofile";
                    var metrics = ctx.Metrics is null ? "nometrics" : "metrics";
                    File.WriteAllText(Path.Combine(ctx.ProjectRoot, "marker-" + ctx.HookType + ".txt"), seen + " " + metrics);
                    return Task.FromResult({{result}});
                }
            }
            """);
    }

    private static string Marker(HookType stage) => $"marker-{stage}.txt";

    private async Task<string> TrainedProjectAsync()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "regression")).ExitCode);
        var root = Path.Combine(_workspace, "p");
        Directory.SetCurrentDirectory(root);
        await File.WriteAllTextAsync("datasets/train.csv", Rows(200, labelled: true));
        await File.WriteAllTextAsync("datasets/predict.csv", Rows(10, labelled: false));
        return root;
    }

    private static async Task TrainAndPromoteAsync()
    {
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);
    }

    [Fact]
    public async Task EveryStageOfHookType_IsRunByACommand()
    {
        var root = await TrainedProjectAsync();
        var stages = Enum.GetValues<HookType>();
        foreach (var stage in stages)
            WriteHook(root, stage);

        await TrainAndPromoteAsync();
        var predicted = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "out.csv");
        Assert.True(predicted.ExitCode == 0, predicted.Stdout + predicted.Stderr);
        var evaluated = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/train.csv");
        Assert.True(evaluated.ExitCode == 0, evaluated.Stdout + evaluated.Stderr);

        Assert.True(stages.Length >= 4, "the guard must see the four stages that exist");
        foreach (var stage in stages)
            Assert.True(File.Exists(Marker(stage)), $"No command ran the {stage} hook — a stage `mloop new hook` writes must be run by one.");

        // What each batch stage was given: the file it is about to score or just scored, and metrics only after the measuring.
        Assert.Equal("file nometrics", await File.ReadAllTextAsync(Marker(HookType.PrePredict)));
        Assert.Equal("file metrics", await File.ReadAllTextAsync(Marker(HookType.PostEvaluate)));
    }

    [Fact]
    public async Task APrePredictHook_ThatRefuses_ScoresNothing()
    {
        var root = await TrainedProjectAsync();
        await TrainAndPromoteAsync();
        WriteHook(root, HookType.PrePredict, refuse: true);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "out.csv");

        Assert.NotEqual(0, exit);
        Assert.False(File.Exists("out.csv"), "a refused input is not scored");
        Assert.Contains("refused on purpose", stdout + stderr, StringComparison.Ordinal);
    }

    [Fact]
    public async Task APostEvaluateHook_ThatRefuses_FailsTheCommand_AndJsonSaysSo()
    {
        var root = await TrainedProjectAsync();
        await TrainAndPromoteAsync();
        WriteHook(root, HookType.PostEvaluate, refuse: true);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/train.csv");
        Assert.NotEqual(0, exit);
        Assert.Contains("refused on purpose", stdout + stderr, StringComparison.Ordinal);

        var json = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/train.csv", "--json");
        Assert.NotEqual(0, json.ExitCode);
        using var document = System.Text.Json.JsonDocument.Parse(json.Stdout.Trim());
        Assert.Contains("refused on purpose", document.RootElement.GetProperty("error").GetString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(HookType.PreTrain, "mloop train")]
    [InlineData(HookType.PostTrain, "mloop train")]
    [InlineData(HookType.PrePredict, "mloop predict")]
    [InlineData(HookType.PostEvaluate, "mloop evaluate")]
    public void TheScaffoldNamesTheCommandThatRunsTheStage(HookType stage, string command)
        => Assert.Equal(command, NewHookCommand.RunningCommand(stage));

    [Fact]
    public async Task TheHooksReadmeInit_Writes_NamesAnOptionlessCommandTheCliAccepts_AndEveryStage()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "regression")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var readme = await File.ReadAllTextAsync(Path.Combine(".mloop", "scripts", "hooks", "README.md"));

        // The command the README tells the user to type is parsed by the CLI as written.
        Assert.DoesNotContain("--type", readme, StringComparison.Ordinal);
        var line = readme.Split('\n').Single(l => l.Contains("mloop new hook <name>", StringComparison.Ordinal));
        var stage = line.Split("mloop new hook <name> ")[1].Split(' ', '`')[0];
        var (exit, stdout, stderr) = await CliRunner.RunAsync("new", "hook", "readme-form", stage);
        Assert.True(exit == 0, stdout + stderr);

        foreach (var s in new[] { "pre-train", "post-train", "pre-predict", "post-evaluate" })
            Assert.Contains(s + "/", readme, StringComparison.Ordinal);
    }
}
