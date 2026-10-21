using MLoop.CLI.Commands;
using MLoop.Core.Runtime;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Tests.Commands;

/// <summary>
/// What three commands showed that was not so: a prep plan's columns in doubled brackets, an unmeasured
/// accuracy as 0.0%, and a runtime list in the help that missed a runtime the registry knows.
/// </summary>
[Collection("FileSystem")]
public sealed class CommandSurfaceRenderingTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-render-" + Guid.NewGuid().ToString("N"));

    public CommandSurfaceRenderingTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Fact]
    public async Task APrepPlan_ShowsItsColumnsAsWritten()
    {
        var (initExit, _, _) = await CliRunner.RunAsync("init", "p", "--task", "regression");
        Assert.Equal(0, initExit);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var (exit, stdout, _) = await CliRunner.RunAsync("prep", "plan", "--set", "normalize:z-score", "--columns", "x");

        Assert.Equal(0, exit);
        Assert.Contains("columns: [x]", stdout);
        Assert.DoesNotContain("[[x]]", stdout);
    }

    [Fact]
    public void AnUnmeasuredCondition_ShowsADash_NotZero()
    {
        var unmeasured = new ConditionResult(RetrainingDefaults.Accuracy(), IsMet: false, CurrentValue: null, "No accuracy data");
        var measured = unmeasured with { CurrentValue = 0.42 };

        Assert.Equal("[grey]-[/]", TriggerCommand.FormatCurrentValue(unmeasured));
        Assert.Contains("42.0%", TriggerCommand.FormatCurrentValue(measured));
    }

    [Fact]
    public async Task TheRuntimeHelp_NamesEveryRuntimeTheRegistryKnows()
    {
        var (_, stdout, _) = await CliRunner.RunAsync("runtime", "install", "--help");

        foreach (var runtime in RuntimeRegistry.All)
            Assert.Contains(runtime.Id, stdout);
    }
}
