namespace MLoop.Tests.Commands;

/// <summary>
/// What <c>mloop train</c> decides around the training itself: the label checks see the rows
/// training gets (they run after the <c>prep:</c> steps), and a post-train hook's refusal keeps the
/// model out of production.
/// </summary>
public sealed class TrainDecisionOrderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-prep-order-{Guid.NewGuid():N}");
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TrainDecisionOrderTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.CreateDirectory(Path.Combine(_root, "datasets"));
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalDirectory);
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    [Fact]
    public async Task A_prep_step_that_leaves_one_class_is_refused_as_one_class()
    {
        File.WriteAllText(Path.Combine(_root, "mloop.yaml"), """
            project: order
            models:
              default:
                task: binary-classification
                label: label
                prep:
                  - type: filter-rows
                    column: label
                    operator: "!="
                    value: b
            """);
        var rows = Enumerable.Range(0, 40).Select(i => $"{i % 7},{i % 3},{(i % 2 == 0 ? "a" : "b")}");
        File.WriteAllLines(Path.Combine(_root, "datasets", "train.csv"), ["x,y,label", .. rows]);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "60", "--max-trials", "10");

        Assert.Equal(1, exitCode);
        var output = stdout + stderr;
        Assert.Contains("only one class", output);
        // Refused at the class check, on the 20 rows the step left — not after the classes of the
        // raw file were reported as balanced.
        Assert.True(output.IndexOf("Preprocessing complete", StringComparison.Ordinal)
                    < output.IndexOf("Label column 'label'", StringComparison.Ordinal), output);
        Assert.DoesNotContain("well-balanced", output);
    }

    [Fact]
    public async Task A_model_a_post_train_hook_refuses_is_kept_in_staging()
    {
        File.WriteAllText(Path.Combine(_root, "mloop.yaml"), """
            project: order
            models:
              default:
                task: binary-classification
                label: label
            """);
        var rows = Enumerable.Range(0, 60).Select(i => $"{i % 7},{i % 3},{(i % 2 == 0 ? "a" : "b")}");
        File.WriteAllLines(Path.Combine(_root, "datasets", "train.csv"), ["x,y,label", .. rows]);
        var hooks = Path.Combine(_root, ".mloop", "scripts", "hooks", "post-train");
        Directory.CreateDirectory(hooks);
        File.WriteAllText(Path.Combine(hooks, "refuse.cs"), """
            using System.Threading.Tasks;
            using MLoop.Extensibility.Hooks;

            public class RefuseHook : IMLoopHook
            {
                public string Name => "Refuse";
                public Task<HookResult> ExecuteAsync(HookContext ctx) => Task.FromResult(HookResult.Abort("not this one"));
            }
            """);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "60", "--max-trials", "10");

        var output = stdout + stderr;
        Assert.Equal(0, exitCode);
        Assert.Contains("Kept in staging", output);
        Assert.Contains("Refuse: not this one", output);
        Assert.False(File.Exists(Path.Combine(_root, MLoop.Core.Storage.ExperimentLayout.ModelsDirectory, "default",
            MLoop.Core.Storage.ExperimentLayout.ProductionDirectory, MLoop.Core.Storage.ExperimentLayout.ModelFileName)), output);
    }
}
