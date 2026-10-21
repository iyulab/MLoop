namespace MLoop.Tests.Commands;

/// <summary>
/// The label checks of <c>mloop train</c> see the rows training gets: they run after the
/// <c>prep:</c> steps of <c>mloop.yaml</c>, which can drop, add or reshape rows.
/// </summary>
public sealed class TrainPrepOrderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-prep-order-{Guid.NewGuid():N}");
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TrainPrepOrderTests()
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

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "5");

        Assert.Equal(1, exitCode);
        var output = stdout + stderr;
        Assert.Contains("only one class", output);
        // Refused at the class check, on the 20 rows the step left — not after the classes of the
        // raw file were reported as balanced.
        Assert.True(output.IndexOf("Preprocessing complete", StringComparison.Ordinal)
                    < output.IndexOf("Label column 'label'", StringComparison.Ordinal), output);
        Assert.DoesNotContain("well-balanced", output);
    }
}
