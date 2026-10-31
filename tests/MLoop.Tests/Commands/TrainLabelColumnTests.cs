using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// A label that is not a column of the file is the whole of what is wrong: it is said as that, with the columns the file has, and not as
/// "only one class" - the sentence the next step happened to fail with when it found no label to read.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainLabelColumnTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-labelcol-" + Guid.NewGuid().ToString("N"));

    public TrainLabelColumnTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Theory]
    [InlineData("nosuch", "binary-classification")]
    [InlineData("Y", "binary-classification")]   // names are case-sensitive; the message says the one that differs only in case
    [InlineData("nosuch", "regression")]
    public async Task ALabelThatIsNoColumn_IsNamedWithTheColumnsTheFileHas(string label, string task)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", task)).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var rng = new Random(1);
        var text = new StringBuilder("x1,x2,y\n");
        for (var i = 0; i < 80; i++)
        {
            var x = rng.NextDouble();
            var y = task == "regression" ? string.Create(CultureInfo.InvariantCulture, $"{x * 3:F3}") : x > 0.5 ? "yes" : "no";
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{y}\n"));
        }
        await File.WriteAllTextAsync("datasets/train.csv", text.ToString());

        var run = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", label, "--max-trials", "2");

        var output = string.Join(' ', (run.Stdout + run.Stderr).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        Assert.NotEqual(0, run.ExitCode);
        Assert.Contains($"Label column '{label}' not found", output, StringComparison.Ordinal);
        Assert.Contains("Available columns: x1, x2, y", output, StringComparison.Ordinal);
        Assert.DoesNotContain("only one class", output, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")));
    }
}
