using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// `mloop train` says so when one column alone nearly is the label — a column computed from the label or recorded after it scores
/// like the label in the training data and is missing where the model is used, and a copy with a little noise stays under the
/// near-perfect-score warning. It is advice: the column is kept and the run succeeds.
/// </summary>
[Collection("FileSystem")]
public sealed class SingleColumnLabelCopyTrainTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-scopy-" + Guid.NewGuid().ToString("N"));

    public SingleColumnLabelCopyTrainTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Table(bool withCopy)
    {
        var rng = new Random(21);
        var text = new StringBuilder(withCopy ? "size,rooms,class_code,segment\n" : "size,rooms,segment\n");
        for (var i = 0; i < 400; i++)
        {
            var segment = rng.Next(3);
            var size = rng.Next(30, 200) + segment * 20;
            var rooms = rng.Next(1, 6);
            var copy = rng.NextDouble() < 0.04 ? (segment + 1) % 3 : segment;
            var line = withCopy
                ? string.Create(CultureInfo.InvariantCulture, $"{size},{rooms},{copy},s{segment}\n")
                : string.Create(CultureInfo.InvariantCulture, $"{size},{rooms},s{segment}\n");
            text.Append(line);
        }
        return text.ToString();
    }

    [Fact]
    public async Task ANoisyCopyOfTheLabel_IsNamedInTheTrainingOutput_AndTheRunStillSucceeds()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "multiclass-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table(withCopy: true));

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "segment", "--time", "20", "--max-trials", "6");
        var said = train.Stdout + train.Stderr;

        Assert.Equal(0, train.ExitCode);
        Assert.Contains("Column 'class_code' alone names the label for", said, StringComparison.Ordinal);
        Assert.Contains("mloop features select --drop class_code", said, StringComparison.Ordinal);
        Assert.Contains("Training keeps it", said, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TwoCopies_GetOneFullSentence_AndOneLineForTheOther()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "multiclass-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var lines = Table(withCopy: true).Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var text = new StringBuilder("size,rooms,class_code,class_twin,segment\n");
        foreach (var line in lines.Skip(1))
        {
            var f = line.Split(',');
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{f[0]},{f[1]},{f[2]},{f[2]},{f[3]}\n"));
        }
        await File.WriteAllTextAsync("datasets/train.csv", text.ToString());

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "segment", "--time", "20", "--max-trials", "6");
        var said = string.Join(' ', (train.Stdout + train.Stderr).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));   // the console wraps lines

        Assert.Equal(0, train.ExitCode);
        Assert.Single(System.Text.RegularExpressions.Regex.Matches(said, "alone names the label for"));
        Assert.Contains("Also this close to the label:", said, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ATableWithoutACopy_SaysNothingOfTheKind()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "multiclass-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table(withCopy: false));

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "segment", "--time", "20", "--max-trials", "6");

        Assert.Equal(0, train.ExitCode);
        Assert.DoesNotContain("alone names the label", train.Stdout + train.Stderr, StringComparison.Ordinal);
    }
}
