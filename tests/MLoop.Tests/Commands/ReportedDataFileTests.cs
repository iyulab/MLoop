using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// A command that reads a table through a composed copy (text written as jamo is read as the same text as syllables) or a converted
/// one still names the file the user gave: a script reading <c>testDataFile</c> or <c>data</c> from the output must find its own file,
/// not a temporary copy that is gone when the command exits.
/// </summary>
[Collection("FileSystem")]
public sealed class ReportedDataFileTests : IDisposable
{
    private const string Decomposed = "한글"; // the text of "Composed" written as jamo

    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-reported-file-" + Guid.NewGuid().ToString("N"));

    public ReportedDataFileTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Table(int rows, int seed, string word)
    {
        var rng = new Random(seed);
        var text = new StringBuilder("x,note,y\n");
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{word}{i % 5},{(rng.NextDouble() < (x > 0.5 ? 0.9 : 0.1) ? "yes" : "no")}\n"));
        }
        return text.ToString();
    }

    [Fact]
    public async Task EvaluateAndExplain_NameTheFileGiven_WhenItsTextHadToBeComposed()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table(160, seed: 1, word: "한글"));
        var trained = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "30", "--max-trials", "4");
        Assert.True(trained.ExitCode == 0, trained.Stdout + trained.Stderr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);
        await File.WriteAllTextAsync("datasets/held-out.csv", Table(60, seed: 2, word: Decomposed));

        var evaluated = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/held-out.csv", "--json");
        Assert.True(evaluated.ExitCode == 0, evaluated.Stdout + evaluated.Stderr);
        using (var json = JsonDocument.Parse(evaluated.Stdout))
            Assert.Equal("held-out.csv", Path.GetFileName(json.RootElement.GetProperty("testDataFile").GetString()));

        var explained = await CliRunner.RunAsync("explain", "exp-001", "--data", "datasets/held-out.csv", "--repeats", "2", "--rows", "40", "--seed", "1", "--json");
        Assert.True(explained.ExitCode == 0, explained.Stdout + explained.Stderr);
        using (var json = JsonDocument.Parse(explained.Stdout))
            Assert.Equal("held-out.csv", json.RootElement.GetProperty("data").GetString());

        var narrated = await CliRunner.RunAsync("explain", "exp-001", "--data", "datasets/held-out.csv", "--repeats", "2", "--rows", "40", "--seed", "1");
        Assert.True(narrated.ExitCode == 0, narrated.Stdout + narrated.Stderr);
        Assert.DoesNotContain("mloop_composed", narrated.Stdout + narrated.Stderr, StringComparison.Ordinal);
    }
}
