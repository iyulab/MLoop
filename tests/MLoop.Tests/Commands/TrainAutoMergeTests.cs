using System.Globalization;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop train --auto-merge</c> joins the same-schema files of <c>datasets/</c> and trains on the result. The result goes to a file of
/// its own: it was written to <c>train.csv</c> — the project's own training file, which the merge leaves out of its sources — so a project
/// that had a <c>train.csv</c> and two more files lost its <c>train.csv</c>, replaced by the rows of the other two.
/// </summary>
[Collection("FileSystem")]
public class TrainAutoMergeTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TrainAutoMergeTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-auto-merge-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private static void WriteTable(string path, int rows, int seed)
    {
        var random = new Random(seed);
        var lines = new List<string> { "a,b,y" };
        for (var i = 0; i < rows; i++)
        {
            var a = random.NextDouble() * 2 - 1;
            var b = random.NextDouble() * 2 - 1;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{a:F4},{b:F4},{(a + 0.3 * b > 0 ? "yes" : "no")}"));
        }
        File.WriteAllLines(path, lines);
    }

    [Fact]
    public async Task A_projects_own_train_file_is_left_as_it_is_and_the_merge_goes_to_a_file_of_its_own()
    {
        Directory.SetCurrentDirectory(_tempDir);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        var project = Path.Combine(_tempDir, "p");
        Directory.SetCurrentDirectory(project);
        var datasets = Path.Combine(project, "datasets");
        var train = Path.Combine(datasets, "train.csv");
        WriteTable(train, 120, seed: 1);
        WriteTable(Path.Combine(datasets, "more1.csv"), 80, seed: 2);
        WriteTable(Path.Combine(datasets, "more2.csv"), 90, seed: 3);
        var trainBefore = await File.ReadAllBytesAsync(train);

        var result = await CliRunner.RunAsync("train", "--auto-merge", "--label", "y", "--max-trials", "3");

        Assert.True(result.ExitCode == 0, result.Stdout + result.Stderr);
        Assert.Equal(trainBefore, await File.ReadAllBytesAsync(train));   // not replaced
        var merged = Path.Combine(datasets, "merged_train.csv");
        Assert.True(File.Exists(merged));
        Assert.Equal(1 + 80 + 90, File.ReadAllLines(merged).Length);       // header + the two files' rows; train.csv is not a source
        Assert.Contains("datasets/train.csv is not part of the merge", result.Stdout, StringComparison.Ordinal);

        // A second run joins the same two files — the merged file is not counted among them.
        var again = await CliRunner.RunAsync("train", "--auto-merge", "--label", "y", "--max-trials", "3");
        Assert.True(again.ExitCode == 0, again.Stdout + again.Stderr);
        Assert.Equal(1 + 80 + 90, File.ReadAllLines(merged).Length);
    }
}
