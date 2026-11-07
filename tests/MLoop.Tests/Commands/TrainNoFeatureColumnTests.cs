using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace MLoop.Tests.Commands;

/// <summary>
/// A table whose only column besides the label is dropped by an exclusion rule (here: a text column whose every value is a
/// JSON record, read as a structured value) has nothing to train on. The run says so before the first trial — naming the
/// column, the rule and the way to keep it — instead of running every trainer into the same failure; and a column whose role
/// is declared under <c>columns:</c> is never excluded, so the same file trains.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainNoFeatureColumnTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-nofeature-" + Guid.NewGuid().ToString("N"));

    public TrainNoFeatureColumnTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    /// <summary>Two columns: a JSON-lines log line and a 0/1 label that the line's level decides.</summary>
    private static string JsonLinesTable()
    {
        var csv = new StringBuilder("Content,Label\n");
        var rng = new Random(11);
        for (var i = 0; i < 300; i++)
        {
            var error = rng.NextDouble() < 0.3;
            var json = string.Create(CultureInfo.InvariantCulture,
                $"{{\"\"n\"\":{i},\"\"level\"\":\"\"{(error ? "ERROR" : "INFO")}\"\",\"\"msg\"\":\"\"{(error ? "upstream timeout" : "request served")} {rng.Next(1000)}\"\"}}");
            csv.Append(CultureInfo.InvariantCulture, $"\"{json}\",{(error ? 1 : 0)}\n");
        }
        return csv.ToString();
    }

    /// <summary>
    /// The model's <c>label:</c> line followed by a <c>columns:</c> block declaring <paramref name="column"/> as <paramref name="type"/>
    /// — in the file's own line ending (a checkout on Windows writes CRLF).
    /// </summary>
    private static string Declaring(string yaml, string column, string type)
    {
        var declared = Regex.Replace(yaml, @"^(    label: Label)(\r?\n)",
            m => m.Groups[1].Value + m.Groups[2].Value
                + string.Join(m.Groups[2].Value, "    columns:", $"      {column}:", $"        type: {type}") + m.Groups[2].Value,
            RegexOptions.Multiline);
        Assert.NotEqual(yaml, declared);
        return declared;
    }

    private async Task<string> ProjectWithTheTableAsync()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", JsonLinesTable());
        return "datasets/train.csv";
    }

    [Fact]
    public async Task EveryColumnExcluded_IsRefusedBeforeTheFirstTrial_NamingTheColumnTheRuleAndTheWayOut()
    {
        var data = await ProjectWithTheTableAsync();

        var (exit, output, error) = await CliRunner.RunAsync("train", data, "--label", "Label", "--max-trials", "3", "--time", "30", "--no-promote");

        Assert.NotEqual(0, exit);
        var said = output + error;
        Assert.Contains("No column is left to train on", said);
        Assert.Contains("Content", said);
        Assert.Contains("Structured", said);
        Assert.Contains("`columns:`", said);
        // refused before the search: no trial was run, so no trial's own failure is quoted
        Assert.DoesNotContain("every trainer failed", said);
        Assert.All(Directory.GetFiles("models", "trials.ndjson", SearchOption.AllDirectories),
            f => Assert.Equal(0, new FileInfo(f).Length));
    }

    [Fact]
    public async Task ADeclaredTextRole_KeepsTheColumn_AndTheSameFileTrains()
    {
        var data = await ProjectWithTheTableAsync();
        await File.WriteAllTextAsync("mloop.yaml", Declaring(await File.ReadAllTextAsync("mloop.yaml"), "Content", "text"));

        var (exit, output, error) = await CliRunner.RunAsync(
            "train", data, "--label", "Label", "--max-trials", "2", "--time", "30", "--no-promote");

        Assert.True(exit == 0, output + error);
        Assert.DoesNotContain("No column is left to train on", output + error);
    }

    [Fact]
    public async Task EveryColumnDeclaredIgnore_IsRefusedBeforeTheFirstTrial_SayingTheIgnoreIsTheCause()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var csv = new StringBuilder("a,Label\n");
        for (var i = 0; i < 100; i++)
            csv.Append(CultureInfo.InvariantCulture, $"{i / 100.0:F2},{(i > 60 ? 1 : 0)}\n");
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        await File.WriteAllTextAsync("mloop.yaml", Declaring(await File.ReadAllTextAsync("mloop.yaml"), "a", "ignore"));

        var (exit, output, error) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "Label", "--max-trials", "3", "--time", "30", "--no-promote");

        Assert.NotEqual(0, exit);
        var said = output + error;
        Assert.Contains("No column is left to train on", said);
        Assert.Contains("declared `type: ignore` — a", said);
        Assert.Contains("remove its `type: ignore`", said);
        Assert.DoesNotContain("every trainer failed", said);
    }

    [Fact]
    public async Task ThePreflight_SaysWhatTrainingWouldRefuse_AndSaysNothingOnceTheRoleIsDeclared()
    {
        var data = await ProjectWithTheTableAsync();

        var (exit, output, error) = await CliRunner.RunAsync(
            "train", data, "--label", "Label", "--max-trials", "3", "--analyze-data");

        Assert.Equal(0, exit);
        Assert.Contains("CRITICAL", output + error);
        Assert.Contains("No column is left to train on", output + error);

        await File.WriteAllTextAsync("mloop.yaml", Declaring(await File.ReadAllTextAsync("mloop.yaml"), "Content", "text"));

        var (_, declaredOut, declaredErr) = await CliRunner.RunAsync(
            "train", data, "--label", "Label", "--max-trials", "3", "--analyze-data");

        Assert.DoesNotContain("No column is left to train on", declaredOut + declaredErr);
    }
}
