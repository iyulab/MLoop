using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// What "anomalous" means to a table anomaly model that is trained and used through the commands: a share of the training rows
/// (<c>contamination</c>, 5 % unless the project says), placed from the training scores themselves — not ML.NET's fixed 0.5 on a
/// relative score, which called a third to two thirds of the rows of public tables anomalous. The threshold is saved beside the
/// model, promoted with it, and read by every path that loads it.
/// </summary>
[Collection("FileSystem")]
public class TabularAnomalyScoringTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TabularAnomalyScoringTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-anomaly-scoring-cli-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    // Three columns that move together, one of noise; the last `anomalies` rows break the shared movement.
    private static void WriteTable(string path, int rows, int anomalies, int seed)
    {
        var random = new Random(seed);
        double Gauss() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        string Cell(double v) => v.ToString("R", CultureInfo.InvariantCulture);
        var lines = new List<string> { "a,b,c,noise" };
        for (var i = 0; i < rows; i++)
        {
            var broken = i >= rows - anomalies;
            var z = Gauss();
            double Column() => broken ? 3 * Gauss() : z + 0.05 * Gauss();
            lines.Add(string.Join(",", Cell(Column()), Cell(Column()), Cell(Column()), Cell(Gauss())));
        }
        File.WriteAllLines(path, lines);
    }

    private async Task<string> ProjectAsync(string name, string? yamlContamination = null)
    {
        var workspace = Path.Combine(_tempDir, name);
        Directory.CreateDirectory(workspace);
        Directory.SetCurrentDirectory(workspace);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "anomaly-detection")).ExitCode);
        var project = Path.Combine(workspace, "p");
        Directory.SetCurrentDirectory(project);
        if (yamlContamination is not null)
        {
            var yaml = Path.Combine(project, "mloop.yaml");
            var text = File.ReadAllText(yaml);
            Assert.Contains("# contamination: 0.05", text, StringComparison.Ordinal);   // init offers the setting
            File.WriteAllText(yaml, text.Replace("# contamination: 0.05", $"contamination: {yamlContamination}", StringComparison.Ordinal));
        }
        WriteTable(Path.Combine(project, "datasets", "train.csv"), 1000, 50, seed: 21);
        WriteTable(Path.Combine(project, "datasets", "later.csv"), 400, 20, seed: 22);
        return project;
    }

    private static async Task<(int Flagged, int Rows, double? Confidence)> FlaggedAsync(string file)
    {
        var predict = await CliRunner.RunAsync("predict", file, "--json");
        Assert.True(predict.ExitCode == 0, predict.Stdout + predict.Stderr);
        using var document = JsonDocument.Parse(predict.Stdout);
        var rows = document.RootElement.GetProperty("predictions").EnumerateArray().ToList();
        var flagged = rows.Count(r => r.GetProperty("isAnomaly").GetBoolean());
        var confidence = rows.Select(r => r.TryGetProperty("confidence", out var c) && c.ValueKind == JsonValueKind.Number ? c.GetDouble() : (double?)null).FirstOrDefault(c => c is not null);
        return (flagged, rows.Count, confidence);
    }

    [Fact]
    public async Task ByDefault_AboutFivePercentOfTheTrainingRowsAreCalledAnomalous_AndTheThresholdIsSavedBesideTheModel()
    {
        var project = await ProjectAsync("default");
        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "10");   // no search: one fit
        Assert.True(train.ExitCode == 0, train.Stdout + train.Stderr);

        var saved = Directory.GetFiles(Path.Combine(project, "models"), ExperimentLayout.AnomalyScoringFileName, SearchOption.AllDirectories);
        Assert.NotEmpty(saved);                                                       // beside the experiment's model, and promoted with it
        Assert.Equal(2, saved.Length);
        var scoring = AnomalyScoring.ReadBeside(Path.Combine(Path.GetDirectoryName(saved[0])!, ExperimentLayout.ModelFileName))!;
        Assert.Equal(0.05, scoring.Contamination);
        Assert.Equal(AnomalyScoring.ReconstructionErrorScore, scoring.Score);

        var (flagged, rows, confidence) = await FlaggedAsync("datasets/train.csv");
        Assert.Equal(1000, rows);
        Assert.InRange(flagged, 40, 60);                                              // 5 % of 1000 — not the 320–610 the fixed 0.5 called
        Assert.NotNull(confidence);

        // Later data of the same kind: the same threshold, so a similar share (the 20 broken rows of 400 are 5 %).
        var later = await FlaggedAsync("datasets/later.csv");
        Assert.InRange(later.Flagged, 10, 40);
    }

    [Fact]
    public async Task TheContaminationInTheYaml_MovesTheThreshold()
    {
        await ProjectAsync("twenty", yamlContamination: "0.2");
        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "10");   // no search: one fit
        Assert.True(train.ExitCode == 0, train.Stdout + train.Stderr);

        var (flagged, rows, _) = await FlaggedAsync("datasets/train.csv");
        Assert.Equal(1000, rows);
        Assert.InRange(flagged, 190, 210);
    }

    [Fact]
    public async Task AContaminationOutsideZeroAndOneHalf_IsRefusedBeforeTraining()
    {
        await ProjectAsync("refused", yamlContamination: "0.7");
        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "10");   // no search: one fit
        Assert.NotEqual(0, train.ExitCode);
        Assert.Contains("contamination", train.Stdout + train.Stderr, StringComparison.Ordinal);
    }
}
