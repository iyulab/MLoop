using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

/// <summary>
/// A table's anomaly model must score by the shape of the rows, not by the units of the columns: a column in the billions
/// next to one in the thousandths stopped training ("The learnt eigenvectors contained NaN values"), and where it trained,
/// the one widest column was the whole model. The table here has three columns that move together (in three different
/// units) and one of noise; the rows that break the shared movement are the anomalies, and nothing about the answer may
/// depend on which unit a column is written in.
/// </summary>
[Collection("FileSystem")]
public class TabularAnomalyScalingTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public TabularAnomalyScalingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-anomaly-scaling-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    // The last `anomalies` rows break the shared movement of the first three columns.
    private static void WriteTable(string path, int rows, int anomalies, int seed, double[] unit)
    {
        var random = new Random(seed);
        double Gauss() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        string Cell(double v) => v.ToString("R", CultureInfo.InvariantCulture);

        var lines = new List<string> { "big,small,mid,noise" };
        for (var i = 0; i < rows; i++)
        {
            var broken = i >= rows - anomalies;
            var z = Gauss();
            double Column() => broken ? Gauss() : z + 0.05 * Gauss();
            lines.Add(string.Join(",", Cell(Column() * unit[0]), Cell(Column() * unit[1]), Cell(Column() * unit[2]), Cell(Gauss())));
        }
        File.WriteAllLines(path, lines);
    }

    // The area under the ROC curve of the scores against the known anomalies (the last `anomalies` rows) — the chance an
    // anomaly scores above a normal row.
    private static double Auc(IReadOnlyList<double?> scores, int anomalies)
    {
        var anomalous = scores.Skip(scores.Count - anomalies).Select(s => s!.Value).ToList();
        var normal = scores.Take(scores.Count - anomalies).Select(s => s!.Value).ToList();
        double wins = 0;
        foreach (var a in anomalous)
            foreach (var n in normal)
                wins += a > n ? 1 : a == n ? 0.5 : 0;
        return wins / ((double)anomalous.Count * normal.Count);
    }

    [Theory]
    [InlineData(1e9, 1e-3, 1.0)]   // the units a table written by different systems really has
    [InlineData(1.0, 1.0, 1.0)]
    public async Task The_anomalies_score_above_the_normal_rows_whatever_the_unit_of_each_column(double big, double small, double mid)
    {
        var workspace = Path.Combine(_tempDir, "cli");
        Directory.CreateDirectory(workspace);
        Directory.SetCurrentDirectory(workspace);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "anomaly-detection")).ExitCode);
        var project = Path.Combine(workspace, "p");
        Directory.SetCurrentDirectory(project);
        WriteTable(Path.Combine(project, "datasets", "train.csv"), 1000, 50, seed: 11, [big, small, mid]);
        WriteTable(Path.Combine(project, "datasets", "p.csv"), 400, 20, seed: 12, [big, small, mid]);

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "10");   // no search: one fit
        Assert.True(train.ExitCode == 0, train.Stdout + train.Stderr);

        var predict = await CliRunner.RunAsync("predict", "datasets/p.csv", "--json");
        Assert.True(predict.ExitCode == 0, predict.Stdout + predict.Stderr);
        using var document = JsonDocument.Parse(predict.Stdout);
        var scores = document.RootElement.GetProperty("predictions").EnumerateArray()
            .Select(p => p.TryGetProperty("anomalyScore", out var s) && s.ValueKind == JsonValueKind.Number ? s.GetDouble() : (double?)null)
            .ToList();

        Assert.Equal(400, scores.Count);
        Assert.DoesNotContain(null, scores);
        var auc = Auc(scores, anomalies: 20);
        Assert.True(auc > 0.95, $"area under the curve {auc:F3} — the model must rank the rows that break the shared movement above the others");
    }

    [Fact]
    public async Task A_table_whose_columns_share_nothing_says_the_kept_components_carry_little_and_a_shared_one_does_not()
    {
        var workspace = Path.Combine(_tempDir, "share");
        Directory.CreateDirectory(workspace);
        Directory.SetCurrentDirectory(workspace);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "anomaly-detection")).ExitCode);
        var project = Path.Combine(workspace, "p");
        Directory.SetCurrentDirectory(project);

        // Five independent columns: the four components the ceiling allows carry about four fifths, never the ninety percent aimed at.
        var random = new Random(31);
        double Gauss() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        var lines = new List<string> { "a,b,c,d,e" };
        for (var i = 0; i < 600; i++)
            lines.Add(string.Join(",", Enumerable.Range(0, 5).Select(_ => Gauss().ToString("R", CultureInfo.InvariantCulture))));
        var independent = Path.Combine(project, "datasets", "independent.csv");
        File.WriteAllLines(independent, lines);
        var shared = Path.Combine(project, "datasets", "shared.csv");
        WriteTable(shared, 600, 0, seed: 32, [1.0, 1.0, 1.0]);

        var flat = await CliRunner.RunAsync("train", "datasets/independent.csv", "--time", "10");   // no search: one fit
        Assert.True(flat.ExitCode == 0, flat.Stdout + flat.Stderr);
        Assert.Contains("carry only", flat.Stdout + flat.Stderr, StringComparison.Ordinal);

        var together = await CliRunner.RunAsync("train", "datasets/shared.csv", "--time", "10");   // no search: one fit
        Assert.True(together.ExitCode == 0, together.Stdout + together.Stderr);
        Assert.DoesNotContain("carry only", together.Stdout + together.Stderr, StringComparison.Ordinal);
    }
}
