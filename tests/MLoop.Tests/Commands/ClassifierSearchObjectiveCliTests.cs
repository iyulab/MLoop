using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// What a classifier project's search optimizes, observed on the command line a user types: <c>init</c> writes
/// <c>metric: auto</c>, and <c>train</c> then ranks its trials by AUC (binary) or log-loss (multiclass) while the
/// experiment still records the task's primary metric, which is what promotion reads.
/// </summary>
/// <remarks>
/// A unit test that hands the trainer <c>auto</c> proves nothing about this: the first version of the change
/// passed every one of those and optimized accuracy anyway, because the configuration merger had already replaced
/// <c>auto</c> with the primary metric before the trainer saw it. Only a run through the dispatch sees what survives.
/// </remarks>
public class ClassifierSearchObjectiveCliTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-objective-" + Guid.NewGuid().ToString("N"));

    public ClassifierSearchObjectiveCliTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { /* nothing left to do */ } }

        if (Directory.Exists(_workspace))
        {
            try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }
    }

    [Theory]
    [InlineData("binary-classification", 2, "auc", "accuracy")]
    [InlineData("multiclass-classification", 3, "log_loss", "macro_accuracy")]
    public async Task AClassifiersSearchRanksItsTrialsByASmoothScore_AndTheExperimentRecordsThePrimaryMetric(
        string task, int classes, string ranked, string primary)
    {
        var project = "p-" + task;
        var (initExit, initOut, _) = await CliRunner.RunAsync("init", project, "--task", task);
        Assert.True(initExit == 0, initOut);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, project));
        File.WriteAllText("train.csv", Rows(classes));

        var (exit, stdout, _) = await CliRunner.RunAsync(
            "train", "train.csv", "--label", "y", "--time", "60", "--max-trials", "10", "--no-promote", "--json");

        Assert.True(exit == 0, stdout);
        var events = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.TrimStart())
            .Where(line => line.StartsWith('{'))
            .Select(line => JsonDocument.Parse(line).RootElement)
            .ToList();
        var trial = events.First(e => e.GetProperty("event").GetString() == "trial");
        Assert.Equal(ranked, trial.GetProperty("metric").GetString());

        var experiment = Directory.GetFiles(Path.Combine("models"), "config.json", SearchOption.AllDirectories).Single();
        using var config = JsonDocument.Parse(File.ReadAllText(experiment));
        Assert.Equal(primary, config.RootElement.GetProperty("metric").GetString());
    }

    // Enough rows, and classes sitting far enough apart, for a ten-second search to finish a trial.
    private static string Rows(int classes)
    {
        var random = new Random(5);
        var lines = new List<string> { "a,b,y" };
        for (var i = 0; i < 300; i++)
        {
            var label = i % classes;
            lines.Add(FormattableString.Invariant($"{label + random.NextDouble() * 0.8:F4},{random.NextDouble():F4},c{label}"));
        }

        return string.Join('\n', lines);
    }
}
