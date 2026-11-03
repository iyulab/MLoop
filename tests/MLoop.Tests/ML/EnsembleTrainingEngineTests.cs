using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A real training run asked for a blend (<c>Ensemble</c>) saves one model, records the blend in the experiment — members, weights,
/// out-of-fold scores — and the serving path reads that saved model and answers.
/// </summary>
[Collection("FileSystem")]
public class EnsembleTrainingEngineTests : IDisposable
{
    private readonly string _tempDir;

    public EnsembleTrainingEngineTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-ensemble-engine-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private async Task<string> NoisyCsvAsync()
    {
        var csv = Path.Combine(_tempDir, "noisy.csv");
        var random = new Random(11);
        var lines = new List<string> { "x1,x2,x3,x4,hit" };
        for (var i = 0; i < 900; i++)
        {
            double X() => random.NextDouble() * 2 - 1;
            var (x1, x2, x3, x4) = (X(), X(), X(), X());
            var logit = 1.6 * x1 - 1.2 * x2 + 1.5 * x3 * x4 + 0.8 * Math.Sin(3 * x1) + (random.NextDouble() - 0.5) * 2.5;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{x3:F4},{x4:F4},{(logit > 0 ? "yes" : "no")}"));
        }
        await File.WriteAllLinesAsync(csv, lines);
        return csv;
    }

    /// <summary>A table with two numbers, a category (a few colors) — none of which tells the classes apart — and a free-text column that alone carries the signal.</summary>
    private async Task<string> TextCsvAsync()
    {
        var csv = Path.Combine(_tempDir, "reviews.csv");
        var random = new Random(7);
        string[] colors = ["red", "green", "blue", "amber"], good = ["great", "love", "excellent", "fast", "works", "happy", "recommend"],
            bad = ["broken", "slow", "refund", "bad", "late", "awful", "waste"], filler = ["the", "a", "it", "this", "was", "is", "very", "product", "item", "really"];
        var lines = new List<string> { "x1,color,review,x2,label" };
        for (var i = 0; i < 900; i++)
        {
            var positive = random.NextDouble() < 0.5;
            var x1 = (random.NextDouble() - 0.5) * 3;
            var color = colors[random.Next(colors.Length)];
            var words = Enumerable.Range(0, 5).Select(_ => filler[random.Next(filler.Length)])
                .Concat(Enumerable.Range(0, 2).Select(_ => (random.NextDouble() < 0.8 ? positive : !positive) ? good[random.Next(good.Length)] : bad[random.Next(bad.Length)]))
                .OrderBy(_ => random.Next());
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x1:F3},{color},{string.Join(' ', words)},{random.NextDouble() - 0.5:F3},{(positive ? "yes" : "no")}"));
        }
        await File.WriteAllLinesAsync(csv, lines);
        return csv;
    }

    [Fact]
    public async Task ABlend_IsRecordedInTheExperiment_AndTheSavedModelServesAnswers()
    {
        var csv = await NoisyCsvAsync();
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
        var originalOut = Console.Out;
        Console.SetOut(new StringWriter());
        TrainingResult result;
        try
        {
            result = await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = "blend",
                DataFile = csv,
                LabelColumn = "hit",
                Task = "binary-classification",
                Metric = "auc",
                TimeLimitSeconds = 120,
                MaxTrials = 16,
                Ensemble = true,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        var experimentPath = store.GetExperimentPath("blend", result.ExperimentId);
        using var metadata = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(experimentPath, ExperimentLayout.MetadataFileName)));
        var recorded = metadata.RootElement.GetProperty("result").GetProperty("ensemble");
        Assert.True(recorded.GetProperty("members").GetArrayLength() >= 2);
        Assert.True(recorded.GetProperty("outOfFoldScore").GetDouble() > recorded.GetProperty("bestSingleOutOfFoldScore").GetDouble());
        Assert.False(recorded.TryGetProperty("skippedBecause", out _));
        Assert.Equal("Ensemble", metadata.RootElement.GetProperty("result").GetProperty("trainer").GetProperty("name").GetString());

        // Read back from disk, the record keeps the blend.
        var loaded = await store.LoadAsync("blend", result.ExperimentId);
        Assert.True(loaded.Result!.Ensemble!.IsBlend);
        Assert.Equal(1.0, loaded.Result.Ensemble.Members!.Sum(m => m.Weight), 6);

        // The serving path: the saved file, the calibration beside it, the schema the run recorded.
        var rows = PredictionEngine.LoadCsvAsRows(csv, result.Schema!, "hit");
        var answered = new PredictionEngine().PredictWithService(
            rows, result.Schema!, Path.Combine(experimentPath, ExperimentLayout.ModelFileName), "binary-classification", "hit");
        Assert.Equal(rows.Count(), answered.Rows.Count());
        Assert.All(answered.Rows, r => Assert.NotNull(r));
    }

    [Fact]
    public async Task ABlendOverACategoryAndATextColumn_TrainsAndTheSavedModelAnswersEveryRow()
    {
        // The table's string columns are read by featurizers the blend's members share; a blend that dropped or re-ran them would fail here, or answer from the numbers alone.
        var csv = await TextCsvAsync();
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
        var originalOut = Console.Out;
        Console.SetOut(new StringWriter());
        TrainingResult result;
        try
        {
            result = await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = "reviews",
                DataFile = csv,
                LabelColumn = "label",
                Task = "binary-classification",
                Metric = "auc",
                TimeLimitSeconds = 120,
                MaxTrials = 16,
                Ensemble = true,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        var loaded = await store.LoadAsync("reviews", result.ExperimentId);
        // Whether a blend beats the best single model out of fold is the data's answer, and on these few hundred rows it is close (three folds
        // on a machine with LightGBM keep the single model): the run must say which it kept and why, and the saved model must serve either way.
        // The blend over text and category columns itself is held by EnsembleChainTests.
        var ensemble = loaded.Result!.Ensemble!;
        Assert.True(ensemble.IsBlend || !string.IsNullOrEmpty(ensemble.SkippedBecause), "neither a blend nor a reason for keeping the single model was recorded");

        var experimentPath = store.GetExperimentPath("reviews", result.ExperimentId);
        var rows = PredictionEngine.LoadCsvAsRows(csv, result.Schema!, "label");
        var answered = new PredictionEngine().PredictWithService(
            rows, result.Schema!, Path.Combine(experimentPath, ExperimentLayout.ModelFileName), "binary-classification", "label");
        Assert.Equal(rows.Count(), answered.Rows.Count());
        Assert.All(answered.Rows, r => Assert.NotNull(r));

        // The words carry the signal, so a model that read the text separates the classes far better than chance.
        Assert.True(result.Metrics["auc"] > 0.8, "held-out auc " + result.Metrics["auc"]);
    }
}
