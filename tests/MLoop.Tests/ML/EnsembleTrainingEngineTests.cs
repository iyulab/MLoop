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
}
