using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Training;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Label-less clustering's structured predict surfaces (serve <c>/predict</c>, <c>mloop predict --json</c>)
/// and its file predict read the vector the model was fit on. The saved model is the trainer alone over
/// <c>Features</c>; every prediction path builds that vector itself — serve from the named columns
/// (<see cref="PredictionService"/>'s <c>EnsureFeaturesColumn</c>), the file path from the loader's range —
/// so a Concatenate inside the model would count a column twice ("expected Vector&lt;Single, 3&gt;, got
/// Vector&lt;Single, 4&gt;"). Training loads a label-less file without a label, so its vector is the same.
/// The whole train → predict → serve seam, through the commands too, is
/// <see cref="MLoop.Tests.ML.LabellessVectorTaskSeamTests"/>.
/// </summary>
public class ClusteringFeaturesContractTests : IDisposable
{
    private readonly List<string> _tempDirs = new();

    public void Dispose()
    {
        foreach (var d in _tempDirs)
        {
            try { if (Directory.Exists(d)) Directory.Delete(d, recursive: true); } catch { }
        }
    }

    private static string WriteLabellessCsv(string dir, string name, int rows)
    {
        var path = Path.Combine(dir, name);
        var rng = new Random(11);
        using var writer = new StreamWriter(path);
        writer.WriteLine("pH,Temp,Current");
        for (int i = 0; i < rows; i++)
        {
            // Two loose blobs so KMeans has something real to separate, with pH first —
            // the column a label-less load once set apart from the vector.
            var center = i % 2 == 0 ? 5.0 : 9.0;
            var pH = center + rng.NextDouble() * 0.5;
            var temp = center * 4 + rng.NextDouble() * 0.5;
            var current = center * 0.7 + rng.NextDouble() * 0.5;
            writer.WriteLine($"{pH:F3},{temp:F3},{current:F3}");
        }
        return path;
    }

    [Fact]
    public async Task LabellessClustering_StructuredPredict_DoesNotThrowDimensionMismatch()
    {
        var ctx = new MLContext(seed: 42);
        var dir = Path.Combine(Path.GetTempPath(), "mloop-d24-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);

        var trainCsv = WriteLabellessCsv(dir, "train.csv", 60);

        var loader = new CsvDataLoader(ctx);
        var runner = new TrainingRunner(ctx, loader);
        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = trainCsv,
            LabelColumn = "",       // label-less
            Task = "clustering",
            TimeLimitSeconds = 30,
            NumClusters = 2,
        };

        var result = await runner.RunAsync(config, cancellationToken: CancellationToken.None);
        Assert.NotNull(result.Schema);

        var modelPath = Path.Combine(dir, "model.zip");
        ctx.Model.Save(result.Model, null, modelPath);

        // --- Path A (the primary repro): PredictionService row-based predict ---
        var rows = new[]
        {
            new Dictionary<string, object> { ["pH"] = 5.1f, ["Temp"] = 20.2f, ["Current"] = 3.6f },
            new Dictionary<string, object> { ["pH"] = 9.1f, ["Temp"] = 36.2f, ["Current"] = 6.4f },
        };
        var service = new PredictionService(ctx);

        // Before the fix this threw InvalidOperationException("Feature vector dimension mismatch...").
        var serviceResult = service.Predict(rows, result.Schema!, modelPath, "clustering");

        Assert.Equal(2, serviceResult.Rows.Count);
        Assert.All(serviceResult.Rows, r => Assert.NotNull(r.ClusterId));

        // --- Path B (regression guard for the CLI-side fix): PredictionEngine CSV predict ---
        var predictCsv = WriteLabellessCsv(dir, "predict.csv", 6);
        var outputCsv = Path.Combine(dir, "output.csv");
        var engine = new PredictionEngine();

        var count = await engine.PredictAsync(
            modelPath, predictCsv, outputCsv, result.Schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: null,
            preserveColumns: null,
            interval: null,
            taskType: "clustering");

        Assert.Equal(6, count);
        var outputLines = File.ReadAllLines(outputCsv);
        Assert.True(outputLines.Length > 1, "CSV predict must emit a header plus data rows.");
        Assert.Contains("PredictedLabel", outputLines[0]);
    }
}
