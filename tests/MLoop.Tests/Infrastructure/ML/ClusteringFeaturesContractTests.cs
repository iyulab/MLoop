using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// D24 (cycle-154, honeai-sim clustering live dogfooding): label-less clustering's structured predict
/// surfaces (serve <c>/predict</c>, <c>mloop predict --json</c>) hard-failed with a Features dimension
/// mismatch ("expected Vector&lt;Single, 3&gt;, got Vector&lt;Single, 4&gt;"), while the CLI CSV path
/// worked by incidental shape-matching.
///
/// <para>Root cause: <see cref="CsvDataLoader"/> picks the first CSV column as a "dummy label" so
/// <c>InferColumns</c> can run on label-less data, then excludes it from the merged "Features" vector —
/// so the trained schema is <c>[dummyLabel(Single), Features(Vector&lt;2&gt;)]</c>. The old
/// <c>RunClusteringAsync</c> embedded a <c>Concatenate("Features", [dummyLabel, "Features"])</c> INSIDE
/// the fitted+saved pipeline, baking in that incidental 2-input shape. At predict time,
/// <see cref="PredictionService"/>'s <c>EnsureFeaturesColumn</c> (D8) independently builds its OWN
/// "Features" from every named non-excluded column (dummyLabel included, since predict has no concept
/// of "dummy") — so the model's embedded Concatenate then re-consumed that already-3-dim "Features" as
/// one of ITS two inputs, alongside the still-present dummyLabel scalar, producing 1+3=4 instead of the
/// 3 KMeans was fit on.</para>
///
/// <para>Fix (D24-A, train-side contract normalization): <c>RunClusteringAsync</c> now pre-featurizes
/// (materializes "Features" from every real feature column, matching what EnsureFeaturesColumn will
/// build) and fits ONLY the trainer on the result — the saved model embeds no Concatenate at all, so
/// there is nothing left to re-collide with EnsureFeaturesColumn's output. The CLI CSV path
/// (<see cref="PredictionEngine"/>) needed a matching fix: since it now bypasses the old embedded
/// concat too, it must explicitly restore the dummy-label dimension InferColumns split off, or its own
/// "Features" would carry one fewer dimension than the new bare-trainer model expects.</para>
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
            // Two loose blobs so KMeans has something real to separate, mirroring the KAMP
            // sensor shape the issue was found on (pH first column = the dummy-label victim).
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
        var runner = new AutoMLRunner(ctx, loader, dir);
        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = trainCsv,
            LabelColumn = "",       // label-less — CsvDataLoader picks "pH" as the InferColumns dummy label
            Task = "clustering",
            TimeLimitSeconds = 30,
            NumClusters = 2,
        };

        var result = await runner.RunAsync(config, cancellationToken: CancellationToken.None);
        Assert.NotNull(result.Schema);

        var modelPath = Path.Combine(dir, "model.zip");
        ctx.Model.Save(result.Model, null, modelPath);

        // --- Path A (D24's primary repro): PredictionService row-based predict ---
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
