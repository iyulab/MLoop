using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Runtime;

namespace MLoop.Core.DeepLearning.Tests.Evaluation;

/// <summary>
/// Validates <see cref="ObjectDetectionEvaluator"/> against a real (tiny) AutoFormerV2 model.
/// Self-contained — it generates a handful of BMP images and YOLO labels — but requires the
/// libtorch runtime, so it skips when that runtime is not installed (e.g. on CI). When it does run
/// it trains a 1-epoch model purely to produce the genuine scored schema, then confirms the
/// evaluator resolves the (hidden, key-typed) prediction columns and returns mAP metrics in range.
/// </summary>
public sealed class ObjectDetectionEvaluatorTests : IDisposable
{
    private readonly string _tempDir;

    public ObjectDetectionEvaluatorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop_od_eval_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch (IOException) { }
    }

    private static bool TorchRuntimeInstalled()
    {
        var runtime = RuntimeRegistry.GetRequiredByTask("object-detection");
        return runtime != null && new RuntimeManager().IsInstalled(runtime);
    }

    [Fact]
    public void Evaluate_OnRealScoredModel_ReturnsMapMetricsInRange()
    {
        if (!TorchRuntimeInstalled())
            return; // libtorch not installed (CI); integration coverage runs only where the runtime exists.

        RuntimeManager.EnsureRuntimeForTask("object-detection");

        var (imagesDir, labelsDir) = (Path.Combine(_tempDir, "images"), Path.Combine(_tempDir, "labels"));
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(labelsDir);
        File.WriteAllText(Path.Combine(_tempDir, "classes.txt"), "defect\n");

        // A few same-shaped images with one centered box each — enough to fit a 1-epoch model and
        // produce the real scored schema. The model need not be accurate; we validate the wiring.
        for (int i = 0; i < 6; i++)
        {
            var stem = $"img{i:00}";
            CreateBmp(Path.Combine(imagesDir, stem + ".bmp"), 64, 64);
            File.WriteAllText(Path.Combine(labelsDir, stem + ".txt"), "0 0.5 0.5 0.4 0.4\n");
        }

        var ml = new MLContext(seed: 42);
        var loader = new YoloDataLoader(ml);
        var data = loader.LoadData(_tempDir, labelColumn: "Label", taskType: "object-detection");
        var (train, test) = loader.SplitData(data, testFraction: 0.34);

        // Mirror AutoMLRunner.RunObjectDetectionAsync so the scored schema matches production.
        var pipeline = ml.Transforms.LoadImages(
                outputColumnName: "Image", imageFolder: string.Empty, inputColumnName: CocoDataLoader.ImagePathColumn)
            .Append(ml.Transforms.Conversion.MapValueToKey(
                outputColumnName: "LabelKey", inputColumnName: "Label"))
            .Append(ml.MulticlassClassification.Trainers.ObjectDetection(
                labelColumnName: "LabelKey",
                boundingBoxColumnName: CocoDataLoader.BoundingBoxColumn,
                imageColumnName: "Image",
                maxEpoch: 1))
            .Append(ml.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        var model = pipeline.Fit(train);
        var scored = model.Transform(test);

        var metrics = ObjectDetectionEvaluator.Evaluate(ml, scored);

        Assert.True(metrics.ContainsKey("map_50"), "map_50 metric should be present.");
        Assert.True(metrics.ContainsKey("map_50_95"), "map_50_95 metric should be present.");
        Assert.InRange(metrics["map_50"], 0.0, 1.0);
        Assert.InRange(metrics["map_50_95"], 0.0, 1.0);
    }

    /// <summary>Writes a minimal valid 24-bit uncompressed BMP of the given dimensions.</summary>
    private static void CreateBmp(string path, int width, int height)
    {
        int rowSize = ((24 * width + 31) / 32) * 4;
        int pixelArraySize = rowSize * height;
        int fileSize = 54 + pixelArraySize;

        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        bw.Write((byte)'B'); bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write(0);
        bw.Write(54);
        bw.Write(40);
        bw.Write(width);
        bw.Write(height);
        bw.Write((short)1);
        bw.Write((short)24);
        bw.Write(0);
        bw.Write(pixelArraySize);
        bw.Write(2835);
        bw.Write(2835);
        bw.Write(0);
        bw.Write(0);
        var row = new byte[rowSize];
        Array.Fill(row, (byte)0xFF);
        for (var y = 0; y < height; y++) bw.Write(row);
    }
}
