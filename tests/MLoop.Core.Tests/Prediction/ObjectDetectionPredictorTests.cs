using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Prediction;

/// <summary>
/// Covers <see cref="ObjectDetectionPredictor"/> two ways: a synthetic scored data view exercises
/// the detection-extraction logic deterministically with no native dependency, and a real (tiny)
/// AutoFormerV2 model confirms the extractor resolves the genuine scored schema. The integration
/// test skips when the libtorch runtime is absent (e.g. on CI), mirroring
/// <c>ObjectDetectionEvaluatorTests</c>.
/// </summary>
public sealed class ObjectDetectionPredictorTests : IDisposable
{
    private readonly string _tempDir;

    public ObjectDetectionPredictorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop_od_pred_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch (IOException) { }
    }

    /// <summary>A scored OD row shape: parallel label/score arrays and a flat 4-per-box float vector.</summary>
    private sealed class ScoredRow
    {
        public string ImagePath { get; set; } = "";
        [VectorType] public string[] PredictedLabel { get; set; } = [];
        [VectorType] public float[] PredictedBoundingBoxes { get; set; } = [];
        [VectorType] public float[] Score { get; set; } = [];
    }

    [Fact]
    public void Predict_ExtractsDetections_FromScoredSchema()
    {
        var ml = new MLContext(seed: 42);
        var rows = new[]
        {
            new ScoredRow
            {
                ImagePath = "a.jpg",
                PredictedLabel = ["defect", "scratch"],
                PredictedBoundingBoxes = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f],
                Score = [0.9f, 0.5f]
            },
            new ScoredRow
            {
                ImagePath = "b.jpg",
                PredictedLabel = [],
                PredictedBoundingBoxes = [],
                Score = []
            }
        };

        var scored = ml.Data.LoadFromEnumerable(rows);

        var results = ObjectDetectionPredictor.Predict(ml, scored);

        Assert.Equal(2, results.Count);

        Assert.Equal("a.jpg", results[0].ImagePath);
        Assert.Equal(2, results[0].Detections.Count);
        var first = results[0].Detections[0];
        Assert.Equal("defect", first.Label);
        Assert.Equal(0.9f, first.Score);
        Assert.Equal((10f, 20f, 30f, 40f), (first.X0, first.Y0, first.X1, first.Y1));
        var second = results[0].Detections[1];
        Assert.Equal("scratch", second.Label);
        Assert.Equal((50f, 60f, 70f, 80f), (second.X0, second.Y0, second.X1, second.Y1));

        Assert.Equal("b.jpg", results[1].ImagePath);
        Assert.Empty(results[1].Detections);
    }

    [Fact]
    public void Predict_MissingColumn_Throws()
    {
        var ml = new MLContext(seed: 42);
        // A data view that lacks the OD prediction columns entirely.
        var scored = ml.Data.LoadFromEnumerable(new[] { new { ImagePath = "a.jpg" } });

        var ex = Assert.Throws<InvalidOperationException>(() => ObjectDetectionPredictor.Predict(ml, scored));
        Assert.Contains("PredictedLabel", ex.Message);
    }

    private static bool TorchRuntimeInstalled()
    {
        var runtime = RuntimeRegistry.GetRequiredByTask("object-detection");
        return runtime != null && new RuntimeManager().IsInstalled(runtime);
    }

    [Fact]
    public void Predict_OnRealScoredModel_ReturnsDetectionsPerImage()
    {
        if (!TorchRuntimeInstalled())
            return; // libtorch not installed (CI); integration coverage runs only where the runtime exists.

        RuntimeManager.EnsureRuntimeForTask("object-detection");

        var (imagesDir, labelsDir) = (Path.Combine(_tempDir, "images"), Path.Combine(_tempDir, "labels"));
        Directory.CreateDirectory(imagesDir);
        Directory.CreateDirectory(labelsDir);
        File.WriteAllText(Path.Combine(_tempDir, "classes.txt"), "defect\n");

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

        var results = ObjectDetectionPredictor.Predict(ml, scored);

        Assert.NotEmpty(results);
        foreach (var image in results)
        {
            Assert.False(string.IsNullOrEmpty(image.ImagePath));
            foreach (var d in image.Detections)
                Assert.InRange(d.Score, 0.0f, 1.0f);
        }
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
