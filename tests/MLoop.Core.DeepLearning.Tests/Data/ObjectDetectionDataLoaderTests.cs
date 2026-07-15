using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.DeepLearning;

namespace MLoop.Core.DeepLearning.Tests.Data;

/// <summary>
/// Covers <see cref="ObjectDetectionDataLoader"/> and its wiring through
/// <see cref="DataLoaderFactory"/> + <see cref="DeepLearningRegistry"/>. Split out of
/// MLoop.Core.Tests's CocoDataLoaderTests/YoloDataLoaderTests (upstream-007 stage 2 t3):
/// <see cref="ObjectDetectionDataLoader"/> moved to MLoop.Core.DeepLearning, which
/// MLoop.Core.Tests does not reference. Registering <see cref="DeepLearningModule"/> here mirrors
/// what MLoop.CLI/MLoop.API do at startup (Task 5).
/// </summary>
public sealed class ObjectDetectionDataLoaderTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;

    static ObjectDetectionDataLoaderTests()
    {
        // Idempotent: DeepLearningRegistry is process-wide, so register once per test run.
        DeepLearningRegistry.Register(new DeepLearningModule());
    }

    public ObjectDetectionDataLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopObjectDetectionLoaderTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
            Directory.Delete(_tempDirectory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void DataLoaderFactory_ReturnsObjectDetectionLoaderForObjectDetection()
    {
        // object-detection routes to the format-detecting dispatcher (COCO or YOLO).
        var loader = DataLoaderFactory.Create("object-detection", _mlContext);
        Assert.IsType<ObjectDetectionDataLoader>(loader);
        Assert.True(DataLoaderFactory.IsDirectoryBased("object-detection"));
    }

    [Fact]
    public void ObjectDetectionLoader_DispatchesToCocoForJsonAnnotations()
    {
        CreateImage("img1.jpg");
        CreateImage("img2.jpg");
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"},{\"id\":2,\"file_name\":\"img2.jpg\"}]," +
            "\"annotations\":[" +
            "{\"image_id\":1,\"category_id\":1,\"bbox\":[0,0,10,10]}," +
            "{\"image_id\":2,\"category_id\":2,\"bbox\":[5,5,15,15]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"},{\"id\":2,\"name\":\"person\"}]}");
        var loader = new ObjectDetectionDataLoader(_mlContext, _ => { });

        var data = loader.LoadData(_tempDirectory, taskType: "object-detection");

        Assert.Contains(data.Schema, c => c.Name == CocoDataLoader.BoundingBoxColumn);
        Assert.Equal(2, data.GetRowCount());
    }

    [Fact]
    public void DataLoaderFactory_ReturnsObjectDetectionLoaderAndDispatchesYolo()
    {
        var loader = DataLoaderFactory.Create("object-detection", _mlContext, _ => { });
        Assert.IsType<ObjectDetectionDataLoader>(loader);

        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "0 0.5 0.5 0.2 0.2");

        // The dispatcher detects the YOLO layout and produces the OD schema.
        var data = loader.LoadData(_tempDirectory, taskType: "object-detection");
        Assert.Contains(data.Schema, c => c.Name == YoloDataLoader.BoundingBoxColumn);
        Assert.Equal(1, data.GetRowCount());
    }

    // --- helpers ---

    private void CreateImage(string fileName)
    {
        File.WriteAllBytes(Path.Combine(_tempDirectory, fileName), new byte[] { 0xFF, 0xD8, 0xFF });
    }

    private void WriteAnnotations(string json)
    {
        File.WriteAllText(Path.Combine(_tempDirectory, "annotations.json"), json);
    }

    private (string imagesDir, string labelsDir) CreateImagesLabelsDirs()
    {
        var images = Path.Combine(_tempDirectory, "images");
        var labels = Path.Combine(_tempDirectory, "labels");
        Directory.CreateDirectory(images);
        Directory.CreateDirectory(labels);
        return (images, labels);
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
