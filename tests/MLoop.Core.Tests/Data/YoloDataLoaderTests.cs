using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class YoloDataLoaderTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;
    private readonly YoloDataLoader _loader;

    public YoloDataLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopYoloTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
        _loader = new YoloDataLoader(_mlContext, _ => { });
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
            Directory.Delete(_tempDirectory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void Constructor_WithNullMLContext_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new YoloDataLoader(null!));
    }

    [Fact]
    public void LoadData_WithNonExistentDirectory_Throws()
    {
        var missing = Path.Combine(_tempDirectory, "nope");
        Assert.Throws<DirectoryNotFoundException>(() => _loader.LoadData(missing));
    }

    [Fact]
    public void LoadData_WithNoYoloLayout_Throws()
    {
        // Empty directory: no images/ + labels/ pair, no flat images.
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void IsYoloDirectory_DetectsImagesLabelsPair()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "0 0.5 0.5 0.2 0.2");

        Assert.True(YoloDataLoader.IsYoloDirectory(_tempDirectory));
        Assert.False(YoloDataLoader.IsYoloDirectory(Path.Combine(_tempDirectory, "missing")));
    }

    [Fact]
    public void LoadData_ProducesThreeColumns()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        CreateBmp(Path.Combine(imagesDir, "b.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "0 0.5 0.5 0.2 0.2");
        File.WriteAllText(Path.Combine(labelsDir, "b.txt"), "1 0.5 0.5 0.4 0.4");

        var data = _loader.LoadData(_tempDirectory);

        Assert.Contains(data.Schema, c => c.Name == YoloDataLoader.ImagePathColumn);
        Assert.Contains(data.Schema, c => c.Name == YoloDataLoader.DefaultLabelColumn);
        Assert.Contains(data.Schema, c => c.Name == YoloDataLoader.BoundingBoxColumn);
        Assert.Equal(2, data.GetRowCount());
    }

    [Fact]
    public void LoadData_ConvertsNormalizedToAbsoluteCorners()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        // cx=0.5 cy=0.5 w=0.2 h=0.2 on 100x100 => x0=40 y0=40 x1=60 y1=60
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "0 0.5 0.5 0.2 0.2");

        var data = _loader.LoadData(_tempDirectory);
        var box = data.GetColumn<VBuffer<float>>(YoloDataLoader.BoundingBoxColumn).First();

        var actual = box.DenseValues().ToArray();
        var expected = new float[] { 40, 40, 60, 60 };
        Assert.Equal(expected.Length, actual.Length);
        // Normalized→absolute conversion is exact up to float rounding (0.6f*100 ≈ 60.0000038).
        for (var i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    [Fact]
    public void LoadData_CollapsesDuplicateBoxes()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        // Same box repeated 5x (KAMP export artifact) => collapses to 1.
        var line = "0 0.5 0.5 0.2 0.2";
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), string.Join("\n", Enumerable.Repeat(line, 5)));

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<VBuffer<ReadOnlyMemory<char>>>(YoloDataLoader.DefaultLabelColumn).First();

        Assert.Single(labels.DenseValues());
    }

    [Fact]
    public void LoadData_SkipsImagesWithoutLabels()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        CreateBmp(Path.Combine(imagesDir, "b.bmp"), 100, 100); // no label for b
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "0 0.5 0.5 0.2 0.2");

        var data = _loader.LoadData(_tempDirectory);
        Assert.Equal(1, data.GetRowCount());
    }

    [Fact]
    public void LoadData_UsesClassNamesFile()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "1 0.5 0.5 0.2 0.2");
        File.WriteAllText(Path.Combine(_tempDirectory, "classes.txt"), "background\ndefect\n");

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<VBuffer<ReadOnlyMemory<char>>>(YoloDataLoader.DefaultLabelColumn).First();

        // class id 1 => second name "defect"
        Assert.Equal("defect", labels.DenseValues().First().ToString());
    }

    [Fact]
    public void LoadData_WithoutClassNames_UsesGenericNames()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        CreateBmp(Path.Combine(imagesDir, "a.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(labelsDir, "a.txt"), "2 0.5 0.5 0.2 0.2");

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<VBuffer<ReadOnlyMemory<char>>>(YoloDataLoader.DefaultLabelColumn).First();

        Assert.Equal("class_2", labels.DenseValues().First().ToString());
    }

    [Fact]
    public void LoadData_FlatLayout_Works()
    {
        // Images and .txt labels in the same directory (no images/ + labels/ subdirs).
        CreateBmp(Path.Combine(_tempDirectory, "a.bmp"), 100, 100);
        File.WriteAllText(Path.Combine(_tempDirectory, "a.txt"), "0 0.5 0.5 0.2 0.2");

        var data = _loader.LoadData(_tempDirectory);
        Assert.Equal(1, data.GetRowCount());
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

    [Fact]
    public void SplitData_ProducesTrainAndTestSets()
    {
        var (imagesDir, labelsDir) = CreateImagesLabelsDirs();
        for (var i = 0; i < 16; i++)
        {
            CreateBmp(Path.Combine(imagesDir, $"img{i}.bmp"), 100, 100);
            File.WriteAllText(Path.Combine(labelsDir, $"img{i}.txt"), "0 0.5 0.5 0.2 0.2");
        }
        var data = _loader.LoadData(_tempDirectory);

        var (train, test) = _loader.SplitData(data, 0.25);
        var trainCount = train.GetColumn<string>(YoloDataLoader.ImagePathColumn).Count();
        var testCount = test.GetColumn<string>(YoloDataLoader.ImagePathColumn).Count();

        Assert.True(trainCount > 0 && testCount > 0);
        Assert.Equal(16, trainCount + testCount);
    }

    // --- helpers ---

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
