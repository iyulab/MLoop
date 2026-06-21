using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class CocoDataLoaderTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;
    private readonly CocoDataLoader _loader;

    public CocoDataLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopCocoTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
        _loader = new CocoDataLoader(_mlContext, _ => { });
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
        Assert.Throws<ArgumentNullException>(() => new CocoDataLoader(null!));
    }

    [Fact]
    public void LoadData_WithNonExistentDirectory_Throws()
    {
        var missing = Path.Combine(_tempDirectory, "does-not-exist");
        Assert.Throws<DirectoryNotFoundException>(() => _loader.LoadData(missing));
    }

    [Fact]
    public void LoadData_WithNoAnnotationFile_Throws()
    {
        CreateImage("img1.jpg");
        Assert.Throws<FileNotFoundException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithMultipleJsonFiles_Throws()
    {
        File.WriteAllText(Path.Combine(_tempDirectory, "a.json"), "{}");
        File.WriteAllText(Path.Combine(_tempDirectory, "b.json"), "{}");
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithNoImages_Throws()
    {
        WriteAnnotations("{\"images\":[],\"annotations\":[],\"categories\":[]}");
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithNoAnnotations_Throws()
    {
        CreateImage("img1.jpg");
        WriteAnnotations("{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"}],\"annotations\":[],\"categories\":[]}");
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithValidCoco_ReturnsThreeColumns()
    {
        CreateValidDataset();

        var data = _loader.LoadData(_tempDirectory);

        Assert.Contains(data.Schema, c => c.Name == CocoDataLoader.ImagePathColumn);
        Assert.Contains(data.Schema, c => c.Name == CocoDataLoader.DefaultLabelColumn);
        Assert.Contains(data.Schema, c => c.Name == CocoDataLoader.BoundingBoxColumn);
        // Two images each with annotations => 2 rows.
        Assert.Equal(2, data.GetRowCount());
    }

    [Fact]
    public void LoadData_ImagePaths_AreAbsolute()
    {
        CreateValidDataset();

        var data = _loader.LoadData(_tempDirectory);
        var paths = data.GetColumn<string>(CocoDataLoader.ImagePathColumn).ToList();

        Assert.All(paths, p => Assert.True(Path.IsPathRooted(p)));
        Assert.All(paths, p => Assert.True(File.Exists(p)));
    }

    [Fact]
    public void LoadData_ConvertsBboxToCornerCoordinates()
    {
        CreateImage("img1.jpg");
        // One box: x=10, y=20, w=30, h=40 => x0=10 y0=20 x1=40 y1=60.
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"}]," +
            "\"annotations\":[{\"image_id\":1,\"category_id\":1,\"bbox\":[10,20,30,40]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"}]}");

        var data = _loader.LoadData(_tempDirectory);
        var boxes = data.GetColumn<VBuffer<float>>(CocoDataLoader.BoundingBoxColumn).First();

        Assert.Equal(new float[] { 10, 20, 40, 60 }, boxes.DenseValues().ToArray());
    }

    [Fact]
    public void LoadData_MultipleObjectsPerImage_FlattensBoxesAndLabels()
    {
        CreateImage("img1.jpg");
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"}]," +
            "\"annotations\":[" +
            "{\"image_id\":1,\"category_id\":1,\"bbox\":[0,0,10,10]}," +
            "{\"image_id\":1,\"category_id\":2,\"bbox\":[5,5,5,5]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"},{\"id\":2,\"name\":\"person\"}]}");

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<VBuffer<ReadOnlyMemory<char>>>(CocoDataLoader.DefaultLabelColumn).First();
        var boxes = data.GetColumn<VBuffer<float>>(CocoDataLoader.BoundingBoxColumn).First();

        var labelValues = labels.DenseValues().Select(v => v.ToString()).ToArray();
        Assert.Equal(new[] { "car", "person" }, labelValues);
        // 2 objects => 8 floats.
        Assert.Equal(8, boxes.DenseValues().Count());
    }

    [Fact]
    public void LoadData_SkipsAnnotationsForMissingImageFiles()
    {
        CreateImage("present.jpg");
        // img2 referenced by annotations but the file is absent on disk.
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"present.jpg\"},{\"id\":2,\"file_name\":\"absent.jpg\"}]," +
            "\"annotations\":[" +
            "{\"image_id\":1,\"category_id\":1,\"bbox\":[0,0,10,10]}," +
            "{\"image_id\":2,\"category_id\":1,\"bbox\":[0,0,10,10]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"}]}");

        var data = _loader.LoadData(_tempDirectory);

        Assert.Equal(1, data.GetRowCount());
    }

    [Fact]
    public void LoadData_UnknownCategoryId_NamedGenerically()
    {
        CreateImage("img1.jpg");
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"}]," +
            "\"annotations\":[{\"image_id\":1,\"category_id\":99,\"bbox\":[0,0,10,10]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"}]}");

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<VBuffer<ReadOnlyMemory<char>>>(CocoDataLoader.DefaultLabelColumn).First();

        Assert.Equal("class_99", labels.DenseValues().First().ToString());
    }

    [Fact]
    public void LoadData_AcceptsDirectJsonPath()
    {
        CreateValidDataset();
        var jsonPath = Path.Combine(_tempDirectory, "annotations.json");

        var data = _loader.LoadData(jsonPath);

        Assert.Equal(2, data.GetRowCount());
    }

    [Fact]
    public void LoadData_ResolvesConventionalRoboflowName()
    {
        CreateImage("img1.jpg");
        File.WriteAllText(Path.Combine(_tempDirectory, "_annotations.coco.json"),
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"}]," +
            "\"annotations\":[{\"image_id\":1,\"category_id\":1,\"bbox\":[0,0,10,10]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"}]}");

        var data = _loader.LoadData(_tempDirectory);

        Assert.Equal(1, data.GetRowCount());
    }

    [Fact]
    public void LoadData_WithCustomLabelColumn_RenamesLabelColumn()
    {
        CreateValidDataset();

        var data = _loader.LoadData(_tempDirectory, labelColumn: "Category");

        Assert.Contains(data.Schema, c => c.Name == "Category");
    }

    [Fact]
    public void LoadData_WithMalformedJson_Throws()
    {
        CreateImage("img1.jpg");
        WriteAnnotations("{ this is not valid json ");
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void SplitData_ProducesTrainAndTestSets()
    {
        CreateImagesAndAnnotations(16);
        var data = _loader.LoadData(_tempDirectory);

        var (train, test) = _loader.SplitData(data, 0.25);

        var trainCount = train.GetColumn<string>(CocoDataLoader.ImagePathColumn).Count();
        var testCount = test.GetColumn<string>(CocoDataLoader.ImagePathColumn).Count();

        Assert.True(trainCount > 0);
        Assert.True(testCount > 0);
        Assert.Equal(16, trainCount + testCount);
    }

    [Fact]
    public void GetSchema_ReturnsColumnsAndRowCount()
    {
        CreateValidDataset();
        var data = _loader.LoadData(_tempDirectory);

        var schema = _loader.GetSchema(data);

        Assert.Equal(2, schema.RowCount);
        Assert.Contains(schema.Columns, c => c.Name == CocoDataLoader.ImagePathColumn);
        Assert.Contains(schema.Columns, c => c.Name == CocoDataLoader.BoundingBoxColumn);
    }

    [Fact]
    public void ValidateLabelColumn_ReturnsTrueWhenPresent()
    {
        CreateValidDataset();
        var data = _loader.LoadData(_tempDirectory);

        Assert.True(_loader.ValidateLabelColumn(data, CocoDataLoader.DefaultLabelColumn));
        Assert.False(_loader.ValidateLabelColumn(data, "Missing"));
    }

    [Fact]
    public void DataLoaderFactory_ReturnsCocoLoaderForObjectDetection()
    {
        var loader = DataLoaderFactory.Create("object-detection", _mlContext);
        Assert.IsType<CocoDataLoader>(loader);
        Assert.True(DataLoaderFactory.IsDirectoryBased("object-detection"));
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

    private void CreateValidDataset()
    {
        CreateImage("img1.jpg");
        CreateImage("img2.jpg");
        WriteAnnotations(
            "{\"images\":[{\"id\":1,\"file_name\":\"img1.jpg\"},{\"id\":2,\"file_name\":\"img2.jpg\"}]," +
            "\"annotations\":[" +
            "{\"image_id\":1,\"category_id\":1,\"bbox\":[0,0,10,10]}," +
            "{\"image_id\":2,\"category_id\":2,\"bbox\":[5,5,15,15]}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"},{\"id\":2,\"name\":\"person\"}]}");
    }

    private void CreateImagesAndAnnotations(int count)
    {
        var images = new List<string>();
        var annotations = new List<string>();
        for (var i = 1; i <= count; i++)
        {
            CreateImage($"img{i}.jpg");
            images.Add($"{{\"id\":{i},\"file_name\":\"img{i}.jpg\"}}");
            annotations.Add($"{{\"image_id\":{i},\"category_id\":1,\"bbox\":[0,0,10,10]}}");
        }
        WriteAnnotations(
            $"{{\"images\":[{string.Join(",", images)}]," +
            $"\"annotations\":[{string.Join(",", annotations)}]," +
            "\"categories\":[{\"id\":1,\"name\":\"car\"}]}");
    }
}
