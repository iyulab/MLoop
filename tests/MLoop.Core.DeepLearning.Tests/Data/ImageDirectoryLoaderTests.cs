using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.DeepLearning.Tests.Data;

public class ImageDirectoryLoaderTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;
    private readonly ImageDirectoryLoader _loader;

    public ImageDirectoryLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopImgTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
        _loader = new ImageDirectoryLoader(_mlContext, _ => { });
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
        Assert.Throws<ArgumentNullException>(() => new ImageDirectoryLoader(null!));
    }

    [Fact]
    public void LoadData_WithNonExistentDirectory_Throws()
    {
        var missing = Path.Combine(_tempDirectory, "does-not-exist");
        Assert.Throws<DirectoryNotFoundException>(() => _loader.LoadData(missing));
    }

    [Fact]
    public void LoadData_WithNoClassSubfolders_Throws()
    {
        // Files directly under the root, no class subfolders.
        File.WriteAllBytes(Path.Combine(_tempDirectory, "loose.jpg"), new byte[] { 1 });
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithClassFoldersButNoImages_Throws()
    {
        Directory.CreateDirectory(Path.Combine(_tempDirectory, "OK"));
        Directory.CreateDirectory(Path.Combine(_tempDirectory, "NG"));
        Assert.Throws<InvalidOperationException>(() => _loader.LoadData(_tempDirectory));
    }

    [Fact]
    public void LoadData_WithValidLayout_ReturnsImagePathAndLabelColumns()
    {
        CreateClass("OK", 3);
        CreateClass("NG", 2);

        var data = _loader.LoadData(_tempDirectory);

        Assert.Contains(data.Schema, c => c.Name == ImageDirectoryLoader.ImagePathColumn);
        Assert.Contains(data.Schema, c => c.Name == ImageDirectoryLoader.DefaultLabelColumn);
        Assert.Equal(5, data.GetRowCount());
    }

    [Fact]
    public void LoadData_LabelValues_MatchFolderNames()
    {
        CreateClass("OK", 2);
        CreateClass("NG", 1);

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).ToList();

        Assert.Equal(3, labels.Count);
        Assert.Equal(2, labels.Count(l => l == "OK"));
        Assert.Equal(1, labels.Count(l => l == "NG"));
    }

    [Fact]
    public void LoadData_ImagePaths_AreAbsolute()
    {
        CreateClass("OK", 2);

        var data = _loader.LoadData(_tempDirectory);
        var paths = data.GetColumn<string>(ImageDirectoryLoader.ImagePathColumn).ToList();

        Assert.All(paths, p => Assert.True(Path.IsPathRooted(p)));
        Assert.All(paths, p => Assert.True(File.Exists(p)));
    }

    [Fact]
    public void LoadData_IgnoresUnsupportedExtensions()
    {
        var okDir = Path.Combine(_tempDirectory, "OK");
        Directory.CreateDirectory(okDir);
        File.WriteAllBytes(Path.Combine(okDir, "img1.jpg"), new byte[] { 1 });
        File.WriteAllText(Path.Combine(okDir, "notes.txt"), "ignore me");
        File.WriteAllText(Path.Combine(okDir, "meta.json"), "{}");
        CreateClass("NG", 1);

        var data = _loader.LoadData(_tempDirectory);

        // Only the .jpg under OK plus the one NG image => 2 rows.
        Assert.Equal(2, data.GetRowCount());
    }

    [Fact]
    public void LoadData_WithCustomLabelColumn_RenamesLabelColumn()
    {
        CreateClass("OK", 1);
        CreateClass("NG", 1);

        var data = _loader.LoadData(_tempDirectory, labelColumn: "Category");

        Assert.Contains(data.Schema, c => c.Name == "Category");
        var values = data.GetColumn<string>("Category").OrderBy(x => x).ToList();
        Assert.Equal(new[] { "NG", "OK" }, values);
    }

    [Fact]
    public void LoadData_SkipsEmptyClassFolder()
    {
        CreateClass("OK", 2);
        Directory.CreateDirectory(Path.Combine(_tempDirectory, "EMPTY"));
        CreateClass("NG", 1);

        var data = _loader.LoadData(_tempDirectory);
        var labels = data.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).ToList();

        Assert.Equal(3, labels.Count);
        Assert.DoesNotContain("EMPTY", labels);
    }

    [Fact]
    public void SplitData_ProducesTrainAndTestSets()
    {
        CreateClass("OK", 8);
        CreateClass("NG", 8);
        var data = _loader.LoadData(_tempDirectory);

        var (train, test) = _loader.SplitData(data, 0.25);

        // TrainTestSplit produces lazy filtered views whose GetRowCount() is null,
        // so materialize the label column to count rows.
        var trainCount = train.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).Count();
        var testCount = test.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).Count();

        Assert.True(trainCount > 0);
        Assert.True(testCount > 0);
        Assert.Equal(16, trainCount + testCount);
    }

    [Fact]
    public void SplitData_WithZeroFraction_UsesAllDataForBoth()
    {
        CreateClass("OK", 2);
        CreateClass("NG", 2);
        var data = _loader.LoadData(_tempDirectory);

        var (train, test) = _loader.SplitData(data, 0);

        Assert.Equal(4, train.GetRowCount());
        Assert.Equal(4, test.GetRowCount());
    }

    [Fact]
    public void ValidateLabelColumn_ReturnsTrueWhenPresent()
    {
        CreateClass("OK", 1);
        CreateClass("NG", 1);
        var data = _loader.LoadData(_tempDirectory);

        Assert.True(_loader.ValidateLabelColumn(data, ImageDirectoryLoader.DefaultLabelColumn));
        Assert.False(_loader.ValidateLabelColumn(data, "Missing"));
    }

    [Fact]
    public void GetSchema_ReturnsColumnsAndRowCount()
    {
        CreateClass("OK", 2);
        CreateClass("NG", 1);
        var data = _loader.LoadData(_tempDirectory);

        var schema = _loader.GetSchema(data);

        Assert.Equal(3, schema.RowCount);
        Assert.Contains(schema.Columns, c => c.Name == ImageDirectoryLoader.ImagePathColumn);
        Assert.Contains(schema.Columns, c => c.Name == ImageDirectoryLoader.DefaultLabelColumn);
    }

    [Fact]
    public void DataLoaderFactory_ReturnsImageLoaderForImageClassification()
    {
        var loader = DataLoaderFactory.Create("image-classification", _mlContext);
        Assert.IsType<ImageDirectoryLoader>(loader);
        Assert.True(DataLoaderFactory.IsDirectoryBased("image-classification"));
    }

    [Fact]
    public void DataLoaderFactory_ReturnsCsvLoaderForTabularTasks()
    {
        var loader = DataLoaderFactory.Create("regression", _mlContext);
        Assert.IsType<CsvDataLoader>(loader);
        Assert.False(DataLoaderFactory.IsDirectoryBased("regression"));
    }

    [Fact]
    public void CountClasses_CountsFoldersWithSupportedImages()
    {
        CreateClass("OK", 3);
        CreateClass("NG", 2);

        Assert.Equal(2, ImageDirectoryLoader.CountClasses(_tempDirectory));
    }

    [Fact]
    public void CountClasses_IgnoresEmptyAndUnsupportedOnlyFolders()
    {
        CreateClass("OK", 1);
        Directory.CreateDirectory(Path.Combine(_tempDirectory, "EMPTY"));            // no files
        var txtOnly = Path.Combine(_tempDirectory, "TXT");
        Directory.CreateDirectory(txtOnly);
        File.WriteAllText(Path.Combine(txtOnly, "notes.txt"), "no images here");      // unsupported only
        CreateClass("NG", 1);

        // OK and NG count; EMPTY and TXT do not.
        Assert.Equal(2, ImageDirectoryLoader.CountClasses(_tempDirectory));
    }

    [Fact]
    public void CountClasses_MissingDirectory_ReturnsZero()
    {
        Assert.Equal(0, ImageDirectoryLoader.CountClasses(Path.Combine(_tempDirectory, "nope")));
    }

    [Fact]
    public void CountImagesPerClass_ReportsEachClassFolderInOrder()
    {
        CreateClass("NG", 2);
        CreateClass("OK", 3);
        Directory.CreateDirectory(Path.Combine(_tempDirectory, "empty"));

        var counts = ImageDirectoryLoader.CountImagesPerClass(_tempDirectory);

        Assert.Equal(["NG", "OK"], counts.Keys);
        Assert.Equal([2, 3], counts.Values);
    }

    [Fact]
    public void ListImages_FindsImagesAtAnyDepthAndSkipsOtherFiles()
    {
        // A folder of new pictures may be flat or still sorted into subfolders.
        CreateClass("OK", 2);
        File.WriteAllBytes(Path.Combine(_tempDirectory, "loose.png"), [0x89]);
        File.WriteAllText(Path.Combine(_tempDirectory, "README.txt"), "not an image");

        var images = ImageDirectoryLoader.ListImages(_tempDirectory);

        Assert.Equal(3, images.Count);
        Assert.All(images, p => Assert.True(Path.IsPathRooted(p)));
        Assert.DoesNotContain(images, p => p.EndsWith(".txt"));
    }

    [Fact]
    public void WriteImageTable_IsTheTableTheModelScores_AndQuotesPathsThatNeedIt()
    {
        var odd = Path.Combine(_tempDirectory, "a,b \"c\".jpg");
        var table = ImageDirectoryLoader.WriteImageTable([odd, "/plain.jpg"]);
        try
        {
            Assert.Equal(".csv", Path.GetExtension(table));
            using var reader = new CsvHelper.CsvReader(new StreamReader(table),
                System.Globalization.CultureInfo.InvariantCulture);
            var rows = reader.GetRecords<dynamic>().Select(r => (string)((IDictionary<string, object>)r)[ImageDirectoryLoader.ImagePathColumn]).ToList();
            Assert.Equal([odd, "/plain.jpg"], rows);
        }
        finally
        {
            File.Delete(table);
        }
    }

    [Fact]
    public void Balance_OversamplesTheSmallerClassToTheRatio_WithItsOwnImages()
    {
        CreateClass("OK", 9);
        CreateClass("NG", 1);
        var data = _loader.LoadData(_tempDirectory);

        var (balanced, outcome) = ImageDirectoryLoader.Balance(_mlContext, data, null, "3");

        var labels = balanced.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).ToList();
        var paths = balanced.GetColumn<string>(ImageDirectoryLoader.ImagePathColumn).ToList();
        Assert.True(outcome.Applied);
        Assert.Equal(9, labels.Count(l => l == "OK"));
        Assert.Equal(3, labels.Count(l => l == "NG")); // ⌈9 ÷ 3⌉
        var ngImage = Path.Combine(_tempDirectory, "NG", "NG_0.jpg");
        Assert.All(paths.Where((_, i) => labels[i] == "NG"), p => Assert.Equal(ngImage, p));
    }

    [Fact]
    public void Balance_KeepsACustomLabelColumn()
    {
        CreateClass("OK", 6);
        CreateClass("NG", 1);
        var data = _loader.LoadData(_tempDirectory, labelColumn: "defect");

        var (balanced, _) = ImageDirectoryLoader.Balance(_mlContext, data, "defect", "2");

        Assert.Equal(3, balanced.GetColumn<string>("defect").Count(l => l == "NG"));
        Assert.Contains(balanced.Schema, c => c.Name == ImageDirectoryLoader.DefaultLabelColumn);
    }

    [Fact]
    public void Balance_WhenTheRatioIsAlreadyMet_ReturnsThePartitionUnchanged()
    {
        CreateClass("OK", 4);
        CreateClass("NG", 2);
        var data = _loader.LoadData(_tempDirectory);

        var (balanced, outcome) = ImageDirectoryLoader.Balance(_mlContext, data, null, "3");

        Assert.False(outcome.Applied);
        Assert.Same(data, balanced);
        Assert.Contains("already meets", outcome.Message);
    }

    [Fact]
    public void HoldOut_TakesEachClassShare_AndNoImageIsOnBothSides()
    {
        CreateClass("OK", 30);
        CreateClass("NG", 10);
        CreateClass("RARE", 1);
        var data = _loader.LoadData(_tempDirectory);

        var (train, validation) = ImageDirectoryLoader.HoldOut(_mlContext, data, null, 0.1);

        Assert.NotNull(validation);
        var held = validation!.GetColumn<string>(ImageDirectoryLoader.DefaultLabelColumn).ToList();
        Assert.Equal(3, held.Count(l => l == "OK"));
        Assert.Equal(1, held.Count(l => l == "NG"));
        Assert.DoesNotContain("RARE", held); // one image cannot be spared
        var trainPaths = train.GetColumn<string>(ImageDirectoryLoader.ImagePathColumn).ToHashSet();
        Assert.Equal(37, trainPaths.Count);
        Assert.DoesNotContain(validation.GetColumn<string>(ImageDirectoryLoader.ImagePathColumn), trainPaths.Contains);
    }

    [Fact]
    public void HoldOut_WhenNoClassCanSpareAnImage_ReturnsNoValidation()
    {
        CreateClass("OK", 5);
        CreateClass("NG", 1);
        var data = _loader.LoadData(_tempDirectory);

        var (train, validation) = ImageDirectoryLoader.HoldOut(_mlContext, data, null, 0.1);

        Assert.Null(validation);
        Assert.Same(data, train);
    }

    /// <summary>Creates a class subfolder with <paramref name="count"/> placeholder image files.</summary>
    private void CreateClass(string label, int count)
    {
        var dir = Path.Combine(_tempDirectory, label);
        Directory.CreateDirectory(dir);
        for (var i = 0; i < count; i++)
            File.WriteAllBytes(Path.Combine(dir, $"{label}_{i}.jpg"), new byte[] { 0xFF, 0xD8, 0xFF });
    }
}
