using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Configuration;

/// <summary>A directory that is not the layout the task reads is refused before a run is queued — <c>mloop train</c> and <c>POST /train</c> ask the same function.</summary>
public sealed class DirectoryDatasetRefusalTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-dirds-{Guid.NewGuid():N}");

    public DirectoryDatasetRefusalTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private string Dir(string name)
    {
        var path = Path.Combine(_root, name);
        Directory.CreateDirectory(path);
        return path;
    }

    [Fact]
    public void ClassificationNeedsClassSubfolders()
    {
        var flat = Dir("flat");
        File.WriteAllBytes(Path.Combine(flat, "a.png"), [1]);
        var classes = Dir("classes");
        Pictures(classes, "cat", 1);
        Pictures(classes, "dog", 1);

        var refusal = TrainingInput.DirectoryDatasetRefusal("image-classification", flat);

        Assert.NotNull(refusal);
        Assert.Contains("No class subfolders found", refusal!.Message, StringComparison.Ordinal);
        Assert.NotNull(refusal.Hint);
        Assert.Null(TrainingInput.DirectoryDatasetRefusal("image-classification", classes));
    }

    [Fact]
    public void DetectionNeedsOneAnnotationsFile()
    {
        var none = Dir("none");
        var one = Dir("one");
        File.WriteAllText(Path.Combine(one, "annotations.json"), "{}");
        var two = Dir("two");
        File.WriteAllText(Path.Combine(two, "a.json"), "{}");
        File.WriteAllText(Path.Combine(two, "b.json"), "{}");

        Assert.Contains("No COCO annotations file found", TrainingInput.DirectoryDatasetRefusal("object-detection", none)!.Message, StringComparison.Ordinal);
        Assert.Contains("Multiple JSON files", TrainingInput.DirectoryDatasetRefusal("object-detection", two)!.Message, StringComparison.Ordinal);
        Assert.Null(TrainingInput.DirectoryDatasetRefusal("object-detection", one));
    }

    [Fact]
    public void AClassFolderWithNoPictureIsRefusedByName()
    {
        var root = Dir("empty-class");
        Pictures(root, "circle", 2);
        Pictures(root, "square", 2);
        Directory.CreateDirectory(Path.Combine(root, "triangle"));
        Directory.CreateDirectory(Path.Combine(root, ".ipynb_checkpoints")); // a tool's own folder is not a class

        var refusal = TrainingInput.DirectoryDatasetRefusal("image-classification", root)!;

        Assert.Contains("holds no picture: triangle.", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("remove the folder", refusal.Hint!, StringComparison.Ordinal);
        Directory.Delete(Path.Combine(root, "triangle"));
        Assert.Null(TrainingInput.DirectoryDatasetRefusal("image-classification", root));
    }

    [Fact]
    public void OneClassIsRefusedByName()
    {
        var root = Dir("one-class");
        Pictures(root, "circle", 3);

        var refusal = TrainingInput.DirectoryDatasetRefusal("image-classification", root)!;

        Assert.Contains("at least two classes", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("circle", refusal.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnnotationsNamingAMissingPictureOrABoxOutsideItAreRefusedByName()
    {
        var root = Dir("coco");
        File.WriteAllBytes(Path.Combine(root, "a.png"), Png(20, 10));
        File.WriteAllText(Path.Combine(root, "annotations.json"), """
            {"images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "gone.png"}],
             "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 20.9, 10]},
                             {"image_id": 1, "category_id": 1, "bbox": [15, 2, 12, 5]},
                             {"image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]}],
             "categories": [{"id": 1, "name": "box"}]}
            """);

        var refusal = TrainingInput.DirectoryDatasetRefusal("object-detection", root)!;

        Assert.Contains("annotations.json", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("1 picture it names is not there: gone.png", refusal.Message, StringComparison.Ordinal);
        // The box a pixel over the edge is rounding; the one seven pixels over is not.
        Assert.Contains("1 box is empty or reaches outside its picture: a.png (20x10) has box x=15, y=2, width=12, height=5", refusal.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AYoloFolderIsNotAskedForACocoFile()
    {
        var root = Dir("yolo");
        Directory.CreateDirectory(Path.Combine(root, "images"));
        Directory.CreateDirectory(Path.Combine(root, "labels"));
        File.WriteAllBytes(Path.Combine(root, "images", "a.png"), Png(8, 8));
        File.WriteAllText(Path.Combine(root, "labels", "a.txt"), "0 0.5 0.5 0.25 0.25\n");

        Assert.Null(TrainingInput.DirectoryDatasetRefusal("object-detection", root));
    }

    [Fact]
    public void AnImageAnomalyFolderWithNoPictureIsRefused()
    {
        var root = Dir("normal");
        File.WriteAllText(Path.Combine(root, "notes.txt"), "none yet");

        Assert.Contains("No pictures in", TrainingInput.DirectoryDatasetRefusal("image-anomaly", root)!.Message, StringComparison.Ordinal);
        File.WriteAllBytes(Path.Combine(root, "a.png"), Png(8, 8));
        Assert.Null(TrainingInput.DirectoryDatasetRefusal("image-anomaly", root));
    }

    [Fact]
    public void AnImageAnomalyProjectDoesNotTakeAllOfDatasetsAsNormalPictures()
    {
        var project = Dir("project");
        Directory.CreateDirectory(Path.Combine(project, "datasets", "check", "scratch"));

        Assert.Null(TrainingInput.ResolveDirectoryDataset("image-anomaly", null, null, project));
        Directory.CreateDirectory(Path.Combine(project, "datasets", "normal"));
        Assert.EndsWith("normal", TrainingInput.ResolveDirectoryDataset("image-anomaly", null, null, project), StringComparison.Ordinal);
    }

    private static void Pictures(string root, string name, int count)
    {
        var folder = Path.Combine(root, name);
        Directory.CreateDirectory(folder);
        for (var i = 0; i < count; i++)
            File.WriteAllBytes(Path.Combine(folder, $"{i}.png"), Png(4, 4));
    }

    private static byte[] Png(int width, int height)
    {
        using var bitmap = new SkiaSharp.SKBitmap(width, height);
        using var data = bitmap.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    [Fact]
    public void ATableTask_IsNotAskedAboutAFolder()
        => Assert.Null(TrainingInput.DirectoryDatasetRefusal("regression", Dir("anything")));
}
