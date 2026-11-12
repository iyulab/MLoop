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
        Directory.CreateDirectory(Path.Combine(classes, "cat"));

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
    public void ATableTask_IsNotAskedAboutAFolder()
        => Assert.Null(TrainingInput.DirectoryDatasetRefusal("regression", Dir("anything")));
}
