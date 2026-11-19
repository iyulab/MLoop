using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A text label of a binary task becomes a boolean, and the loader says so. One loader reads both partitions of a training run, so the
/// statement — which is about the file's label, not about a partition — was printed twice in a row.
/// </summary>
public sealed class LabelConversionAnnouncementTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-lcvt-" + Guid.NewGuid().ToString("N"));

    public LabelConversionAnnouncementTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Table()
    {
        var path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllLines(path, ["x,y,outcome", .. Enumerable.Range(0, 60).Select(i => $"{i},{i % 7},{(i % 3 == 0 ? "malignant" : "benign")}")]);
        return path;
    }

    [Fact]
    public void ALoaderThatReadsTwoPartitions_SaysOnce_WhichTextBecomesWhichBoolean()
    {
        var said = new List<string>();
        var loader = new CsvDataLoader(new MLContext(seed: 0), said.Add);

        loader.LoadData(Table(), "outcome", "binary-classification");
        loader.LoadData(Table(), "outcome", "binary-classification");

        Assert.Single(said, m => m.StartsWith("[Info] Converting label:", StringComparison.Ordinal));
    }

    [Fact]
    public void AnotherLoader_SaysItAgain()
    {
        var first = new List<string>();
        var second = new List<string>();

        new CsvDataLoader(new MLContext(seed: 0), first.Add).LoadData(Table(), "outcome", "binary-classification");
        new CsvDataLoader(new MLContext(seed: 0), second.Add).LoadData(Table(), "outcome", "binary-classification");

        Assert.Single(first, m => m.Contains("Converting label", StringComparison.Ordinal));
        Assert.Single(second, m => m.Contains("Converting label", StringComparison.Ordinal));
    }
}
