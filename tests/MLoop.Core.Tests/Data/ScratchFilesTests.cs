using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// The temp copies of a data file are deleted: each when the step after it has written its own, the last when the
/// process ends — and nothing that was not made here, the user's own file above all.
/// </summary>
public sealed class ScratchFilesTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-scratch-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static HashSet<string> Copies() =>
        Directory.EnumerateFiles(Path.GetTempPath(), "mloop_*").ToHashSet(StringComparer.Ordinal);

    [Fact]
    public void ACopyMadeHere_IsDeletedWhenReleased()
    {
        var path = ScratchFiles.New("mloop_test_", ".csv");
        File.WriteAllText(path, "a\n1\n");

        ScratchFiles.Release(path);

        Assert.False(File.Exists(path));
        Assert.False(ScratchFiles.Owns(path));
    }

    [Fact]
    public void AFileNotMadeHere_IsNeverDeleted()
    {
        var users = Path.Combine(_dir, "train.csv");
        File.WriteAllText(users, "a\n1\n");

        ScratchFiles.Release(users);
        ScratchFiles.Advance(users, ScratchFiles.New("mloop_test_", ".csv"), input: Path.Combine(_dir, "other.csv"));

        Assert.True(File.Exists(users));
    }

    [Fact]
    public void AChainsInput_IsKept_EvenWhenItIsACopyMadeHere()
    {
        var shared = ScratchFiles.New("mloop_test_", ".csv");
        File.WriteAllText(shared, "a\n1\n");

        var next = ScratchFiles.Advance(shared, ScratchFiles.New("mloop_test_", ".csv"), input: shared);

        Assert.True(File.Exists(shared));
        ScratchFiles.Release(shared);
        ScratchFiles.Release(next);
    }

    /// <summary>
    /// A file every preparing step rewrites — a CP949 encoding, a multi-line quoted field, a constant and a sparse
    /// column — leaves one new copy behind after loading: the one the loaded view still reads. It left one per step.
    /// </summary>
    [Fact]
    public void LoadingAFileThatEveryStepRewrites_LeavesOnlyTheCopyTheViewReads()
    {
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        var path = Path.Combine(_dir, "messy.csv");
        var lines = new List<string> { "이름,x,y,constant,sparse,label" };
        for (var i = 0; i < 60; i++)
            lines.Add($"\"행 {i}\",{i % 7}.5,{(i * 3) % 11}.25,same,{(i == 0 ? "1" : "")},{(i % 2 == 0 ? "a" : "b")}");
        lines.Add("\"여러\n줄\",1.5,2.5,same,,a");
        File.WriteAllLines(path, lines, System.Text.Encoding.GetEncoding(949));
        var before = Copies();

        var view = new CsvDataLoader(new MLContext(seed: 0)).LoadData(path, "label", "binary-classification");

        var left = Copies().Except(before).ToList();
        Assert.True(left.Count <= 1, $"{left.Count} copies left: {string.Join(", ", left.Select(Path.GetFileName))}");
        Assert.All(left, copy => Assert.True(ScratchFiles.Owns(copy)));   // and that one goes when the process ends
        Assert.True(view.Preview(maxRows: 5).RowView.Length > 0);          // the view still reads it
    }
}
