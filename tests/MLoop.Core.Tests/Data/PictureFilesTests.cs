using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public sealed class PictureFilesTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("mloop-pictures-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void AFileThatIsNotAPictureIsNamed()
    {
        var good = Path.Combine(_root, "good.png");
        using (var bitmap = new SkiaSharp.SKBitmap(4, 4))
        using (var data = bitmap.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100))
            File.WriteAllBytes(good, data.ToArray());
        var broken = Path.Combine(_root, "broken.png");
        File.WriteAllBytes(broken, [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4]);

        Assert.Null(PictureFiles.Unreadable([good]));
        var refusal = PictureFiles.Unreadable([good, broken])!;

        Assert.Contains(broken, refusal, StringComparison.Ordinal);
        Assert.DoesNotContain(good, refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void ManyAreCountedAndTheFirstFiveNamed()
    {
        var files = Enumerable.Range(0, 7).Select(i => Path.Combine(_root, $"{i}.png")).ToList();
        files.ForEach(f => File.WriteAllText(f, "text"));

        var refusal = PictureFiles.Unreadable(files)!;

        Assert.StartsWith("7 pictures cannot be read:", refusal, StringComparison.Ordinal);
        Assert.Contains("and 2 more", refusal, StringComparison.Ordinal);
    }
}
