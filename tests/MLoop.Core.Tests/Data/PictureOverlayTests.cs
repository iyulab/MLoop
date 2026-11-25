using MLoop.Core.Data;
using MLoop.Core.Embedding;
using SkiaSharp;

namespace MLoop.Core.Tests.Data;

public sealed class PictureOverlayTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("mloop-overlay-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static readonly SKColor Grey = new(180, 180, 180);

    private string Picture(string name, int width = 160, int height = 120)
    {
        var path = Path.Combine(_root, "in", name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var bitmap = new SKBitmap(width, height);
        using (var canvas = new SKCanvas(bitmap))
            canvas.Clear(Grey);
        using var data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(path, data.ToArray());
        return path;
    }

    [Fact]
    public void CopiesKeepTheirPlaceUnderTheFolder_AndTwoPicturesOfOneNameKeepBoth()
    {
        var read = Path.Combine(_root, "in");
        string[] sources = [Path.Combine(read, "a.jpg"), Path.Combine(read, "a.png"), Path.Combine(read, "line2", "b.bmp")];

        var targets = PictureOverlay.Targets(sources, read, Path.Combine(_root, "out"));

        Assert.Equal(
            ["a.jpg.png", "a.png.png", Path.Combine("line2", "b.png")],
            targets.Select(t => Path.GetRelativePath(Path.Combine(_root, "out"), t)).ToArray());
    }

    [Fact]
    public void TheFolderBeingReadAndAnOriginalsOwnPlaceAreRefused()
    {
        var read = Path.Combine(_root, "in");
        var picture = Picture("a.png");

        Assert.Contains("inside the folder of pictures being read",
            PictureOverlay.TargetRefusal([picture], read, read, Path.Combine(read, "out")), StringComparison.Ordinal);
        // One picture, not a folder: beside it is fine unless the copy would take its place.
        Assert.Contains("over the picture", PictureOverlay.TargetRefusal([picture], null, read, read), StringComparison.Ordinal);
        Assert.Null(PictureOverlay.TargetRefusal([picture], null, read, Path.Combine(read, "out")));
        Assert.Null(PictureOverlay.TargetRefusal([picture], read, read, Path.Combine(_root, "out")));
    }

    [Fact]
    public void ABoxIsOutlinedWhereItIs_AndTheOriginalIsNotChanged()
    {
        var source = Picture("a.png");
        var before = File.ReadAllBytes(source);
        var target = Path.Combine(_root, "out", "a.png");

        PictureOverlay.SaveBoxes(source, [new DrawnBox("red", 0.88f, 40, 40, 100, 90)], target);

        Assert.Equal(before, File.ReadAllBytes(source));
        using var drawn = SKBitmap.Decode(target);
        Assert.Equal((160, 120), (drawn.Width, drawn.Height));
        Assert.NotEqual(Grey, drawn.GetPixel(40, 65)); // the box's left edge
        Assert.NotEqual(Grey, drawn.GetPixel(100, 65)); // and its right
        Assert.Equal(Grey, drawn.GetPixel(70, 65)); // the inside is left as it was
        Assert.Equal(Grey, drawn.GetPixel(140, 110)); // and so is the rest
    }

    [Fact]
    public void AHeatMapTintsOnlyWhatIsBeyondNormal_AndOutlinesTheStrongestPatch()
    {
        var source = Picture("a.png", 160, 160);
        var target = Path.Combine(_root, "out", "a.png");
        // A 4 x 4 grid over the whole picture: patches of 40 pixels. One far beyond where normal ended, one a little.
        var distances = Enumerable.Repeat(0.05f, 16).ToArray();
        distances[2 * 4 + 1] = 0.60f;
        distances[3 * 4 + 3] = 0.15f;

        PictureOverlay.SaveHeatMap(source, distances, 4, 4, new ImageRegion(0, 0, 160, 160), 0.10, target);

        using var drawn = SKBitmap.Decode(target);
        var strong = drawn.GetPixel(60, 100);
        var weak = drawn.GetPixel(140, 140);
        Assert.True(strong.Red > strong.Blue + 80, $"the strongest patch is not red: {strong}");
        Assert.True(weak.Red > weak.Blue && weak.Red - weak.Blue < strong.Red - strong.Blue, $"the weaker patch: {weak}");
        Assert.Equal(Grey, drawn.GetPixel(100, 60)); // within normal: clear
        Assert.NotEqual(Grey, drawn.GetPixel(40, 100)); // the outline of the strongest patch
    }

    [Fact]
    public void APictureWithinNormalStaysClearButForItsScore()
    {
        var source = Picture("a.png", 160, 160);
        var target = Path.Combine(_root, "out", "a.png");
        var distances = Enumerable.Range(0, 16).Select(i => 0.05f + i * 0.001f).ToArray();

        PictureOverlay.SaveHeatMap(source, distances, 4, 4, new ImageRegion(0, 0, 160, 160), 0.10, target);

        using var drawn = SKBitmap.Decode(target);
        Assert.Equal(Grey, drawn.GetPixel(60, 60));
        Assert.Equal(Grey, drawn.GetPixel(120, 140)); // the edge of the strongest patch: not outlined
        Assert.NotEqual(Grey, drawn.GetPixel(3, 3)); // the score's tag
    }

    [Fact]
    public void ALabelOutsideTheBuiltInLettersIsStillDrawn()
    {
        var source = Picture("a.png");
        var target = Path.Combine(_root, "out", "a.png");

        PictureOverlay.SaveBoxes(source, [new DrawnBox("결함", 0.5f, 20, 40, 80, 90)], target);

        using var drawn = SKBitmap.Decode(target);
        Assert.NotEqual(Grey, drawn.GetPixel(22, 33)); // the tag above the box
    }

    [Fact]
    public void TheBuiltInFontHasEveryPrintableCharacter()
    {
        var all = new string([.. Enumerable.Range(' ', '~' - ' ' + 1).Select(c => (char)c)]);
        var source = Picture("a.png", 800, 200);

        PictureOverlay.SaveBoxes(source, [new DrawnBox(all, 1f, 10, 100, 700, 150)], Path.Combine(_root, "out", "a.png"));
    }
}
