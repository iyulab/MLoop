using System.Text.Json;
using MLoop.Core.Embedding;

namespace MLoop.Core.Tests.Embedding;

/// <summary>
/// An image becomes the same encoder input, and the same vector, as the processors encoders are published with compute
/// — checked against values written by Pillow and numpy (Fixtures/embedding/make_fixtures.py) on a synthetic image and
/// a four-dimensional encoder graph shaped like a vision transformer's output.
/// </summary>
public class ImageEmbedderTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");

    private static readonly EncoderDefinition Tiny = new(
        "tiny", "1", Dimension: 4, ImagePreprocessing.ImageNet(shortestEdge: 8, cropSize: 6),
        InputName: "pixel_values", OutputName: "last_hidden_state", EncoderPooling.FirstToken);

    private sealed record Expected(int[] Resized, float[] Tensor, float[] Embedding);

    private static readonly Lazy<Expected> Reference = new(() => JsonSerializer.Deserialize<Expected>(
        File.ReadAllText(Path.Combine(Fixtures, "expected.json")), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!);

    private static byte[] Pattern => File.ReadAllBytes(Path.Combine(Fixtures, "pattern.png"));

    [Fact]
    public void TheResizedSize_KeepsTheShortestEdge_AndTruncatesTheOther()
    {
        Assert.Equal((12, 8), ImagePreprocessor.ResizedSize(37, 23, 8));
        Assert.Equal((256, 341), ImagePreprocessor.ResizedSize(480, 640, 256));
        Assert.Equal(Reference.Value.Resized[0], ImagePreprocessor.ResizedSize(37, 23, 8).Width);
    }

    /// <summary>
    /// Bicubic resampling in Pillow's fixed point with its rounding between passes: a level off anywhere is 1/255 ÷ std ≈
    /// 0.017 in the tensor, so agreement within 1e-5 means every resampled pixel matches.
    /// </summary>
    [Fact]
    public void ThePreprocessedTensor_IsWhatPillowComputes()
    {
        var tensor = ImagePreprocessor.ToTensor(Pattern, Tiny.Preprocessing!);

        Assert.Equal(Reference.Value.Tensor.Length, tensor.Length);
        for (var i = 0; i < tensor.Length; i++)
            Assert.True(Math.Abs(Reference.Value.Tensor[i] - tensor[i]) < 1e-5, $"element {i}: Pillow {Reference.Value.Tensor[i]}, here {tensor[i]}");
    }

    /// <summary>
    /// An image with transparency: converting to RGB drops the alpha and keeps the colour stored under it. A decoder that
    /// premultiplies darkens every translucent pixel by its own alpha (and blackens a transparent one), and the vector
    /// moves — measured at cosine 0.965 against the encoder's own processor on a screenshot with a translucent layer.
    /// </summary>
    [Fact]
    public void ATranslucentImage_IsReadAsItsStoredColours_AsConvertingToRgbDoes()
    {
        var expected = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "translucent.json"))).RootElement
            .GetProperty("tensor").EnumerateArray().Select(e => e.GetSingle()).ToArray();

        var tensor = ImagePreprocessor.ToTensor(File.ReadAllBytes(Path.Combine(Fixtures, "translucent.png")), Tiny.Preprocessing!);

        Assert.Equal(expected.Length, tensor.Length);
        for (var i = 0; i < tensor.Length; i++)
            Assert.True(Math.Abs(expected[i] - tensor[i]) < 1e-5, $"element {i}: Pillow {expected[i]}, here {tensor[i]}");
    }

    [Fact]
    public void AnImage_BecomesTheEncodersUnitVector_FromTheFirstToken()
    {
        using var embedder = ImageEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-encoder.onnx"));

        var vectors = embedder.Embed([Pattern, Pattern]).Vectors;

        Assert.Equal(2, vectors.Count);
        Assert.All(vectors, v =>
        {
            Assert.Equal(4, v.Length);
            Assert.Equal(1.0, Math.Sqrt(v.Sum(x => (double)x * x)), 5);
            for (var d = 0; d < 4; d++)
                Assert.Equal(Reference.Value.Embedding[d], v[d], 4);
        });
    }

    [Fact]
    public void AnUndecodableImage_IsNamedByItsPosition()
    {
        using var embedder = ImageEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-encoder.onnx"));

        var ex = Assert.Throws<UnreadableImageException>(() => embedder.Embed([Pattern, "not an image"u8.ToArray()]));

        Assert.Equal(1, ex.Index);
    }

    [Fact]
    public void AGraphWithoutTheDescribedOutput_IsRefusedWhenOpened()
    {
        var other = Tiny with { OutputName = "image_embeds" };

        var ex = Assert.Throws<InvalidOperationException>(() => ImageEmbedder.Open(other, Path.Combine(Fixtures, "tiny-encoder.onnx")));

        Assert.Contains("image_embeds", ex.Message);
    }

    [Fact]
    public void AnOutputOfAnotherWidth_IsRefused_NotTruncated()
    {
        using var embedder = ImageEmbedder.Open(Tiny with { Dimension = 384 }, Path.Combine(Fixtures, "tiny-encoder.onnx"));

        var ex = Assert.Throws<InvalidOperationException>(() => embedder.Embed([Pattern]));

        Assert.Contains("384", ex.Message);
    }

    private static readonly EncoderDefinition TinyPatches = Tiny with { PatchSize = 3 };

    private sealed record ExpectedPatches(float[][] Patches, double[] Crop);

    private static ExpectedPatches PatchReference => JsonSerializer.Deserialize<ExpectedPatches>(
        File.ReadAllText(Path.Combine(Fixtures, "patches.json")), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    /// <summary>
    /// The graph puts a class token and a register token before its 2 × 2 patches: the patches are the last four tokens,
    /// row by row, each a unit vector, and the grid covers the centre crop — given in the image's own pixels.
    /// </summary>
    [Fact]
    public void Patches_AreTheLastTokens_RowByRow_OverTheCentreCrop()
    {
        using var embedder = ImageEmbedder.Open(TinyPatches, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));

        var result = embedder.Embed([Pattern, Pattern], patches: true);

        Assert.Equal(new PatchGrid(2, 2, 3, 6), result.Grid);
        var reference = PatchReference;
        Assert.Equal(2, result.Patches!.Count);
        Assert.All(result.Patches, image =>
        {
            Assert.Equal(new ImageRegion(reference.Crop[0], reference.Crop[1], reference.Crop[2], reference.Crop[3]), image.Crop);
            Assert.Equal(4, image.Vectors.Count);
            for (var t = 0; t < 4; t++)
                for (var d = 0; d < 4; d++)
                    Assert.Equal(reference.Patches[t][d], image.Vectors[t][d], 4);
        });
        for (var d = 0; d < 4; d++)
            Assert.Equal(Reference.Value.Embedding[d], result.Vectors[0][d], 4);   // the class token, as without patches
    }

    [Fact]
    public void WithoutPatches_TheAnswerCarriesNoGrid()
    {
        using var embedder = ImageEmbedder.Open(TinyPatches, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));

        var result = embedder.Embed([Pattern]);

        Assert.Null(result.Grid);
        Assert.Null(result.Patches);
    }

    [Fact]
    public void PatchesOfAnEncoderThatDeclaresNoPatchSize_AreRefused()
    {
        using var embedder = ImageEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-encoder.onnx"));

        var ex = Assert.Throws<InvalidOperationException>(() => embedder.Embed([Pattern], patches: true));

        Assert.Contains("patchSize", ex.Message);
    }

    [Fact]
    public void AGridLargerThanTheOutput_IsRefused_NotReadFromTheClassToken()
    {
        using var embedder = ImageEmbedder.Open(Tiny with { PatchSize = 2 }, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));

        var ex = Assert.Throws<InvalidOperationException>(() => embedder.Embed([Pattern], patches: true));

        Assert.Contains("6 tokens", ex.Message);
    }

    [Theory]
    [InlineData(4, "does not tile")]
    [InlineData(0, "does not tile")]
    public void APatchSizeThatDoesNotTileTheCrop_CannotRun(int patch, string reason) =>
        Assert.Contains(reason, (Tiny with { PatchSize = patch }).Problem());

    [Fact]
    public void TheCropRegion_ScalesTheCentreCropBackToTheImage()
    {
        // 480 × 640 → 256 × 341; the 224 crop starts at (16, 58) of the resized image.
        var region = ImagePreprocessor.CropRegion(480, 640, ImagePreprocessing.ImageNet(256, 224));

        Assert.Equal(16 * 480 / 256.0, region.X0, 9);
        Assert.Equal(58 * 640 / 341.0, region.Y0, 9);
        Assert.Equal(240 * 480 / 256.0, region.X1, 9);
        Assert.Equal(282 * 640 / 341.0, region.Y1, 9);
    }

    /// <summary>
    /// The preprocessor's revision is part of every vector's identity, so the tensors it makes are pinned to it: when one
    /// changes, this fails until <see cref="ImagePreprocessor.Revision"/> is raised and the digest below replaced. Rounded
    /// to 1e-3 — one 8-bit level is 0.017 in the tensor, so any change of a pixel shows.
    /// </summary>
    [Fact]
    public void TheTensorsThePreprocessorMakes_ArePinnedToItsRevision()
    {
        var digests = new Dictionary<string, string>
        {
            ["pillow/2"] = "75743d96d6ad5373d0dc048f33f617fa0ebbfe1782ca754db65c7ebd35ca8cf2",
        };

        var tensors = new[] { "pattern.png", "translucent.png" }
            .SelectMany(f => ImagePreprocessor.ToTensor(File.ReadAllBytes(Path.Combine(Fixtures, f)), Tiny.Preprocessing!))
            .Select(v => Math.Round(v, 3).ToString("0.000", System.Globalization.CultureInfo.InvariantCulture));
        var digest = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.ASCII.GetBytes(string.Join(",", tensors))));

        Assert.True(digests.TryGetValue(ImagePreprocessor.Revision, out var pinned), $"no digest pinned for {ImagePreprocessor.Revision}");
        Assert.True(pinned == digest, $"The tensors changed under revision {ImagePreprocessor.Revision} (digest {digest}): raise ImagePreprocessor.Revision and pin the new digest.");
    }

    [Fact]
    public void AVectorsIdentity_NamesThePreprocessorRevision()
    {
        Assert.Equal(ImagePreprocessor.Revision, EncoderIdentity.Of(Tiny).Preprocessor);
    }
}
