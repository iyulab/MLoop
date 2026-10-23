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
        var tensor = ImagePreprocessor.ToTensor(Pattern, Tiny.Preprocessing);

        Assert.Equal(Reference.Value.Tensor.Length, tensor.Length);
        for (var i = 0; i < tensor.Length; i++)
            Assert.True(Math.Abs(Reference.Value.Tensor[i] - tensor[i]) < 1e-5, $"element {i}: Pillow {Reference.Value.Tensor[i]}, here {tensor[i]}");
    }

    [Fact]
    public void AnImage_BecomesTheEncodersUnitVector_FromTheFirstToken()
    {
        using var embedder = ImageEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-encoder.onnx"));

        var vectors = embedder.Embed([Pattern, Pattern]);

        Assert.Equal(2, vectors.Length);
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
}
