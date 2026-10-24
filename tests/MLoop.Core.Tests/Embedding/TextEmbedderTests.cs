using System.Text.Json;
using MLoop.Core.Embedding;

namespace MLoop.Core.Tests.Embedding;

/// <summary>
/// A text becomes the same token ids, and the same vector, as the tokenizer encoders are published with computes —
/// checked against ids written by transformers' XLMRobertaTokenizer and vectors written by numpy
/// (Fixtures/embedding/make_text_fixtures.py) on a small SentencePiece model and a four-dimensional encoder graph shaped
/// like a sentence encoder's output.
/// </summary>
public class TextEmbedderTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private static readonly string Tokenizer = Path.Combine(Fixtures, "tiny-tokenizer.model");

    private sealed record Case(string Text, long[] Ids, bool Truncated, float[] Vector);

    private sealed record Reference(int MaxTokens, int Dimension, Case[] Cases);

    private static readonly Lazy<Reference> Expected = new(() => JsonSerializer.Deserialize<Reference>(
        File.ReadAllText(Path.Combine(Fixtures, "text.json")), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!);

    private static EncoderDefinition Tiny => new(
        "tiny-text", "1", Dimension: Expected.Value.Dimension, Preprocessing: null,
        InputName: "input_ids", OutputName: "last_hidden_state", EncoderPooling.MeanOfTokens,
        Kind: EncoderKind.Text, Text: new TextPreprocessing(Expected.Value.MaxTokens));

    [Fact]
    public void EveryText_GetsTheIdsXlmRobertasTokenizerGives_CutAtMaxTokens()
    {
        var preprocessor = TextPreprocessor.Open(Tokenizer, Tiny.Text!);

        Assert.All(Expected.Value.Cases, c =>
        {
            var (ids, truncated) = preprocessor.Encode(c.Text);
            Assert.Equal(c.Ids, ids);
            Assert.Equal(c.Truncated, truncated);
        });
        Assert.Contains(Expected.Value.Cases, c => c.Truncated);   // the cut is exercised, not only the fit
        Assert.Contains(Expected.Value.Cases, c => c.Ids.Contains(3));   // and the unknown piece
    }

    [Fact]
    public void APrefix_GoesBeforeTheText()
    {
        var plain = TextPreprocessor.Open(Tokenizer, Tiny.Text!);
        var prefixed = TextPreprocessor.Open(Tokenizer, Tiny.Text! with { Prefix = "the " });

        Assert.Equal(plain.Encode("the product").Ids, prefixed.Encode("product").Ids);
    }

    [Theory]
    [InlineData("tiny-text-encoder.onnx")]
    [InlineData("tiny-text-encoder-no-types.onnx")]   // a graph without token_type_ids is fed without them
    public void TextsInOneBatch_BecomeTheMeanOfTheirOwnTokens_UnitLength(string graph)
    {
        using var embedder = TextEmbedder.Open(Tiny, Path.Combine(Fixtures, graph), Tokenizer);
        var cases = Expected.Value.Cases;

        var result = embedder.Embed([.. cases.Select(c => c.Text)]);

        // One batch padded to its longest text: a short text's vector leaves the padding out.
        Assert.Equal(cases.Length, result.Vectors.Count);
        for (var i = 0; i < cases.Length; i++)
        {
            for (var d = 0; d < cases[i].Vector.Length; d++)
                Assert.True(Math.Abs(cases[i].Vector[d] - result.Vectors[i][d]) < 1e-5, $"text {i}, dimension {d}");
        }
        Assert.Equal(cases.Count(c => c.Truncated), result.Truncated);
        Assert.Equal(EncoderKind.Text, result.Encoder.Kind);
        Assert.Equal("xlm-roberta/1", result.Encoder.Preprocessor);
    }

    [Fact]
    public void MoreTextsThanOneBatch_AreAllEmbedded_InOrder()
    {
        using var embedder = TextEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-text-encoder.onnx"), Tokenizer);
        var cases = Expected.Value.Cases;
        string[] texts = [.. Enumerable.Range(0, 75).Select(i => cases[i % cases.Length].Text)];

        var result = embedder.Embed(texts);

        Assert.Equal(75, result.Vectors.Count);
        for (var i = 0; i < texts.Length; i++)
            Assert.Equal(cases[i % cases.Length].Vector[0], result.Vectors[i][0], 1e-5f);
    }

    [Fact]
    public void AnImageEncoder_IsNotOpenedForText_NorATextEncoderForImages()
    {
        var image = new EncoderDefinition("img", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "last_hidden_state", EncoderPooling.FirstToken);

        Assert.Throws<ArgumentException>(() => TextEmbedder.Open(image, Path.Combine(Fixtures, "tiny-encoder.onnx"), Tokenizer));
        Assert.Throws<ArgumentException>(() => ImageEmbedder.Open(Tiny, Path.Combine(Fixtures, "tiny-text-encoder.onnx")));
    }

    [Fact]
    public void AGraphWithoutAnAttentionMask_IsNotTheTextEncoderDescribed()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => TextEmbedder.Open(
            Tiny with { InputName = "pixel_values" }, Path.Combine(Fixtures, "tiny-encoder.onnx"), Tokenizer));
        Assert.Contains("attention_mask", ex.Message);
    }

    [Fact]
    public void AFileThatIsNotASentencePieceModel_IsRefusedAsATokenizer()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => TextPreprocessor.Open(Path.Combine(Fixtures, "text.json"), Tiny.Text!));
        Assert.Contains("SentencePiece", ex.Message);
    }

    [Fact]
    public void ADefinition_ThatMixesTheKinds_CannotRun()
    {
        Assert.Null(Tiny.Problem());
        Assert.Contains("text", (Tiny with { Text = null }).Problem());
        Assert.NotNull((Tiny with { Preprocessing = ImagePreprocessing.ImageNet(8, 6) }).Problem());
        Assert.NotNull((Tiny with { Text = new TextPreprocessing(MaxTokens: 2) }).Problem());
        var image = new EncoderDefinition("img", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "out", EncoderPooling.MeanOfTokens);
        Assert.Contains("attention mask", image.Problem());
    }
}
