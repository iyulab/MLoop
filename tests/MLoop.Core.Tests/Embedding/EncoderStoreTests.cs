using MLoop.Core.Embedding;

namespace MLoop.Core.Tests.Embedding;

/// <summary>
/// The encoders installed on a machine: added from a definition and a graph with the graph's SHA-256 recorded (the
/// identity of the vectors), found by id or as the only one, and refused when they could not run.
/// </summary>
public sealed class EncoderStoreTests : IDisposable
{
    private static readonly string Graph = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding", "tiny-encoder.onnx");
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-encoders-{Guid.NewGuid():N}");

    private static readonly EncoderDefinition Tiny = new(
        "tiny", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "last_hidden_state", EncoderPooling.FirstToken);

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void AnAddedEncoder_IsListedAndFound_WithItsGraphsSha256()
    {
        var store = new EncoderStore(_root);

        var added = store.Add(Tiny, Graph);

        Assert.Equal(64, added.Definition.Sha256!.Length);
        var listed = Assert.Single(store.List());
        Assert.Equal(added.Definition, listed.Definition);
        Assert.Equal(added.Definition, store.Find("tiny")!.Definition);
        Assert.True(File.Exists(listed.ModelPath));
    }

    [Fact]
    public void TheDefinitionFile_RoundTrips_WithThePoolingSpelledAsWords()
    {
        var store = new EncoderStore(_root);
        store.Add(Tiny, Graph);
        var written = Path.Combine(_root, "tiny", EncoderStore.DefinitionFileName);

        Assert.Contains("\"first-token\"", File.ReadAllText(written));
        Assert.Equal(EncoderPooling.FirstToken, EncoderStore.ReadDefinition(written).Pooling);
    }

    [Fact]
    public void ARequestWithoutAnEncoder_GetsTheOnlyOne_AndIsToldWhenThereAreSeveral()
    {
        var store = new EncoderStore(_root);
        Assert.Contains("No encoder is installed", store.Resolve(null).Refusal);

        store.Add(Tiny, Graph);
        Assert.Equal("tiny", store.Resolve(null).Encoder!.Definition.Id);

        store.Add(Tiny with { Id = "tiny-b" }, Graph);
        Assert.Contains("Several encoders", store.Resolve(null).Refusal);
        Assert.Equal("tiny-b", store.Resolve("tiny-b").Encoder!.Definition.Id);
        Assert.Contains("'other' is not installed", store.Resolve("other").Refusal);
    }

    [Fact]
    public void APinnedSha256_TheGraphDoesNotHave_IsRefused()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => new EncoderStore(_root).Add(Tiny with { Sha256 = new string('0', 64) }, Graph));

        Assert.Contains("pins", ex.Message);
        Assert.Empty(new EncoderStore(_root).List());
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("a b")]
    [InlineData("..")]
    public void AnIdThatIsNotADirectoryName_IsRefused(string id) =>
        Assert.Throws<ArgumentException>(() => new EncoderStore(_root).Add(Tiny with { Id = id }, Graph));

    [Fact]
    public void ADefinitionThatCannotRun_IsRefusedWhenAdded_NotWhenFirstUsed()
    {
        var cropsTooMuch = Tiny with { Preprocessing = ImagePreprocessing.ImageNet(shortestEdge: 6, cropSize: 8) };

        var ex = Assert.Throws<ArgumentException>(() => new EncoderStore(_root).Add(cropsTooMuch, Graph));

        Assert.Contains("crops 8", ex.Message);
    }

    [Fact]
    public void ARemovedEncoder_IsGone()
    {
        var store = new EncoderStore(_root);
        store.Add(Tiny, Graph);

        Assert.True(store.Remove("tiny"));
        Assert.False(store.Remove("tiny"));
        Assert.Empty(store.List());
    }
}
