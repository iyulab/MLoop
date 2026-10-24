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

    // An install downloads a catalog entry's pinned file and refuses any other.
    private sealed class Serve(byte[] body) : HttpMessageHandler
    {
        public string? Asked { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Asked = request.RequestUri!.ToString();
            return Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new ByteArrayContent(body) });
        }
    }

    private static string Sha256(byte[] bytes) => Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes));

    [Fact]
    public async Task AnInstall_DownloadsThePinnedFile_AndRecordsItAsAnyAddedEncoder()
    {
        var bytes = File.ReadAllBytes(Graph);
        var entry = Tiny with { Url = "https://example.test/tiny.onnx", Sha256 = Sha256(bytes) };
        var handler = new Serve(bytes);
        var reported = new List<long>();
        var store = new EncoderStore(_root);

        var installed = await store.InstallAsync(entry, new HttpClient(handler), new SyncProgress(p => reported.Add(p.Downloaded)));

        Assert.Equal(entry.Url, handler.Asked);
        Assert.Equal(entry.Sha256, installed.Definition.Sha256);
        Assert.Equal(bytes, File.ReadAllBytes(installed.ModelPath));
        Assert.Equal(bytes.Length, reported[^1]);
        Assert.Equal(["tiny"], Directory.GetFileSystemEntries(_root).Select(Path.GetFileName));   // no download left beside it
    }

    [Fact]
    public async Task ADownloadWithAnotherSha256_IsRefused_AndLeavesTheInstalledEncoderAlone()
    {
        var bytes = File.ReadAllBytes(Graph);
        var store = new EncoderStore(_root);
        var before = store.Add(Tiny, Graph);
        var entry = Tiny with { Url = "https://example.test/tiny.onnx", Sha256 = new string('0', 64) };

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => store.InstallAsync(entry, new HttpClient(new Serve(bytes))));

        Assert.Contains("nothing was installed", refused.Message);
        Assert.Equal(before.Definition, store.Find("tiny")!.Definition);
        Assert.Equal(["tiny"], Directory.GetFileSystemEntries(_root).Select(Path.GetFileName));
    }

    [Fact]
    public async Task AnEntryWithoutAPinnedDownload_CannotBeInstalled()
        => await Assert.ThrowsAsync<ArgumentException>(() => new EncoderStore(_root).InstallAsync(Tiny, new HttpClient(new Serve([]))));

    [Fact]
    public void EveryCatalogEntry_CanRun_AndPinsAnHttpsDownload()
    {
        Assert.NotEmpty(EncoderCatalog.All);
        Assert.All(EncoderCatalog.All, e =>
        {
            Assert.Null(e.Problem());
            Assert.StartsWith("https://", e.Url);
            Assert.Matches("^[0-9a-f]{64}$", e.Sha256!);
        });
        Assert.Same(EncoderCatalog.DinoV2Small, EncoderCatalog.Find("DINOv2-small"));
    }

    [Fact]
    public void AMissingCatalogEncoder_IsAnsweredWithItsInstallCommand()
    {
        var store = new EncoderStore(_root);

        Assert.Contains("mloop encoder install dinov2-small", store.Resolve("dinov2-small").Refusal);
        Assert.Contains("mloop encoder install", store.Resolve(null).Refusal);
        Assert.Contains("mloop encoder add", store.Resolve("mine").Refusal);
    }

    // Progress<T> posts to a context; a test reads what was reported straight away.
    /// <summary>
    /// An install made before the catalog declared patch sizes has no <c>patchSize</c> in its encoder.json; its weights
    /// are the catalog entry's, so the entry's patch size applies to it — another definition keeps what it says.
    /// </summary>
    [Fact]
    public void AnInstallOfCatalogWeights_WithoutAPatchSize_TakesTheCatalogs()
    {
        var store = new EncoderStore(_root);
        var dir = Directory.CreateDirectory(Path.Combine(_root, "dinov2-small")).FullName;
        var old = EncoderCatalog.DinoV2Small with { PatchSize = null };
        File.WriteAllText(Path.Combine(dir, EncoderStore.DefinitionFileName),
            System.Text.Json.JsonSerializer.Serialize(old, new System.Text.Json.JsonSerializerOptions
            {
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
                Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter(System.Text.Json.JsonNamingPolicy.KebabCaseLower) },
            }));
        File.Copy(Graph, Path.Combine(dir, EncoderStore.ModelFileName));
        store.Add(Tiny, Graph);

        Assert.DoesNotContain("patchSize", File.ReadAllText(Path.Combine(dir, EncoderStore.DefinitionFileName)));
        Assert.Equal(14, store.Find("dinov2-small")!.Definition.PatchSize);
        Assert.Null(store.Find("tiny")!.Definition.PatchSize);
    }

    private sealed class SyncProgress(Action<(long Downloaded, long? Total)> report) : IProgress<(long Downloaded, long? Total)>
    {
        public void Report((long Downloaded, long? Total) value) => report(value);
    }

}
