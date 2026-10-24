using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using MLoop.Core.Embedding;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /embed</c> — image contents in (base64 JSON or multipart files), unit vectors out with the encoder's
/// identity; an unreadable image named by position, a missing encoder 409, too many images 413.
/// </summary>
public class EmbedApiTests : IClassFixture<TestWebApplicationFactory>
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public EmbedApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private void InstallTiny() => new EncoderStore(_factory.EncoderRoot).Add(
        new EncoderDefinition("tiny", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "last_hidden_state", EncoderPooling.FirstToken,
            PatchSize: 3),
        Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));

    private void InstallWithoutPatches() => new EncoderStore(_factory.EncoderRoot).Add(
        new EncoderDefinition("plain", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "last_hidden_state", EncoderPooling.FirstToken),
        Path.Combine(Fixtures, "tiny-encoder.onnx"));

    private static string Pattern => Convert.ToBase64String(File.ReadAllBytes(Path.Combine(Fixtures, "pattern.png")));

    private static float[] Expected() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "expected.json")))
        .RootElement.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray();

    [Fact]
    public async Task Base64Images_BecomeUnitVectors_WithTheEncodersIdentity()
    {
        InstallTiny();

        var response = await _client.PostAsJsonAsync("/embed?encoder=tiny", new { images = new[] { Pattern, Pattern } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var encoder = doc.RootElement.GetProperty("encoder");
        Assert.Equal("tiny", encoder.GetProperty("id").GetString());
        Assert.Equal(64, encoder.GetProperty("sha256").GetString()!.Length);
        Assert.Equal(4, encoder.GetProperty("dimension").GetInt32());
        Assert.True(encoder.GetProperty("normalized").GetBoolean());
        Assert.Equal(ImagePreprocessor.Revision, encoder.GetProperty("preprocessor").GetString());
        Assert.False(doc.RootElement.TryGetProperty("patches", out _));
        var vectors = doc.RootElement.GetProperty("vectors");
        Assert.Equal(2, vectors.GetArrayLength());
        var expected = Expected();
        var first = vectors[0].EnumerateArray().Select(v => v.GetSingle()).ToArray();
        for (var d = 0; d < 4; d++)
            Assert.Equal(expected[d], first[d], 4);
    }

    [Fact]
    public async Task MultipartFiles_AreAcceptedToo()
    {
        InstallTiny();
        using var form = new MultipartFormDataContent();
        var file = new ByteArrayContent(File.ReadAllBytes(Path.Combine(Fixtures, "pattern.png")));
        file.Headers.ContentType = new MediaTypeHeaderValue("image/png");
        form.Add(file, "image", "pattern.png");

        var response = await _client.PostAsync("/embed?encoder=tiny", form);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(1, doc.RootElement.GetProperty("vectors").GetArrayLength());
    }

    [Fact]
    public async Task AnUnreadableImage_Is400_NamingItsPosition()
    {
        InstallTiny();

        var response = await _client.PostAsJsonAsync("/embed?encoder=tiny", new { images = new[] { Pattern, Convert.ToBase64String("not an image"u8.ToArray()) } });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(1, doc.RootElement.GetProperty("index").GetInt32());
    }

    [Fact]
    public async Task AnEncoderThatIsNotInstalled_Is409_WithHowToAddOne()
    {
        var response = await _client.PostAsJsonAsync("/embed?encoder=missing", new { images = new[] { Pattern } });

        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        Assert.Contains("mloop encoder add", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task MoreImagesThanOneRequestTakes_Is413()
    {
        var response = await _client.PostAsJsonAsync("/embed", new { images = Enumerable.Repeat(Pattern, EmbedderPool.MaxImages + 1) });

        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
    }

    [Fact]
    public async Task Patches_AreGivenPerImage_WithTheGridAndTheCrop()
    {
        InstallTiny();

        var response = await _client.PostAsJsonAsync("/embed?encoder=tiny&patches=true", new { images = new[] { Pattern, Pattern } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var grid = doc.RootElement.GetProperty("grid");
        Assert.Equal(2, grid.GetProperty("rows").GetInt32());
        Assert.Equal(2, grid.GetProperty("columns").GetInt32());
        Assert.Equal(3, grid.GetProperty("patchSize").GetInt32());
        var patches = doc.RootElement.GetProperty("patches");
        Assert.Equal(2, patches.GetArrayLength());
        Assert.Equal(4, patches[0].GetProperty("vectors").GetArrayLength());
        Assert.Equal(9.25, patches[0].GetProperty("crop").GetProperty("x0").GetDouble(), 9);
        Assert.Equal(2, doc.RootElement.GetProperty("vectors").GetArrayLength());
    }

    [Fact]
    public async Task PatchesOfAnEncoderWithoutAGrid_Are409()
    {
        InstallWithoutPatches();

        var response = await _client.PostAsJsonAsync("/embed?encoder=plain&patches=true", new { images = new[] { Pattern } });

        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        Assert.Contains("patchSize", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task MorePatchImagesThanOneRequestTakes_Is413()
    {
        var response = await _client.PostAsJsonAsync("/embed?patches=true", new { images = Enumerable.Repeat(Pattern, EmbedderPool.MaxPatchImages + 1) });

        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
    }
}
