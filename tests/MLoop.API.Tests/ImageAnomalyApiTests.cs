using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using SkiaSharp;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /predict</c> for an image-anomaly model: images by content in, a score and a grid of patch distances per image out;
/// an encoder that is not the bank's is a 409, rows sent to it a 400.
/// </summary>
public class ImageAnomalyApiTests : IClassFixture<TestWebApplicationFactory>
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    private static readonly EncoderDefinition Tiny = new(
        "tiny", "1", 4, ImagePreprocessing.ImageNet(8, 6), "pixel_values", "last_hidden_state", EncoderPooling.FirstToken, PatchSize: 3);

    public ImageAnomalyApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private static byte[] Picture(int seed, bool anomalous)
    {
        using var bitmap = new SKBitmap(12, 12);
        for (var y = 0; y < 12; y++)
            for (var x = 0; x < 12; x++)
                bitmap.SetPixel(x, y, anomalous && x is >= 2 and < 9 && y is >= 3 and < 10
                    ? new SKColor(255, 0, 0)
                    : new SKColor((byte)(100 + x * 6 + seed % 5), (byte)(110 + y * 5 + seed % 3), (byte)(120 + (x + y) * 2)));
        using var data = SKImage.FromBitmap(bitmap).Encode(SKEncodedImageFormat.Png, 100);
        return data.ToArray();
    }

    /// <summary>Trains a bank from normal pictures with the tiny patch encoder and seeds it as production, as `mloop train` + promote leave it.</summary>
    private string SeedProductionModel(string modelName)
    {
        new EncoderStore(_factory.EncoderRoot).Add(Tiny, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));
        var pictures = Path.Combine(_factory.TestProjectRoot, "normal-" + modelName);
        Directory.CreateDirectory(pictures);
        for (var i = 0; i < 20; i++)
            File.WriteAllBytes(Path.Combine(pictures, $"n{i:00}.png"), Picture(i, anomalous: false));
        var (artifacts, _) = ImageAnomalyTrainer.Train(pictures, "tiny", null, pool: new EmbedderPool(new EncoderStore(_factory.EncoderRoot)));

        var stagingDir = Path.Combine(_factory.TestProjectRoot, "models", modelName, "staging", "exp-001");
        var productionDir = Path.Combine(_factory.TestProjectRoot, "models", modelName, "production");
        Directory.CreateDirectory(stagingDir);
        Directory.CreateDirectory(productionDir);
        var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        File.WriteAllText(Path.Combine(stagingDir, "config.json"), JsonSerializer.Serialize(new
        {
            dataFile = pictures,
            labelColumn = "",
            timeLimitSeconds = 10,
            metric = "auto",
            testSplit = 0.2,
            inputSchema = new { columns = Array.Empty<object>(), capturedAt = DateTime.UtcNow },
        }, json));
        File.WriteAllText(Path.Combine(stagingDir, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-001",
            timestamp = DateTime.UtcNow,
            status = "Completed",
            task = "image-anomaly",
            result = new { bestTrainer = "PatchBank", trainingTimeSeconds = 1.0 },
        }, json));
        File.WriteAllText(Path.Combine(stagingDir, "metrics.json"), JsonSerializer.Serialize(new Dictionary<string, double> { ["bank_patches"] = artifacts.Bank.Count }, json));
        File.WriteAllBytes(Path.Combine(productionDir, "model.zip"), []);
        artifacts.WriteBeside(Path.Combine(productionDir, "model.zip"));
        File.WriteAllText(Path.Combine(productionDir, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-001",
            promotedAt = DateTime.UtcNow,
            metrics = new Dictionary<string, double>(),
            task = "image-anomaly",
            bestTrainer = "PatchBank",
            labelColumn = "",
        }, json));
        return modelName;
    }

    [Fact]
    public async Task Images_AreScored_WithAGridOverTheCrop_AndAnAnomalousOneScoresHigher()
    {
        var name = SeedProductionModel("anom");

        var response = await _client.PostAsJsonAsync($"/predict?name={name}", new
        {
            images = new[] { Convert.ToBase64String(Picture(3, false)), Convert.ToBase64String(Picture(3, true)) },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var root = doc.RootElement;
        Assert.Equal("image-anomaly", root.GetProperty("task").GetString());
        Assert.Equal(2, root.GetProperty("count").GetInt32());
        Assert.Equal("tiny", root.GetProperty("encoder").GetProperty("id").GetString());
        Assert.Equal(2, root.GetProperty("grid").GetProperty("rows").GetInt32());
        var images = root.GetProperty("images");
        Assert.True(images[1].GetProperty("score").GetDouble() > images[0].GetProperty("score").GetDouble());
        Assert.Equal(4, images[1].GetProperty("distances").GetArrayLength());
        Assert.True(images[1].GetProperty("crop").GetProperty("x1").GetDouble() > images[1].GetProperty("crop").GetProperty("x0").GetDouble());
    }

    [Fact]
    public async Task RowsSentToAnImageAnomalyModel_AreRefused_AndAnUnreadableImageIsNamedByPosition()
    {
        var name = SeedProductionModel("anom-rows");

        var rows = await _client.PostAsJsonAsync($"/predict?name={name}", new[] { new { a = 1 } });
        var garbage = await _client.PostAsJsonAsync($"/predict?name={name}", new { images = new[] { Convert.ToBase64String([1, 2, 3, 4]) } });

        Assert.Equal(HttpStatusCode.BadRequest, rows.StatusCode);
        Assert.Contains("send images by content", await rows.Content.ReadAsStringAsync());
        Assert.Equal(HttpStatusCode.BadRequest, garbage.StatusCode);
    }

    [Fact]
    public async Task AnEncoderThatIsGone_IsA409_NamingIt()
    {
        var name = SeedProductionModel("anom-gone");
        new EncoderStore(_factory.EncoderRoot).Remove("tiny");

        var response = await _client.PostAsJsonAsync($"/predict?name={name}", new { images = new[] { Convert.ToBase64String(Picture(1, false)) } });

        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        var text = await response.Content.ReadAsStringAsync();
        Assert.Contains("tiny", text);
        Assert.Contains("not installed", text);
    }
}
