using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// A model that reads pictures answers pictures sent to <c>POST /predict</c> by their contents only. A row names a column of
/// the model's input, and for these models that column is a file path: accepting one has the server open a file the caller
/// names. The refusal comes before the model is read, so a placeholder model file is enough here.
/// </summary>
public class ImagePathRowApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ImagePathRowApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private string SeedModel(string task)
    {
        var modelName = "paths-" + task;
        var modelDir = Path.Combine(_factory.TestProjectRoot, "models", modelName);
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

        var production = Path.Combine(modelDir, "production");
        Directory.CreateDirectory(production);
        File.WriteAllBytes(Path.Combine(production, "model.zip"), [0]);
        File.WriteAllText(Path.Combine(production, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-001",
            promotedAt = DateTime.UtcNow,
            task,
            labelColumn = "Label",
        }, options));

        // A candidate in staging, for ?experiment=.
        var candidate = Path.Combine(modelDir, "staging", "exp-002");
        Directory.CreateDirectory(candidate);
        File.WriteAllBytes(Path.Combine(candidate, "model.zip"), [0]);
        File.WriteAllText(Path.Combine(candidate, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-002",
            timestamp = DateTime.UtcNow,
            status = "Completed",
            task,
        }, options));
        File.WriteAllText(Path.Combine(candidate, "config.json"), JsonSerializer.Serialize(new
        {
            dataFile = "datasets/images",
            labelColumn = "Label",
            timeLimitSeconds = 0,
            metric = "accuracy",
            testSplit = 0.2,
        }, options));

        return modelName;
    }

    [Theory]
    [InlineData("image-classification", "")]
    [InlineData("image-classification", "&experiment=exp-002")]
    [InlineData("object-detection", "")]
    [InlineData("object-detection", "&experiment=exp-002")]
    [InlineData("image-anomaly", "")]
    [InlineData("image-anomaly", "&experiment=exp-002")]
    public async Task A_row_naming_a_file_on_the_server_is_refused_and_not_logged(string task, string query)
    {
        var modelName = SeedModel(task);
        var onTheServer = Path.Combine(_factory.TestProjectRoot, "datasets", "on-the-server.png");
        File.WriteAllBytes(onTheServer, [1, 2, 3]);

        foreach (var body in new object[] { new { ImagePath = onTheServer }, new[] { new { ImagePath = onTheServer } } })
        {
            var response = await _client.PostAsJsonAsync($"/predict?name={modelName}{query}", body);

            var text = await response.Content.ReadAsStringAsync();
            response.StatusCode.Should().Be(HttpStatusCode.BadRequest, text);
            text.Should().Contain(task).And.Contain("send images by content").And.NotContain("on-the-server");
        }

        Directory.Exists(Path.Combine(_factory.TestProjectRoot, ".mloop", "logs", modelName)).Should().BeFalse();
    }
}
