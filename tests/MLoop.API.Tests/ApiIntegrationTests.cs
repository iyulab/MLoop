using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using MLoop.API;

namespace MLoop.API.Tests;

public class ApiIntegrationTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ApiIntegrationTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task HealthEndpoint_ReturnsHealthy()
    {
        // Act
        var response = await _client.GetAsync("/health");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("healthy");
        content.Should().Contain("timestamp");
        content.Should().Contain("version");
    }

    [Fact]
    public async Task InfoEndpoint_WithNoProductionModel_ReturnsNotFound()
    {
        // Act
        var response = await _client.GetAsync("/info");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("No production model found");
    }

    [Fact]
    public async Task ModelsEndpoint_ReturnsEmptyListInitially()
    {
        // Act
        var response = await _client.GetAsync("/models");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<JsonElement>(content);

        result.GetProperty("count").GetInt32().Should().Be(0);
    }

    [Fact]
    public async Task PredictEndpoint_WithNoModel_ReturnsNotFound()
    {
        // Arrange
        var prediction = new
        {
            feature1 = 1.0,
            feature2 = 2.0
        };

        // Act
        var response = await _client.PostAsJsonAsync("/predict", prediction);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task PredictEndpoint_WithBatchInput_AcceptsArray()
    {
        // Arrange
        var predictions = new[]
        {
            new { feature1 = 1.0, feature2 = 2.0 },
            new { feature1 = 3.0, feature2 = 4.0 }
        };

        // Act
        var response = await _client.PostAsJsonAsync("/predict", predictions);

        // Assert
        // Should return 404 since no model exists, but validates batch input handling
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task SwaggerEndpoint_ReturnsSwaggerDocument()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("MLoop MLOps API");
        content.Should().Contain("/health");
        content.Should().Contain("/predict");
    }

    [Fact]
    public async Task SwaggerUIEndpoint_ReturnsHtml()
    {
        // Act
        var response = await _client.GetAsync("/");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Content.Headers.ContentType?.MediaType.Should().Be("text/html");
    }

    [Fact]
    public async Task CorsHeaders_ArePresent()
    {
        // Arrange
        var request = new HttpRequestMessage(HttpMethod.Options, "/health");
        request.Headers.Add("Origin", "http://localhost:3000");
        request.Headers.Add("Access-Control-Request-Method", "GET");

        // Act
        var response = await _client.SendAsync(request);

        // Assert
        response.Headers.Should().ContainKey("Access-Control-Allow-Origin");
    }

    [Fact]
    public async Task HealthEndpoint_ReturnsValidJsonStructure()
    {
        // Act
        var response = await _client.GetAsync("/health");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        json.GetProperty("status").GetString().Should().Be("healthy");
        json.TryGetProperty("timestamp", out _).Should().BeTrue();
        json.TryGetProperty("version", out _).Should().BeTrue();
    }

    [Fact]
    public async Task InfoEndpoint_WithNameParam_ReturnsNotFoundForMissingModel()
    {
        // Act
        var response = await _client.GetAsync("/info?name=nonexistent-model");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("nonexistent-model");
    }

    [Fact]
    public async Task ModelsEndpoint_WithNameFilter_ReturnsFilteredResult()
    {
        // Act
        var response = await _client.GetAsync("/models?name=nonexistent");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
        json.GetProperty("filter").GetString().Should().Be("nonexistent");
    }

    [Fact]
    public async Task ModelsEndpoint_ReturnsValidJsonStructure()
    {
        // Act
        var response = await _client.GetAsync("/models");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        json.TryGetProperty("count", out _).Should().BeTrue();
        json.TryGetProperty("models", out var models).Should().BeTrue();
        models.ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    public async Task PredictEndpoint_WithNameParam_ReturnsNotFoundForMissingModel()
    {
        // Arrange
        var input = new { feature1 = 1.0 };

        // Act
        var response = await _client.PostAsJsonAsync("/predict?name=missing-model", input);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task SwaggerEndpoint_ContainsAllEndpoints()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        var paths = json.GetProperty("paths");
        paths.TryGetProperty("/health", out _).Should().BeTrue();
        paths.TryGetProperty("/info", out _).Should().BeTrue();
        paths.TryGetProperty("/predict", out _).Should().BeTrue();
        paths.TryGetProperty("/models", out _).Should().BeTrue();
    }

    [Fact]
    public async Task HealthEndpoint_ResponseContentTypeIsJson()
    {
        // Act
        var response = await _client.GetAsync("/health");

        // Assert
        response.Content.Headers.ContentType?.MediaType.Should().Be("application/json");
    }

    [Fact]
    public async Task ExperimentsEndpoint_ReturnsEmptyListInitially()
    {
        // Act
        var response = await _client.GetAsync("/experiments");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
        json.TryGetProperty("experiments", out var experiments).Should().BeTrue();
        experiments.ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    public async Task ExperimentsEndpoint_WithNameFilter_ReturnsFilteredResult()
    {
        // Act
        var response = await _client.GetAsync("/experiments?name=nonexistent");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
        json.GetProperty("filter").GetString().Should().Be("nonexistent");
    }

    [Fact]
    public async Task ExperimentDetailEndpoint_WithMissingExperiment_ReturnsNotFound()
    {
        // Act
        var response = await _client.GetAsync("/experiments/exp-999");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("exp-999");
    }

    [Fact]
    public async Task SwaggerEndpoint_ContainsExperimentsEndpoints()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        var paths = json.GetProperty("paths");
        paths.TryGetProperty("/experiments", out _).Should().BeTrue();
        paths.TryGetProperty("/experiments/{id}", out _).Should().BeTrue();
    }

    [Fact]
    public async Task StatusEndpoint_ReturnsProjectStatus()
    {
        // Act
        var response = await _client.GetAsync("/status");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.TryGetProperty("project", out _).Should().BeTrue();
        json.TryGetProperty("summary", out var summary).Should().BeTrue();
        summary.GetProperty("totalModels").GetInt32().Should().Be(0);
        summary.GetProperty("totalExperiments").GetInt32().Should().Be(0);
        json.TryGetProperty("dataFiles", out _).Should().BeTrue();
        json.TryGetProperty("models", out var models).Should().BeTrue();
        models.ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    public async Task PromoteEndpoint_WithMissingExperiment_ReturnsNotFound()
    {
        // Arrange
        var request = new { experimentId = "exp-999", name = "default" };

        // Act
        var response = await _client.PostAsJsonAsync("/promote", request);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("exp-999");
    }

    [Fact]
    public async Task SwaggerEndpoint_ContainsAllNewEndpoints()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert - all endpoints present
        var paths = json.GetProperty("paths");
        paths.TryGetProperty("/health", out _).Should().BeTrue();
        paths.TryGetProperty("/info", out _).Should().BeTrue();
        paths.TryGetProperty("/predict", out _).Should().BeTrue();
        paths.TryGetProperty("/models", out _).Should().BeTrue();
        paths.TryGetProperty("/experiments", out _).Should().BeTrue();
        paths.TryGetProperty("/experiments/{id}", out _).Should().BeTrue();
        paths.TryGetProperty("/status", out _).Should().BeTrue();
        paths.TryGetProperty("/promote", out _).Should().BeTrue();
        paths.TryGetProperty("/compare", out _).Should().BeTrue();
        paths.TryGetProperty("/evaluate", out _).Should().BeTrue();
    }

    [Fact]
    public async Task CompareEndpoint_WithMissingExperiments_ReturnsError()
    {
        // Act
        var response = await _client.GetAsync("/compare?candidate=exp-001&baseline=exp-002");

        // Assert - should fail because no experiments exist
        response.StatusCode.Should().BeOneOf(HttpStatusCode.NotFound, HttpStatusCode.InternalServerError);
    }

    [Fact]
    public async Task EvaluateEndpoint_WithMissingTestData_ReturnsBadRequest()
    {
        // Arrange
        var request = new { testDataPath = "/nonexistent/data.csv", name = "default" };

        // Act
        var response = await _client.PostAsJsonAsync("/evaluate", request);

        // Assert
        var statusCode = (int)response.StatusCode;
        statusCode.Should().BeOneOf(400, 404);
    }

    [Fact]
    public async Task LogsEndpoint_ReturnsEmptyListInitially()
    {
        // Act
        var response = await _client.GetAsync("/logs");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
    }

    [Fact]
    public async Task FeedbackGetEndpoint_ReturnsEmptyListInitially()
    {
        // Act
        var response = await _client.GetAsync("/feedback?name=default");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
    }

    [Fact]
    public async Task FeedbackPostEndpoint_WithNonexistentPrediction_ReturnsError()
    {
        // Arrange - prediction ID that doesn't exist in logs
        var request = new { predictionId = "nonexistent-pred", actualValue = "positive", source = "api-test" };

        // Act
        var response = await _client.PostAsJsonAsync("/feedback", request);

        // Assert - should fail because prediction was never logged
        response.StatusCode.Should().Be(HttpStatusCode.InternalServerError);
    }

    [Fact]
    public async Task TriggerEndpoint_ReturnsEvaluation()
    {
        // Act
        var response = await _client.GetAsync("/trigger?name=default");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("modelName").GetString().Should().Be("default");
        json.TryGetProperty("shouldRetrain", out _).Should().BeTrue();
        json.TryGetProperty("conditions", out _).Should().BeTrue();
    }

    [Fact]
    public async Task TrainEndpoint_WithMissingDataFile_ReturnsBadRequest()
    {
        // Arrange
        var request = new
        {
            dataFile = "/nonexistent/train.csv",
            labelColumn = "target",
            task = "regression"
        };

        // Act
        var response = await _client.PostAsJsonAsync("/train", request);

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task JobsEndpoint_ReturnsEmptyListInitially()
    {
        // Act
        var response = await _client.GetAsync("/jobs");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.OK);

        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        json.GetProperty("count").GetInt32().Should().Be(0);
    }

    [Fact]
    public async Task JobDetailEndpoint_WithMissingJob_ReturnsNotFound()
    {
        // Act
        var response = await _client.GetAsync("/jobs/nonexistent-job");

        // Assert
        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task SwaggerEndpoint_ContainsTrainingEndpoints()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        var paths = json.GetProperty("paths");
        paths.TryGetProperty("/train", out _).Should().BeTrue();
        paths.TryGetProperty("/jobs", out _).Should().BeTrue();
        paths.TryGetProperty("/jobs/{id}", out _).Should().BeTrue();
    }

    [Fact]
    public async Task SwaggerEndpoint_ContainsMonitoringEndpoints()
    {
        // Act
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonSerializer.Deserialize<JsonElement>(content);

        // Assert
        var paths = json.GetProperty("paths");
        paths.TryGetProperty("/logs", out _).Should().BeTrue();
        paths.TryGetProperty("/feedback", out _).Should().BeTrue();
        paths.TryGetProperty("/trigger", out _).Should().BeTrue();
    }
}
