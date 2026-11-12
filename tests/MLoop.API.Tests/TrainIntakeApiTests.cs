using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// What <c>POST /train</c> reads from its body. A field it does not know is refused by name — a request that says
/// <c>modelName</c> where the model is <c>name</c> used to be accepted and trained the default model, with no sign
/// that the field had done nothing — and a blank label or task is no choice at all, as a blank <c>dataFile</c> is.
/// </summary>
public class TrainIntakeApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public TrainIntakeApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private string WriteTable()
    {
        var path = Path.Combine(_factory.TestProjectRoot, "datasets", "intake.csv");
        File.WriteAllText(path, "x,y\n" + string.Join('\n', Enumerable.Range(0, 40).Select(i => $"{i},{i * 2}")) + "\n");
        return path;
    }

    private static StringContent Json(string body) => new(body, Encoding.UTF8, "application/json");

    [Fact]
    public async Task AFieldTheRequestDoesNotKnow_IsRefusedByName_AndNoJobIsMade()
    {
        var data = WriteTable();
        var before = await JobCountAsync();

        // `modelName` is what a caller reasonably guesses; the model is `name`.
        var response = await _client.PostAsync("/train", Json(
            $$"""{"dataFile":{{JsonSerializer.Serialize(data)}},"labelColumn":"y","task":"regression","modelName":"elsewhere"}"""));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("Unknown field 'modelName'");
        (await JobCountAsync()).Should().Be(before, "a refused request leaves nothing in the queue");
    }

    [Fact]
    public async Task ABlankLabelColumn_IsNotAColumnNamedNothing()
    {
        var images = Path.Combine(_factory.TestProjectRoot, "datasets", "imgs");
        Directory.CreateDirectory(Path.Combine(images, "a"));
        Directory.CreateDirectory(Path.Combine(images, "b"));

        var response = await _client.PostAsJsonAsync("/train",
            new { dataFile = images, labelColumn = "", task = "image-classification", promote = false });

        response.StatusCode.Should().Be(HttpStatusCode.Accepted);
        var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();
        var job = JsonDocument.Parse(await (await _client.GetAsync($"/jobs/{jobId}")).Content.ReadAsStringAsync()).RootElement;
        job.GetProperty("labelColumn").GetString().Should().NotBeNullOrWhiteSpace(
            "the job records the label it trains with, and an image task's loader makes one");
    }

    [Fact]
    public async Task AJobAnswers_InThePathsOfTheProject_NotOfTheHost()
    {
        WriteTable();

        var accepted = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/intake.csv", labelColumn = "y", task = "regression", promote = false });
        accepted.StatusCode.Should().Be(HttpStatusCode.Accepted);
        var jobId = JsonDocument.Parse(await accepted.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        var text = await (await _client.GetAsync($"/jobs/{jobId}")).Content.ReadAsStringAsync();
        var job = JsonDocument.Parse(text).RootElement;

        job.GetProperty("dataFile").GetString()!.Replace('\\', '/').Should().Be("datasets/intake.csv");
        text.Should().NotContain(_factory.TestProjectRoot, "the answer names the project's paths, not the host's directories");
        (await (await _client.GetAsync("/jobs")).Content.ReadAsStringAsync()).Should().NotContain(_factory.TestProjectRoot);
    }

    [Fact]
    public async Task TheSchema_ListsOnlyTheFieldsTheRequestTakes()
    {
        var response = await _client.GetAsync("/swagger/v1/swagger.json");
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var schema = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement
            .GetProperty("components").GetProperty("schemas").GetProperty("TrainingJobRequest").GetProperty("properties");

        schema.TryGetProperty("name", out _).Should().BeTrue();
        schema.TryGetProperty("modelName", out _).Should().BeFalse("the model is `name`; a derived `modelName` listed as input read as accepted and was ignored");
    }

    private async Task<int> JobCountAsync()
    {
        var response = await _client.GetAsync("/jobs");
        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var root = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        return root.ValueKind == JsonValueKind.Array ? root.GetArrayLength() : root.GetProperty("jobs").GetArrayLength();
    }
}
