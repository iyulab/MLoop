using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// What the search validated the kept model at on the training rows, with its standard error, is on the job a client polls and
/// on the experiment it reads — the same numbers <c>mloop train</c> prints under its table.
/// </summary>
public class ValidatedScoreApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ValidatedScoreApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task ACompletedJob_AndItsExperiment_CarryTheValidatedScore()
    {
        var csv = new StringBuilder("a,b,y\n");
        var rng = new Random(21);
        for (var i = 0; i < 160; i++)
        {
            var a = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{a:F4},{rng.NextDouble():F4},{3 * a + rng.NextDouble():F4}\n");
        }
        await File.WriteAllTextAsync(Path.Combine(_factory.TestProjectRoot, "datasets", "validated.csv"), csv.ToString());

        var started = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/validated.csv",
            labelColumn = "y",
            task = "regression",
            name = "validated",
            timeLimitSeconds = 10,
            maxTrials = 4,
        });
        started.StatusCode.Should().Be(HttpStatusCode.Accepted, await started.Content.ReadAsStringAsync());
        var jobId = JsonDocument.Parse(await started.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        JsonElement job = default;
        string status = "";
        for (var i = 0; i < 240; i++)
        {
            job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement.Clone();
            var s = job.GetProperty("status");
            status = s.ValueKind == JsonValueKind.String ? s.GetString()! : s.GetRawText();
            if (status is "Failed" or "3" or "Completed" or "2") break;
            await Task.Delay(250);
        }
        status.Should().BeOneOf("Completed", "2", job.GetRawText());

        var validated = job.GetProperty("validated");
        validated.GetProperty("metric").GetString().Should().Be("r_squared");
        validated.GetProperty("folds").GetInt32().Should().BeGreaterThan(0);
        validated.GetProperty("standardError").GetDouble().Should().BeGreaterThanOrEqualTo(0);
        validated.GetProperty("value").GetDouble().Should().BeInRange(-1, 1);

        var experimentId = job.GetProperty("experimentId").GetString();
        var experiment = JsonDocument.Parse(await _client.GetStringAsync($"/experiments/{experimentId}?name=validated")).RootElement;
        experiment.GetProperty("validated").GetRawText().Should().Be(validated.GetRawText(), "the job and the experiment say the same thing");
    }
}
