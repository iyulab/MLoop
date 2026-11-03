using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /train</c> refuses a binary-classification task whose label holds more than two values as <c>mloop train</c> does:
/// the job fails at once and says why, instead of every trainer failing its first trials on "Schema mismatch for label column".
/// </summary>
public class BinaryLabelApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public BinaryLabelApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task AJobWhoseBinaryLabelHoldsThreeValues_FailsAtOnce_NamingTheLabelAndTheWayOut()
    {
        var csv = new StringBuilder("a,b,y\n");
        var rng = new Random(9);
        for (var i = 0; i < 90; i++)
            csv.Append(CultureInfo.InvariantCulture, $"{rng.NextDouble():F3},{rng.NextDouble():F3},{"xyz"[i % 3]}\n");
        await File.WriteAllTextAsync(Path.Combine(_factory.TestProjectRoot, "datasets", "three.csv"), csv.ToString());

        var started = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/three.csv",
            labelColumn = "y",
            task = "binary-classification",
            name = "three",
            timeLimitSeconds = 10,
            maxTrials = 3,
        });
        started.StatusCode.Should().Be(HttpStatusCode.Accepted, await started.Content.ReadAsStringAsync());
        var jobId = JsonDocument.Parse(await started.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        JsonElement job = default;
        string text = "";
        for (var i = 0; i < 120; i++)
        {
            job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement.Clone();
            var status = job.GetProperty("status");
            text = status.ValueKind == JsonValueKind.String ? status.GetString()! : status.GetRawText();
            if (text is "Failed" or "3" or "Completed" or "2") break;
            await Task.Delay(250);
        }

        text.Should().BeOneOf("Failed", "3");
        var message = job.GetProperty("statusMessage").GetString()!;
        message.Should().Contain("'y'").And.Contain("3 different values").And.Contain("multiclass-classification");
        message.Should().NotContain("Schema mismatch");
    }
}
