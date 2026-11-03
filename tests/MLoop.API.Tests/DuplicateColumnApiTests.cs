using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /train</c> refuses a table with two columns of one name as <c>mloop train</c> does: the job fails at once and names the
/// columns, instead of searching to the end and losing the run to a dictionary error.
/// </summary>
public class DuplicateColumnApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public DuplicateColumnApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task AJobOnATableWithTwoColumnsOfOneName_FailsAtOnce_NamingThem()
    {
        var csv = new StringBuilder("a,v,v,y\n");
        for (var i = 0; i < 60; i++)
            csv.Append($"{i / 60.0:F3},0.5,0.25,{(i % 2 == 0 ? "yes" : "no")}\n");
        await File.WriteAllTextAsync(Path.Combine(_factory.TestProjectRoot, "datasets", "dup.csv"), csv.ToString());

        var started = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/dup.csv",
            labelColumn = "y",
            task = "binary-classification",
            name = "dup",
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
        message.Should().Contain("'v' is columns 2, 3").And.Contain("Rename one of them");
        message.Should().NotContain("same key");
    }
}
