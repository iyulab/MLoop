using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /train</c> on a file with nothing to learn from says what is wrong with the file, as <c>mloop train</c> does: a job on a
/// file with no content, or with a header and no rows, fails at once with the file's name and the cause - not the parser's state
/// or advice about the label column.
/// </summary>
public class EmptyInputApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public EmptyInputApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Theory]
    [InlineData("empty-input.csv", "", "is empty")]
    [InlineData("header-only-input.csv", "a,b,y\n", "has a header row and no data rows")]
    public async Task AJobOnAFileWithNothingToLearnFrom_FailsAtOnce_SayingSo(string fileName, string content, string expected)
    {
        await File.WriteAllTextAsync(Path.Combine(_factory.TestProjectRoot, "datasets", fileName), content);

        var started = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = $"datasets/{fileName}",
            labelColumn = "y",
            task = "binary-classification",
            name = "emptyinput",
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
        message.Should().Contain(fileName).And.Contain(expected);
        message.Should().NotContain("only one class").And.NotContain("No header record");
    }
}
