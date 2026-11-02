using System.Net;
using System.Text;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// A request body that leaves out what an endpoint needs is the caller's mistake, and the answer is a 4xx that says so — never a 500
/// (which reads as the server's fault and, in development, carries a stack trace). Every endpoint that takes a body is sent the
/// bodies a careless caller sends: nothing, <c>null</c>, <c>{}</c>, an array, and an object of the wrong shape.
/// </summary>
public class EmptyBodyApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly HttpClient _client;

    public EmptyBodyApiTests(TestWebApplicationFactory factory) => _client = factory.CreateClient();

    public static TheoryData<string> Endpoints => new()
    {
        "/train", "/evaluate", "/promote", "/rollback", "/samples", "/feedback", "/predict", "/embed",
    };

    public static TheoryData<string, string> Bodies()
    {
        var data = new TheoryData<string, string>();
        foreach (var endpoint in new[] { "/train", "/evaluate", "/promote", "/rollback", "/samples", "/feedback", "/predict", "/embed" })
            foreach (var body in new[] { "", "null", "{}", "[]", "{\"unexpected\":1}", "{\"dataFile\":null,\"task\":null}", "\"text\"", "{" })
                data.Add(endpoint, body);
        return data;
    }

    [Theory]
    [MemberData(nameof(Bodies))]
    public async Task ABodyThatLeavesOutWhatIsNeeded_IsRefusedAs4xx_NotA500(string endpoint, string body)
    {
        using var content = new StringContent(body, Encoding.UTF8, "application/json");

        var response = await _client.PostAsync(endpoint, content);

        ((int)response.StatusCode).Should().BeLessThan(500,
            $"POST {endpoint} with body '{body}' answered {(int)response.StatusCode}: {Truncate(await response.Content.ReadAsStringAsync())}");
    }

    private static string Truncate(string text) => text.Length <= 300 ? text : text[..300];

    /// <summary>An experiment id is a folder name: one that climbs out of the model's folder is not an experiment.</summary>
    [Theory]
    [InlineData("..")]
    [InlineData("../..")]
    [InlineData("a/b")]
    [InlineData("a\\b")]
    [InlineData("   ")]
    public async Task AnExperimentIdThatIsNotAFolderName_IsRefusedAs4xx(string id)
    {
        using var content = new StringContent(System.Text.Json.JsonSerializer.Serialize(new { experimentId = id }), Encoding.UTF8, "application/json");

        var response = await _client.PostAsync("/promote", content);

        ((int)response.StatusCode).Should().BeInRange(400, 499, await response.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("/experiments/..")]
    [InlineData("/experiments/..%2F..")]
    [InlineData("/experiments/a%2Fb")]
    public async Task AnExperimentIdInThePath_ThatIsNotAFolderName_IsNotFoundOrRefused(string path)
    {
        var response = await _client.GetAsync(path);

        ((int)response.StatusCode).Should().BeInRange(400, 499, await response.Content.ReadAsStringAsync());
    }
}
