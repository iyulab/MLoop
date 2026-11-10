using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.API.Tests;

/// <summary>
/// The promotion history names who changed the production slot: the <c>sub</c> of the calling token for <c>POST /promote</c> and
/// <c>POST /rollback</c>, and nobody when the token carries none. Its own fixture, as <see cref="PromoteQualityGateApiTests"/> is:
/// the experiments it writes would show up in the suites that assert an empty project.
/// </summary>
public class PromotionOriginApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public PromotionOriginApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private async Task SeedAsync(string model, string experimentId)
    {
        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var path = store.GetExperimentPath(model, experimentId);
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model " + experimentId);
        await store.SaveAsync(model, new MLoop.CLI.Infrastructure.FileSystem.ExperimentData
        {
            ModelName = model,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new MLoop.CLI.Infrastructure.FileSystem.ExperimentConfig
            {
                DataFile = "train.csv",
                LabelColumn = "y",
                Metric = "r_squared",
                TimeLimitSeconds = 60,
                TestSplit = 0.2,
            },
            Metrics = new Dictionary<string, double> { ["r_squared"] = 0.8, ["rmse"] = 1.0 },
        }, CancellationToken.None);
    }

    private static HttpRequestMessage Post(string url, object body, string? subject)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = JsonContent.Create(body) };
        if (subject is not null)
            request.Headers.Add(TestAuthHandler.SubjectHeader, subject);
        return request;
    }

    [Fact]
    public async Task ThePromotionHistory_NamesTheTokenSubjectWhoPromotedAndRolledBack_AndNobodyForATokenWithoutOne()
    {
        await SeedAsync("who", "exp-001");
        await SeedAsync("who", "exp-002");

        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-001", name = "who" }, "ci-bot-9"))).EnsureSuccessStatusCode();
        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-002", name = "who" }, null))).EnsureSuccessStatusCode();
        (await _client.SendAsync(Post("/rollback", new { name = "who" }, "alice@example.test"))).EnsureSuccessStatusCode();

        var rows = JsonDocument.Parse(await _client.GetStringAsync("/promotions?name=who")).RootElement
            .GetProperty("promotions").EnumerateArray().ToList();

        rows.Select(r => r.GetProperty("action").GetString()).Should().Equal("promote", "promote", "rollback");
        rows.Select(r => r.GetProperty("via").GetString()).Should().OnlyContain(v => v == "http");
        rows[0].GetProperty("by").GetString().Should().Be("ci-bot-9");
        // The server leaves a null out of its JSON: a token without a subject has no "by" at all, never an empty name.
        (rows[1].TryGetProperty("by", out var nobody) ? nobody.ValueKind : JsonValueKind.Null).Should().Be(JsonValueKind.Null);
        rows[2].GetProperty("by").GetString().Should().Be("alice@example.test");
    }

    [Fact]
    public async Task ARollbackWithNoBodyAtAll_ReturnsTheDefaultModelToItsPreviousExperiment()
    {
        // `mloop rollback` takes neither a name nor a target; the endpoint's body is as optional, so a bare POST is a rollback of the
        // default model and not a 400 about a body.
        await SeedAsync("default", "exp-001");
        await SeedAsync("default", "exp-002");
        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-001" }, null))).EnsureSuccessStatusCode();
        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-002" }, null))).EnsureSuccessStatusCode();

        using var bare = new HttpRequestMessage(HttpMethod.Post, "/rollback");
        var response = await _client.SendAsync(bare);

        response.StatusCode.Should().Be(System.Net.HttpStatusCode.OK, await response.Content.ReadAsStringAsync());
        var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        body.GetProperty("experimentId").GetString().Should().Be("exp-001");
        body.GetProperty("previousExperimentId").GetString().Should().Be("exp-002");
    }

    [Fact]
    public async Task PromotingTheExperimentAlreadyInProduction_ChangesNothing_UnlessForced()
    {
        await SeedAsync("same", "exp-001");
        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-001", name = "same" }, null))).EnsureSuccessStatusCode();

        var again = await _client.SendAsync(Post("/promote", new { experimentId = "exp-001", name = "same" }, null));
        again.EnsureSuccessStatusCode();
        JsonDocument.Parse(await again.Content.ReadAsStringAsync()).RootElement.GetProperty("unchanged").GetBoolean().Should().BeTrue();
        CountPromotions("same").Should().Be(1, "a promotion that changes nothing is not a change of production");

        (await _client.SendAsync(Post("/promote", new { experimentId = "exp-001", name = "same", force = true }, null))).EnsureSuccessStatusCode();
        CountPromotions("same").Should().Be(2);
    }

    private int CountPromotions(string model)
        => JsonDocument.Parse(_client.GetStringAsync("/promotions?name=" + model).GetAwaiter().GetResult()).RootElement.GetProperty("promotions").GetArrayLength();

    [Fact]
    public async Task ThePromotionHistory_ReadsARecordWrittenBeforeWhoWasRecorded_WithNoNameAndNoDoor()
    {
        // history.json files of earlier versions have neither field: they read as unknown, not as a failure.
        var manager = new MLoop.Ops.Services.FilePromotionManager(_factory.Services.GetRequiredService<MLoop.API.ProjectRootPath>().Value);
        var old = await manager.GetHistoryAsync("never-recorded");
        old.Should().BeEmpty();

        var record = JsonSerializer.Deserialize<MLoop.Ops.Interfaces.PromotionRecord>(
            "{\"modelName\":\"m\",\"experimentId\":\"exp-001\",\"previousExperimentId\":null,\"action\":\"promote\",\"reason\":null,\"timestamp\":\"2026-01-01T00:00:00+00:00\"}",
            new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true });

        record!.By.Should().BeNull();
        record.Via.Should().BeNull();
    }
}
