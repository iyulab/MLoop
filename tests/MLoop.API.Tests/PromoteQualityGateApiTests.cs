using System.Net;
using System.Net.Http.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /promote</c> applies the quality gate. Its own fixture, because the experiment it writes
/// would otherwise show up in the suites that assert an empty project.
/// </summary>
public class PromoteQualityGateApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public PromoteQualityGateApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task PromoteEndpoint_RefusesAModelTheQualityGateRefuses_UnlessForced()
    {
        // The gate training and `mloop promote` apply: a model answering one class to every row
        // (accuracy 0.993 on 0.6% positives, F1 0) is refused with the reason; force overrides.
        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var path = store.GetExperimentPath("gate", "exp-001");
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
        await store.SaveAsync("gate", new MLoop.CLI.Infrastructure.FileSystem.ExperimentData
        {
            ModelName = "gate",
            ExperimentId = "exp-001",
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "binary-classification",
            Config = new MLoop.CLI.Infrastructure.FileSystem.ExperimentConfig
            {
                DataFile = "train.csv", LabelColumn = "y", Metric = "f1_score", TimeLimitSeconds = 60, TestSplit = 0.2,
            },
            Metrics = new Dictionary<string, double>
            {
                ["accuracy"] = 0.993, ["f1_score"] = 0, ["negative_recall"] = 1, ["auc"] = 0.95,
            },
        }, CancellationToken.None);

        var refused = await _client.PostAsJsonAsync("/promote", new { experimentId = "exp-001", name = "gate" });
        refused.StatusCode.Should().Be(HttpStatusCode.UnprocessableEntity);
        (await refused.Content.ReadAsStringAsync()).Should().Contain("quality gate").And.Contain("f1_score");

        var forced = await _client.PostAsJsonAsync("/promote", new { experimentId = "exp-001", name = "gate", force = true });
        forced.StatusCode.Should().Be(HttpStatusCode.OK);
    }
}
