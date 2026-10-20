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

    [Fact]
    public async Task PromoteEndpoint_PassesAModelThatRanksButSitsOnOneClass_WithAQualityWarning()
    {
        // Ranks well (AUC 0.968) but answers one class at its default threshold: not refused on a
        // ranking metric, and the response says what its predicted labels are worth.
        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var path = store.GetExperimentPath("ranks", "exp-001");
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
        await store.SaveAsync("ranks", new MLoop.CLI.Infrastructure.FileSystem.ExperimentData
        {
            ModelName = "ranks", ExperimentId = "exp-001", Timestamp = DateTime.UtcNow, Status = "Completed",
            Task = "binary-classification",
            Config = new MLoop.CLI.Infrastructure.FileSystem.ExperimentConfig
            {
                DataFile = "train.csv", LabelColumn = "y", Metric = "auc", TimeLimitSeconds = 60, TestSplit = 0.2,
            },
            Metrics = new Dictionary<string, double> { ["accuracy"] = 0.969, ["f1_score"] = 0, ["negative_recall"] = 1, ["auc"] = 0.968 },
        }, CancellationToken.None);

        var response = await _client.PostAsJsonAsync("/promote", new { experimentId = "exp-001", name = "ranks" });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        (await response.Content.ReadAsStringAsync()).Should().Contain("\"qualityWarning\":\"it ranks the classes (auc 0.9680)");
    }

    [Fact]
    public async Task RollbackEndpoint_ReturnsProductionToWhatItReplaced()
    {
        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        foreach (var id in new[] { "exp-001", "exp-002" })
        {
            var path = store.GetExperimentPath("rb", id);
            Directory.CreateDirectory(path);
            await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
            await store.SaveAsync("rb", new MLoop.CLI.Infrastructure.FileSystem.ExperimentData
            {
                ModelName = "rb", ExperimentId = id, Timestamp = DateTime.UtcNow, Status = "Completed",
                Task = "binary-classification",
                Config = new MLoop.CLI.Infrastructure.FileSystem.ExperimentConfig
                {
                    DataFile = "train.csv", LabelColumn = "y", Metric = "auc", TimeLimitSeconds = 60, TestSplit = 0.2,
                },
                Metrics = new Dictionary<string, double> { ["accuracy"] = 0.9, ["f1_score"] = 0.88, ["negative_recall"] = 0.9, ["auc"] = 0.95 },
            }, CancellationToken.None);
            (await _client.PostAsJsonAsync("/promote", new { experimentId = id, name = "rb" })).StatusCode.Should().Be(HttpStatusCode.OK);
        }

        var response = await _client.PostAsJsonAsync("/rollback", new { name = "rb" });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        (await response.Content.ReadAsStringAsync()).Should().Contain("\"experimentId\":\"exp-001\"");

        var refused = await _client.PostAsJsonAsync("/rollback", new { name = "rb", to = "exp-009" });
        refused.StatusCode.Should().Be(HttpStatusCode.Conflict);
    }
}
