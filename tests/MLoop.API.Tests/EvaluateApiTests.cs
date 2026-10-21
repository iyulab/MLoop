using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /evaluate</c> weighs the rows the way <c>mloop evaluate</c> does and compares two models on
/// the same rows the same way — the endpoint once evaluated a weighted experiment unweighted, with
/// nothing in the response saying so. Its own fixture: the experiments it trains would otherwise show
/// up in the suites that assert an empty project.
/// </summary>
public class EvaluateApiTests : IClassFixture<TestWebApplicationFactory>, IAsyncLifetime
{
    private const string Model = "weighted";
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;
    private string _withWeight = "";
    private string _withoutWeight = "";

    public EvaluateApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    public async Task InitializeAsync()
    {
        var datasets = Path.Combine(_factory.TestProjectRoot, "datasets");
        _withWeight = Path.Combine(datasets, "weighted.csv");
        _withoutWeight = Path.Combine(datasets, "unweighted.csv");
        if (File.Exists(_withWeight))
            return; // the class fixture is shared by every test in this class; train once

        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync(_withWeight, csv.ToString());
        await File.WriteAllLinesAsync(_withoutWeight,
            (await File.ReadAllLinesAsync(_withWeight)).Select(l => string.Join(',', l.Split(',').Where((_, i) => i != 2))));

        var fs = _factory.Services.GetRequiredService<IFileSystemManager>();
        var store = _factory.Services.GetRequiredService<IExperimentStore>();
        // Training narrates to the console; this assembly runs in parallel, so the console is left alone.
        for (var run = 0; run < 2; run++)
            await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = Model,
                DataFile = _withWeight,
                LabelColumn = "y",
                Task = "binary-classification",
                WeightColumn = "w",
                TimeLimitSeconds = 5,
            }, progress: null, CancellationToken.None);
    }

    public Task DisposeAsync() => Task.CompletedTask;

    private async Task<(HttpStatusCode Status, JsonElement Body)> EvaluateAsync(object request)
    {
        var response = await _client.PostAsJsonAsync("/evaluate", request);
        var text = await response.Content.ReadAsStringAsync();
        return (response.StatusCode, JsonDocument.Parse(text).RootElement.Clone());
    }

    [Fact]
    public async Task AWeightedExperiment_IsScoredByItsWeight_WhenTheFileHasIt()
    {
        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withWeight });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.GetProperty("weightColumn").GetString().Should().Be("w");
        body.TryGetProperty("warnings", out _).Should().BeFalse("the API leaves null fields out");
    }

    [Fact]
    public async Task AWeightedExperiment_OnAFileWithoutItsWeight_SaysTheMetricsAreUnweighted()
    {
        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withoutWeight });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        body.TryGetProperty("weightColumn", out _).Should().BeFalse("the API leaves null fields out");
        body.GetProperty("warnings")[0].GetString().Should().Contain("weighted by 'w'");
    }

    [Fact]
    public async Task Against_PairsTheTwoModelsOnTheSameRowsAndWeights()
    {
        var (status, body) = await EvaluateAsync(new { name = Model, experimentId = "exp-002", testDataPath = _withWeight, against = "exp-001" });

        status.Should().Be(HttpStatusCode.OK, body.ToString());
        var comparison = body.GetProperty("comparison");
        comparison.GetProperty("against").GetString().Should().Be("exp-001");
        comparison.GetProperty("metrics").GetProperty("accuracy").GetDouble().Should().BeInRange(0, 1);
        var paired = comparison.GetProperty("paired");
        paired.GetProperty("measure").GetString().Should().Be("error_rate");
        paired.GetProperty("rows").GetInt32().Should().Be(200);
        paired.GetProperty("low").GetDouble().Should().BeLessThanOrEqualTo(paired.GetProperty("high").GetDouble());
        paired.GetProperty("verdict").GetString().Should().BeOneOf("CandidateBetter", "BaselineBetter", "Indistinguishable");
    }

    [Fact]
    public async Task AgainstItself_IsABadRequest_AndAgainstAnUnknownExperiment_IsNotFound()
    {
        var (self, selfBody) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withWeight, against = "exp-001" });
        var (missing, missingBody) = await EvaluateAsync(new { name = Model, experimentId = "exp-001", testDataPath = _withWeight, against = "exp-404" });

        self.Should().Be(HttpStatusCode.BadRequest);
        selfBody.GetProperty("error").GetString().Should().Contain("the experiment being evaluated");
        missing.Should().Be(HttpStatusCode.NotFound);
        missingBody.GetProperty("error").GetString().Should().Contain("exp-404");
    }
}
