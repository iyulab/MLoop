using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /train</c> trains under the project's model definition, as <c>mloop train</c> does — the
/// columns, the weight — and refuses up front what the definition asks for and the API cannot honour.
/// Its own fixture: it writes mloop.yaml, which the other suites do not expect.
/// </summary>
public class TrainDefinitionApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public TrainDefinitionApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private string WriteData()
    {
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(5);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        var path = Path.Combine(_factory.TestProjectRoot, "datasets", "weighted.csv");
        File.WriteAllText(path, csv.ToString());
        return path;
    }

    private void WriteYaml(string model, string body)
        => File.WriteAllText(Path.Combine(_factory.TestProjectRoot, "mloop.yaml"),
            $"project: t\nmodels:\n  {model}:\n{body}");

    [Fact]
    public async Task AWeightOnATaskItWouldNotReach_IsRefusedBeforeAJobExists()
    {
        var data = WriteData();
        WriteYaml("wclust", "    task: clustering\n    label: ''\n    weight: w\n");

        var response = await _client.PostAsJsonAsync("/train", new { dataFile = data, labelColumn = "", task = "clustering", name = "wclust" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("weighted training applies to");
    }

    [Fact]
    public async Task PrepTheApiDoesNotRun_IsRefused_NotSkipped()
    {
        var data = WriteData();
        WriteYaml("wprep", "    task: binary-classification\n    label: y\n    prep:\n      - type: normalize\n        columns: [x]\n");

        var response = await _client.PostAsJsonAsync("/train", new { dataFile = data, labelColumn = "y", task = "binary-classification", name = "wprep" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("prep");
    }

    [Fact]
    public async Task AJob_TrainsWithTheDefinitionsWeightAndColumns()
    {
        var data = WriteData();
        WriteYaml("wbin", "    task: binary-classification\n    label: y\n    weight: w\n    columns:\n      z:\n        type: ignore\n");

        var response = await _client.PostAsJsonAsync("/train",
            new { dataFile = data, labelColumn = "y", task = "binary-classification", name = "wbin", timeLimitSeconds = 5 });
        response.StatusCode.Should().Be(HttpStatusCode.Accepted);
        var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        string? status = null;
        for (var i = 0; i < 240 && status is not ("Completed" or "Failed"); i++)
        {
            await Task.Delay(500);
            var job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
            status = job.GetProperty("status").GetString();
        }
        status.Should().Be("Completed");

        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var experiment = await store.LoadAsync("wbin", "exp-001", CancellationToken.None);
        experiment.Config.WeightColumn.Should().Be("w");
        experiment.Config.InputSchema!.Columns.Single(c => c.Name == "z").Purpose.Should().Be("Exclude");
        experiment.Config.InputSchema.Columns.Single(c => c.Name == "w").Purpose.Should().Be("Exclude");
    }

    [Fact]
    public async Task AJob_MergesAdditionalFiles_UnweightedRowsCountOne_UnlabeledRowsDropped_BudgetFromTheYaml()
    {
        // The original training data and a retraining set from feedback: the second has no weight column
        // (its rows count 1) and rows nobody gave feedback on (empty label — dropped, not a class).
        var data = WriteData();
        var extra = Path.Combine(_factory.TestProjectRoot, "datasets", "feedback.csv");
        File.WriteAllText(extra, "x,z,y\n0.9,0.1,true\n0.1,0.2,false\n0.5,0.3,\n");
        WriteYaml("wmerge", "    task: binary-classification\n    label: y\n    weight: w\n    training:\n      time_limit_seconds: 5\n");

        // No timeLimitSeconds in the request: mloop.yaml's 5 s decides, as an absent --time does.
        var response = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "wmerge",
            additionalDataFiles = new[] { Path.Combine("datasets", "feedback.csv") },   // the project's relative path, as POST /samples answers
        });
        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await response.Content.ReadAsStringAsync());
        var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        JsonElement job = default;
        string? status = null;
        for (var i = 0; i < 240 && status is not ("Completed" or "Failed"); i++)
        {
            await Task.Delay(500);
            job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
            status = job.GetProperty("status").GetString();
        }
        status.Should().Be("Completed", job.GetRawText());
        job.TryGetProperty("timeLimitSeconds", out var limit).Should().Be(false, "an absent budget is left to mloop.yaml");
        job.GetProperty("additionalDataFiles").EnumerateArray().Select(f => f.GetString())
            .Should().Equal(Path.GetFullPath(extra));

        var merged = File.ReadAllLines(Path.Combine(_factory.TestProjectRoot, ".mloop", "merged", $"{jobId}.csv"));
        merged.Should().HaveCount(1 + 200 + 3);
        merged.TakeLast(3).Should().AllSatisfy(line => line.Split(',')[2].Should().Be("1"));

        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var experiment = await store.LoadAsync("wmerge", "exp-001", CancellationToken.None);
        experiment.Config.WeightColumn.Should().Be("w");
        experiment.Config.InputSchema!.Columns.Single(c => c.Name == "y").CategoricalValues
            .Should().BeEquivalentTo(["true", "false"], "an empty label is not a class");
    }

    [Fact]
    public async Task TheRequestsWeight_TrainsWeighted_WhenTheYamlDeclaresNone()
    {
        // A retraining set answered weightColumn = "w"; mloop.yaml says nothing about weights.
        var data = WriteData();
        WriteYaml("wreq", "    task: binary-classification\n    label: y\n");

        var response = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "wreq",
            timeLimitSeconds = 5,
            weight = "w",
        });
        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await response.Content.ReadAsStringAsync());
        var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

        string? status = null;
        for (var i = 0; i < 240 && status is not ("Completed" or "Failed"); i++)
        {
            await Task.Delay(500);
            status = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement.GetProperty("status").GetString();
        }
        status.Should().Be("Completed");

        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        (await store.LoadAsync("wreq", "exp-001", CancellationToken.None)).Config.WeightColumn.Should().Be("w");
    }

    [Fact]
    public async Task AnAdditionalFileOutsideTheProjectOrMissing_IsRefusedBeforeAJobExists()
    {
        var data = WriteData();
        var missing = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            additionalDataFiles = new[] { Path.Combine(_factory.TestProjectRoot, "datasets", "nope.csv") },
        });
        missing.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await missing.Content.ReadAsStringAsync()).Should().Contain("nope.csv");

        var outside = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            additionalDataFiles = new[] { Path.Combine(Path.GetTempPath(), "elsewhere.csv") },
        });
        outside.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    private string WriteImbalanced(string name, int positives)
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(9);
        for (var i = 0; i < 200; i++)
        {
            var positive = i < positives;
            csv.Append(CultureInfo.InvariantCulture, $"{(positive ? 0.7 : 0.3) + rng.NextDouble() * 0.3:F4},{rng.NextDouble():F4},{(positive ? "yes" : "no")}\n");
        }
        var path = Path.Combine(_factory.TestProjectRoot, "datasets", name);
        File.WriteAllText(path, csv.ToString());
        return path;
    }

    private async Task<JsonElement> AwaitJobAsync(HttpResponseMessage response)
    {
        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await response.Content.ReadAsStringAsync());
        var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();
        JsonElement job = default;
        for (var i = 0; i < 240; i++)
        {
            await Task.Delay(500);
            job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
            if (job.GetProperty("status").GetString() is "Completed" or "Failed")
                break;
        }
        return job;
    }

    [Theory]
    [InlineData("binary-classification", "lots", null, "Invalid balance option")]
    [InlineData("regression", "auto", null, "classification")]
    [InlineData("binary-classification", "auto", "z", "both change how often rows count")]
    public async Task ABalanceTheRunCouldNotHonour_IsRefusedBeforeAJobExists(string task, string balance, string? weight, string message)
    {
        var data = WriteImbalanced("refused.csv", 20);
        WriteYaml("bref", $"    task: {task}\n    label: {(task == "regression" ? "x" : "y")}\n");

        var response = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = task == "regression" ? "x" : "y",
            task,
            name = "bref",
            balance,
            weight,
        });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain(message);
    }

    [Fact]
    public async Task ABalancedJob_BalancesTheTrainPartOnly_AsMloopTrainDoes()
    {
        var data = WriteImbalanced("imbalanced.csv", 20);   // 9:1
        WriteYaml("bal", "    task: binary-classification\n    label: y\n");

        var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "bal",
            timeLimitSeconds = 5,
            balance = "2",
        }));

        job.GetProperty("status").GetString().Should().Be("Completed", job.GetRawText());
        job.GetProperty("balance").GetString().Should().Be("2");
        var store = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IExperimentStore>();
        var config = (await store.LoadAsync("bal", "exp-001", CancellationToken.None)).Config;
        config.DataFile.Should().EndWith("_balanced.csv");
        config.TestSplit.Should().Be(0, "the test part was split off before balancing, not split again from balanced rows");
    }

    [Fact]
    public async Task AJobAskedNotToPromote_KeepsItsExperimentInStaging_AndSaysSo()
    {
        var data = WriteImbalanced("nopromote.csv", 60);
        WriteYaml("nopro", "    task: binary-classification\n    label: y\n");

        var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "nopro",
            timeLimitSeconds = 5,
            promote = false,
        }));

        job.GetProperty("status").GetString().Should().Be("Completed", job.GetRawText());
        job.GetProperty("promote").GetBoolean().Should().BeFalse();
        job.GetProperty("promoted").GetBoolean().Should().BeFalse();
        job.GetProperty("keptInStagingReason").GetString().Should().Contain("\"promote\": false");
        job.GetProperty("statusMessage").GetString().Should().Contain("kept in staging");
        var registry = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IModelRegistry>();
        (await registry.GetProductionAsync("nopro")).Should().BeNull("the first model of a model name would otherwise have gone in");
    }

    [Fact]
    public async Task AJobWithANearPerfectScore_CarriesTheDiagnosticsWarning()
    {
        // The label is the feature crossing a line: held out perfectly, on enough rows to mean something.
        var path = Path.Combine(_factory.TestProjectRoot, "datasets", "leak.csv");
        var random = new Random(5);
        var lines = new List<string> { "x,noise,y" };
        for (var i = 0; i < 800; i++)
        {
            var x = random.NextDouble();
            lines.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{x:F4},{random.NextDouble():F4},{(x > 0.5 ? 1 : 0)}"));
        }
        File.WriteAllLines(path, lines);
        WriteYaml("leak", "    task: binary-classification\n    label: y\n");

        var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
        {
            dataFile = path,
            labelColumn = "y",
            task = "binary-classification",
            name = "leak",
            timeLimitSeconds = 5,
            promote = false,
        }));

        job.GetProperty("status").GetString().Should().Be("Completed", job.GetRawText());
        job.GetProperty("warnings").EnumerateArray().Select(w => w.GetString())
            .Should().Contain(w => w!.Contains("Near-perfect on held-out rows"), job.GetRawText());
    }

    [Fact]
    public async Task AJob_PromotesByDefault_AndTheCompletedJobSaysItWas()
    {
        var data = WriteImbalanced("promote.csv", 60);
        WriteYaml("pro", "    task: binary-classification\n    label: y\n");

        var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "pro",
            timeLimitSeconds = 5,
        }));

        job.GetProperty("status").GetString().Should().Be("Completed", job.GetRawText());
        job.GetProperty("promote").GetBoolean().Should().BeTrue();
        job.GetProperty("promoted").GetBoolean().Should().BeTrue(job.GetRawText());
        (job.TryGetProperty("keptInStagingReason", out var reason) ? reason.ValueKind : JsonValueKind.Null).Should().Be(JsonValueKind.Null);
        // Completed is written after the decision, so production is already in place when a poller reads it.
        var registry = _factory.Services.GetRequiredService<MLoop.CLI.Infrastructure.FileSystem.IModelRegistry>();
        (await registry.GetProductionAsync("pro"))!.ExperimentId.Should().Be(job.GetProperty("experimentId").GetString());
    }

    [Fact]
    public async Task ExperimentRows_CarryTheirMetricsAndTheProductionFlag_AsMloopListJsonDoes()
    {
        var data = WriteImbalanced("listed.csv", 60);
        WriteYaml("lst", "    task: binary-classification\n    label: y\n");
        foreach (var promote in new[] { true, false })
        {
            var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
            {
                dataFile = data,
                labelColumn = "y",
                task = "binary-classification",
                name = "lst",
                timeLimitSeconds = 5,
                promote,
            }));
            job.GetProperty("status").GetString().Should().Be("Completed", job.GetRawText());
        }

        var listed = JsonDocument.Parse(await _client.GetStringAsync("/experiments?name=lst")).RootElement;

        var rows = listed.GetProperty("experiments").EnumerateArray().ToList();
        rows.Select(r => r.GetProperty("experimentId").GetString()).Should().Equal("exp-002", "exp-001"); // newest first
        rows.Select(r => r.GetProperty("isProduction").GetBoolean()).Should().Equal(false, true);
        rows.Should().AllSatisfy(r => r.GetProperty("metrics").TryGetProperty("auc", out _).Should().BeTrue(r.GetRawText()));
        listed.GetProperty("production").GetProperty("lst").GetString().Should().Be("exp-001");
    }

    [Fact]
    public async Task AJobOnOneClass_FailsWithTheReason()
    {
        var data = WriteImbalanced("oneclass.csv", 0);
        WriteYaml("one", "    task: binary-classification\n    label: y\n");

        var job = await AwaitJobAsync(await _client.PostAsJsonAsync("/train", new
        {
            dataFile = data,
            labelColumn = "y",
            task = "binary-classification",
            name = "one",
            timeLimitSeconds = 5,
        }));

        job.GetProperty("status").GetString().Should().Be("Failed");
        job.GetProperty("statusMessage").GetString().Should().Contain("only one class");
    }
}
