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
}
