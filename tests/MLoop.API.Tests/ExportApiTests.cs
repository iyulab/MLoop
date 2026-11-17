using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using MLoop.Core.Export;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /experiments/{id}/export/onnx</c> writes what <c>mloop export onnx</c> writes — a consumer of the service has no
/// command line beside it — and the two files come back over <c>GET</c>. One portable model is trained for the suite; the
/// tests read what it wrote, never what the endpoint says about it.
/// </summary>
public class ExportApiTests : IClassFixture<TestWebApplicationFactory>
{
    private const string Model = "xport";

    private static readonly SemaphoreSlim Training = new(1, 1);
    private static string? _trainedAt;
    private static string? _experimentId;

    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ExportApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private static string Csv()
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(11);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        return csv.ToString();
    }

    /// <summary>Trains the portable model once for the class and returns its experiment id and the file it was trained on.</summary>
    private async Task<(string Id, string Data)> TrainAsync()
    {
        await Training.WaitAsync();
        try
        {
            if (_experimentId is not null && _trainedAt == _factory.TestProjectRoot)
                return (_experimentId, Path.Combine(_factory.TestProjectRoot, "datasets", "export.csv"));

            var data = Path.Combine(_factory.TestProjectRoot, "datasets", "export.csv");
            File.WriteAllText(data, Csv());
            var response = await _client.PostAsJsonAsync("/train",
                new { dataFile = data, labelColumn = "y", task = "binary-classification", name = Model, timeLimitSeconds = 60, maxTrials = 10, portable = true });
            response.StatusCode.Should().Be(HttpStatusCode.Accepted);
            var jobId = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("jobId").GetString();

            string? status = null;
            JsonElement job = default;
            for (var i = 0; i < 240 && status is not ("Completed" or "Failed"); i++)
            {
                await Task.Delay(500);
                job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
                status = job.GetProperty("status").GetString();
            }
            status.Should().Be("Completed", job.ToString());
            _experimentId = job.GetProperty("experimentId").GetString()!;
            _trainedAt = _factory.TestProjectRoot;
            return (_experimentId, data);
        }
        finally
        {
            Training.Release();
        }
    }

    private static string Sha256(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    [Fact]
    public async Task APortableExperiment_IsExported_AndTheTwoFilesComeBackWithTheShasTheAnswerNames()
    {
        var (id, _) = await TrainAsync();

        var response = await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", content: null);

        response.StatusCode.Should().Be(HttpStatusCode.OK, await response.Content.ReadAsStringAsync());
        var answer = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        answer.GetProperty("experimentId").GetString().Should().Be(id);
        answer.GetProperty("task").GetString().Should().Be("binary-classification");
        answer.GetProperty("portable").GetBoolean().Should().BeTrue("it was trained with the request's portable");

        var graph = await _client.GetByteArrayAsync(answer.GetProperty("onnxUrl").GetString());
        var manifestBytes = await _client.GetByteArrayAsync(answer.GetProperty("manifestUrl").GetString());
        graph.Length.Should().BeGreaterThan(0);
        Sha256(graph).Should().Be(answer.GetProperty("onnxSha256").GetString(), "the answer's hash is of the bytes the URL serves");
        Sha256(manifestBytes).Should().Be(answer.GetProperty("manifestSha256").GetString());

        var manifest = OnnxManifest.FromJson(Encoding.UTF8.GetString(manifestBytes));
        manifest.OnnxSha256.Should().Be(Sha256(graph), "the manifest names the graph beside it");
        manifest.SourceModelSha256.Should().Be(answer.GetProperty("sourceModelSha256").GetString());
        manifest.ExperimentId.Should().Be(id);
        manifest.Portable.Should().BeTrue();

        // What the command line writes into the experiment's directory is here too — one exporter.
        var directory = Path.Combine(_factory.TestProjectRoot, "models", Model, "staging", id);
        File.ReadAllBytes(Path.Combine(directory, "model.onnx")).Should().Equal(graph);
    }

    /// <summary>The answer's field names are a contract a consumer's code reads; a verification is there only when asked for.</summary>
    [Fact]
    public async Task TheAnswerHasTheseFields_AndNoVerificationUnlessAskedFor()
    {
        var (id, _) = await TrainAsync();

        var answer = JsonDocument.Parse(await (await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null)).Content.ReadAsStringAsync()).RootElement;

        answer.EnumerateObject().Select(p => p.Name).Should().Equal(
            "modelName", "experimentId", "task", "onnxSha256", "manifestSha256", "sourceModelSha256", "onnxUrl", "manifestUrl", "portable");
    }

    [Fact]
    public async Task Annotations_AreInTheManifestTheShaCovers_AndTooManyAreRefused()
    {
        var (id, _) = await TrainAsync();
        var plain = JsonDocument.Parse(await (await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null)).Content.ReadAsStringAsync()).RootElement;

        var response = await _client.PostAsJsonAsync($"/experiments/{id}/export/onnx?name={Model}",
            new { annotations = new Dictionary<string, string> { ["prep"] = "v2", ["build"] = "7" } });

        response.StatusCode.Should().Be(HttpStatusCode.OK, await response.Content.ReadAsStringAsync());
        var answer = JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;
        var manifestBytes = await _client.GetByteArrayAsync(answer.GetProperty("manifestUrl").GetString());
        var manifest = OnnxManifest.FromJson(Encoding.UTF8.GetString(manifestBytes));
        manifest.Annotations.Should().Equal(new SortedDictionary<string, string> { ["build"] = "7", ["prep"] = "v2" });
        Sha256(manifestBytes).Should().Be(answer.GetProperty("manifestSha256").GetString());
        answer.GetProperty("manifestSha256").GetString().Should().NotBe(plain.GetProperty("manifestSha256").GetString(),
            "what is signed over the manifest covers the annotations");
        answer.GetProperty("onnxSha256").GetString().Should().Be(plain.GetProperty("onnxSha256").GetString(), "the graph is the model's, whatever is written beside it");

        var tooMany = Enumerable.Range(0, OnnxAnnotations.MaxPairs + 1).ToDictionary(i => $"k{i}", _ => "v");
        var refused = await _client.PostAsJsonAsync($"/experiments/{id}/export/onnx?name={Model}", new { annotations = tooMany });
        refused.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await refused.Content.ReadAsStringAsync()).Should().Contain("at most");
    }

    [Fact]
    public async Task TheDefaultModelsUrls_NeedNoNameQuery_AndOthersCarryIt()
    {
        var (id, _) = await TrainAsync();

        var answer = JsonDocument.Parse(await (await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null)).Content.ReadAsStringAsync()).RootElement;

        answer.GetProperty("onnxUrl").GetString().Should().Be($"/experiments/{id}/export/onnx/model.onnx?name={Model}");
    }

    [Fact]
    public async Task AnExperimentThatIsNotThere_IsNotFound_AndSaysWhereToLook()
    {
        var response = await _client.PostAsync("/experiments/exp-999/export/onnx?name=nomodel", null);

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        (await response.Content.ReadAsStringAsync()).Should().Contain("GET /experiments");
    }

    [Fact]
    public async Task TheTrainingFileGone_IsRefusedWithTheFieldToPass_AndANamedFileExports()
    {
        var (id, data) = await TrainAsync();
        var kept = data + ".kept";
        File.Move(data, kept);
        try
        {
            var refused = await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null);

            refused.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            var text = await refused.Content.ReadAsStringAsync();
            text.Should().Contain("\\\"data\\\"", "the refusal names the request's own field, not a command-line option")
                .And.NotContain("--data").And.NotContain(_factory.TestProjectRoot, "a path is the project's, not the host's");

            var named = Path.Combine(_factory.TestProjectRoot, "datasets", "export-again.csv");
            File.Copy(kept, named, overwrite: true);
            var ok = await _client.PostAsJsonAsync($"/experiments/{id}/export/onnx?name={Model}", new { data = "datasets/export-again.csv" });
            ok.StatusCode.Should().Be(HttpStatusCode.OK, await ok.Content.ReadAsStringAsync());
        }
        finally
        {
            File.Move(kept, data, overwrite: true);
        }
    }

    [Fact]
    public async Task ADataFileOutsideTheProject_IsRefused()
    {
        var (id, _) = await TrainAsync();
        var outside = Path.Combine(Path.GetTempPath(), $"outside-{Guid.NewGuid():N}.csv");
        File.WriteAllText(outside, Csv());
        try
        {
            var response = await _client.PostAsJsonAsync($"/experiments/{id}/export/onnx?name={Model}", new { data = outside });

            response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
            (await response.Content.ReadAsStringAsync()).Should().Contain("within the project directory");
        }
        finally
        {
            File.Delete(outside);
        }
    }

    [Fact]
    public async Task ABodyFieldTheExportDoesNotKnow_IsRefused_NotIgnored()
    {
        var (id, _) = await TrainAsync();

        var response = await _client.PostAsJsonAsync($"/experiments/{id}/export/onnx?name={Model}", new { outputDirectory = "/tmp/elsewhere" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest, "a caller names an experiment, not a place on the server");
    }

    [Fact]
    public async Task TwoExportsAtOnce_BothAnswer_AndLeaveAGraphTheManifestNames()
    {
        var (id, _) = await TrainAsync();

        var answers = await Task.WhenAll(Enumerable.Range(0, 3).Select(_ => _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null)));

        answers.Should().OnlyContain(a => a.StatusCode == HttpStatusCode.OK);
        var directory = Path.Combine(_factory.TestProjectRoot, "models", Model, "staging", id);
        var manifest = OnnxManifest.FromJson(File.ReadAllText(Path.Combine(directory, "model.onnx.json")));
        manifest.OnnxSha256.Should().Be(Sha256(File.ReadAllBytes(Path.Combine(directory, "model.onnx"))));
        Directory.GetFiles(directory, "*.partial").Should().BeEmpty("a half-written graph is never left behind");
    }

    [Theory]
    [InlineData("model.zip")]
    [InlineData("config.json")]
    [InlineData("..%2Fmetadata.json")]
    public async Task OnlyTheTwoExportFiles_AreServed(string file)
    {
        var (id, _) = await TrainAsync();
        await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", null);

        var response = await _client.GetAsync($"/experiments/{id}/export/onnx/{file}?name={Model}");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task AnExperimentNotYetExported_HasNoFileToServe_AndSaysHowToMakeOne()
    {
        var (id, _) = await TrainAsync();
        var directory = Path.Combine(_factory.TestProjectRoot, "models", Model, "staging", id);
        foreach (var f in new[] { "model.onnx", "model.onnx.json" })
            File.Delete(Path.Combine(directory, f));

        var response = await _client.GetAsync($"/experiments/{id}/export/onnx/model.onnx?name={Model}");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
        (await response.Content.ReadAsStringAsync()).Should().Contain($"POST /experiments/{id}/export/onnx");
    }

    /// <summary>
    /// This fixture signs every caller in as an admin, so the role is read from the route's own metadata: the writer wants
    /// the Admin policy (as /promote and /train do), the reader of the files only a signed-in caller.
    /// </summary>
    [Fact]
    public void ExportingNeedsTheAdminRole_ReadingTheFilesDoesNot()
    {
        var routes = _factory.Services.GetRequiredService<Microsoft.AspNetCore.Routing.EndpointDataSource>().Endpoints
            .OfType<Microsoft.AspNetCore.Routing.RouteEndpoint>().ToList();
        string? PolicyOf(string pattern, string method) => routes
            .Single(e => e.RoutePattern.RawText == pattern
                         && e.Metadata.GetMetadata<Microsoft.AspNetCore.Routing.HttpMethodMetadata>()!.HttpMethods.Contains(method))
            .Metadata.GetMetadata<Microsoft.AspNetCore.Authorization.IAuthorizeData>()?.Policy;

        PolicyOf("/experiments/{id}/export/onnx", "POST").Should().Be("Admin");
        PolicyOf("/experiments/{id}/export/onnx/{file}", "GET").Should().BeNull("any signed-in caller reads the files");
    }
}
