using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /train</c> reads its input as <c>mloop train</c> does: a directory of images for the image tasks (named, or
/// the task's place under <c>datasets/</c>), a file for a table — and a request no task could train is refused with a
/// 400 that says why, never a 500 or a job that fails minutes later.
/// </summary>
public class TrainInputResolutionApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public TrainInputResolutionApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private string Dir(params string[] parts)
    {
        var path = Path.Combine([_factory.TestProjectRoot, .. parts]);
        Directory.CreateDirectory(path);
        return path;
    }

    private static async Task<string> BodyOf(HttpResponseMessage response) => await response.Content.ReadAsStringAsync();

    [Fact]
    public async Task AnImageTaskWithNoDataFileAndNoDatasetDirectory_IsRefusedWithWhereToPutOne_NotA500()
    {
        // datasets/ exists but holds no images/ — and the convention falls back to datasets/ itself, which exists,
        // so name a project where even that is absent by pointing at a directory that is not there.
        var response = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/nowhere", labelColumn = "label", task = "image-classification", name = "img-missing" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await BodyOf(response);
        body.Should().Contain("Image dataset directory not found").And.Contain("datasets/images").And.NotContain("ArgumentNullException");
    }

    [Fact]
    public async Task AnImageTaskWithNoDataFile_TrainsOnTheTasksPlaceUnderDatasets()
    {
        var images = Dir("datasets", "images");
        Dir("datasets", "images", "cat");
        Dir("datasets", "images", "dog");

        var response = await _client.PostAsJsonAsync("/train", new { labelColumn = "label", task = "image-classification", name = "img-default", promote = false });

        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await BodyOf(response));
        var jobId = JsonDocument.Parse(await BodyOf(response)).RootElement.GetProperty("jobId").GetString();
        var job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
        // The job answers in the project's paths (relative), and it resolved them against the project, not the server's working directory.
        job.GetProperty("dataFile").GetString()!.Replace('\\', '/').Should().Be("datasets/images");
    }

    [Fact]
    public async Task AnImageTaskNamingADirectory_TrainsOnIt_AndAFileInsteadIsRefused()
    {
        var folder = Dir("datasets", "named-images");
        Dir("datasets", "named-images", "a");
        var coco = Dir("datasets", "coco-named");
        File.WriteAllText(Path.Combine(coco, "annotations.json"), "{}");

        var directory = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/named-images", labelColumn = "label", task = "image-classification", name = "img-named", promote = false });
        directory.StatusCode.Should().Be(HttpStatusCode.Accepted, await BodyOf(directory));
        var jobId = JsonDocument.Parse(await BodyOf(directory)).RootElement.GetProperty("jobId").GetString();
        var job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
        job.GetProperty("dataFile").GetString()!.Replace('\\', '/').Should().Be("datasets/named-images");

        var file = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/coco-named/annotations.json", labelColumn = "label", task = "object-detection", name = "od-file" });
        file.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(file)).Should().Contain("is a file").And.Contain("directory");
    }

    [Fact]
    public async Task ATableTaskGivenADirectory_OrNoDataFile_OrAMissingOne_IsRefusedEachWithItsOwnReason()
    {
        Dir("datasets", "a-folder");
        File.WriteAllText(Path.Combine(_factory.TestProjectRoot, "datasets", "t.csv"), "x,y\n1,a\n2,b\n");

        var folder = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/a-folder", labelColumn = "y", task = "binary-classification", name = "tab-dir" });
        var none = await _client.PostAsJsonAsync("/train", new { labelColumn = "y", task = "binary-classification", name = "tab-none" });
        var missing = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/absent.csv", labelColumn = "y", task = "binary-classification", name = "tab-absent" });

        folder.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(folder)).Should().Contain("is a directory");
        none.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(none)).Should().Contain("dataFile is required").And.NotContain("ArgumentNullException");
        missing.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(missing)).Should().Contain("Data file not found");
    }

    [Fact]
    public async Task ATaskNoToolKnows_IsRefusedAtTheDoor_AsTheCommandRefusesIt()
    {
        File.WriteAllText(Path.Combine(_factory.TestProjectRoot, "datasets", "n.csv"), "x,y\n1,a\n2,b\n");

        var response = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/n.csv", labelColumn = "y", task = "nonsense", name = "nonsense" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(response)).Should().Contain("Invalid task type 'nonsense'").And.Contain("Valid options");
    }

    [Fact]
    public async Task OptionsThatMeanATable_AreRefusedForAnImageTask_NotIgnored()
    {
        Dir("datasets", "opt-images");
        File.WriteAllText(Path.Combine(_factory.TestProjectRoot, "datasets", "extra.csv"), "x\n1\n");

        var merged = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/opt-images",
            labelColumn = "label",
            task = "image-classification",
            name = "img-merge",
            additionalDataFiles = new[] { "datasets/extra.csv" },
        });
        var sampled = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/opt-images",
            labelColumn = "label",
            task = "image-classification",
            name = "img-rows",
            maxRows = 10,
        });

        merged.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(merged)).Should().Contain("additionalDataFiles");
        sampled.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await BodyOf(sampled)).Should().Contain("maxRows");
    }

    [Fact]
    public async Task AnImageJob_ReachesTheImageTrainer_NotTheTableReader()
    {
        // A COCO folder with no images: the trainer's own loader refuses it. What matters is who refuses — the job
        // must not have tried to read the folder as a table (merge, sample, class counts) on the way.
        var coco = Dir("datasets", "coco-probe");
        File.WriteAllText(Path.Combine(coco, "annotations.json"), "{\"images\":[],\"annotations\":[],\"categories\":[]}");

        var response = await _client.PostAsJsonAsync("/train", new { dataFile = "datasets/coco-probe", labelColumn = "label", task = "object-detection", name = "od-probe", promote = false });
        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await BodyOf(response));
        var jobId = JsonDocument.Parse(await BodyOf(response)).RootElement.GetProperty("jobId").GetString();

        string? status = null, message = null;
        for (var i = 0; i < 120 && status is not ("Completed" or "Failed"); i++)
        {
            await Task.Delay(500);
            var job = JsonDocument.Parse(await _client.GetStringAsync($"/jobs/{jobId}")).RootElement;
            status = job.GetProperty("status").GetString();
            message = job.TryGetProperty("statusMessage", out var m) ? m.GetString() : null;
        }

        status.Should().Be("Failed");
        message.Should().NotBeNull().And.NotContain("Label column").And.NotContain("rows without a label").And.NotContain("class distribution");
        message.Should().Contain("COCO annotations");
    }
}
