using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// A job whose data had to be prepared before it was trained on — here a table written with decomposed Hangul, which is
/// read through a composed copy, and then balanced — is exported without naming a data file, as any other job is. The
/// prepared copy lies outside the project (or is gone by then), so the exporter reads the input columns from the file the
/// request named.
/// </summary>
public class ExportAfterPreparationApiTests : IClassFixture<TestWebApplicationFactory>
{
    private const string Model = "prepared";

    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ExportAfterPreparationApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    /// <summary>One in six rows is "true". <paramref name="form"/> is how the category values are written: decomposed (NFD), as
    /// a file written on macOS often is, or composed (NFC), which is read as it is.</summary>
    private static string Csv(NormalizationForm form)
    {
        var colours = new[] { "빨강", "파랑", "초록" }.Select(c => c.Normalize(form)).ToArray();
        var csv = new StringBuilder("x,colour,y\n");
        var rng = new Random(11);
        for (var i = 0; i < 240; i++)
        {
            var positive = i % 6 == 0;
            var x = (positive ? 0.6 : 0.2) + rng.NextDouble() * 0.4;
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{colours[rng.Next(colours.Length)]},{(positive ? "true" : "false")}\n");
        }
        return csv.ToString();
    }

    [Fact]
    public async Task AJobTrainedOnAPreparedCopy_IsExportedWithoutNamingData()
    {
        Directory.CreateDirectory(Path.Combine(_factory.TestProjectRoot, "datasets"));
        var data = Path.Combine(_factory.TestProjectRoot, "datasets", "prepared.csv");
        File.WriteAllText(data, Csv(NormalizationForm.FormD));
        var id = await TrainAsync(
            new { dataFile = data, labelColumn = "y", task = "binary-classification", name = Model, timeLimitSeconds = 60, maxTrials = 10, portable = true, balance = "2" });

        var export = await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", content: null);

        export.StatusCode.Should().Be(HttpStatusCode.OK, await export.Content.ReadAsStringAsync());

        // The named file stands in only while it has what the model reads. Without a column it is not used, and the
        // answer is the same refusal as when there is nothing to read.
        File.WriteAllText(data, "x,y\n0.1,false\n0.9,true\n");

        var refused = await _client.PostAsync($"/experiments/{id}/export/onnx?name={Model}", content: null);

        var text = await refused.Content.ReadAsStringAsync();
        refused.StatusCode.Should().Be(HttpStatusCode.BadRequest, text);
        text.Should().Contain("\\\"data\\\"", "the refusal names the request's own field");
    }

    /// <summary>
    /// Files merged for a weighted run: the first one has no weight column (its rows count 1). When the merged copy is
    /// gone the first file stands in - the weight is not an input of the model, so its absence does not matter.
    /// </summary>
    [Fact]
    public async Task AMergedWeightedJob_IsExportedFromTheFirstFile_WhichHasNoWeightColumn()
    {
        const string name = "merged";
        var directory = Path.Combine(_factory.TestProjectRoot, "datasets");
        Directory.CreateDirectory(directory);
        var first = Path.Combine(directory, "first.csv");
        // Composed text, so that the merged file is the one trained on (decomposed text would be read through yet another copy).
        File.WriteAllText(first, Csv(NormalizationForm.FormC));
        var more = Path.Combine(directory, "more.csv");
        var lines = Csv(NormalizationForm.FormC).TrimEnd('\n').Split('\n');
        File.WriteAllText(more, string.Join('\n', lines.Take(61).Select((line, i) => line + (i == 0 ? ",w" : ",2"))) + "\n");
        var id = await TrainAsync(
            new { dataFile = first, additionalDataFiles = new[] { more }, weight = "w", labelColumn = "y", task = "binary-classification", name, timeLimitSeconds = 60, maxTrials = 10, portable = true });
        var merged = Directory.GetFiles(Path.Combine(_factory.TestProjectRoot, ".mloop", "merged"));
        merged.Should().ContainSingle("the job merged the two files into one");
        var config = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(_factory.TestProjectRoot, "models", name, "staging", id, "config.json"))).RootElement;
        Path.GetFileName(config.GetProperty("dataFile").GetString()).Should().Be(Path.GetFileName(merged[0]), "the merged file is what was trained on");
        File.Delete(merged[0]);

        var export = await _client.PostAsync($"/experiments/{id}/export/onnx?name={name}", content: null);

        export.StatusCode.Should().Be(HttpStatusCode.OK, await export.Content.ReadAsStringAsync());
    }

    private async Task<string> TrainAsync(object request)
    {
        var response = await _client.PostAsJsonAsync("/train", request);
        response.StatusCode.Should().Be(HttpStatusCode.Accepted, await response.Content.ReadAsStringAsync());
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
        return job.GetProperty("experimentId").GetString()!;
    }
}
