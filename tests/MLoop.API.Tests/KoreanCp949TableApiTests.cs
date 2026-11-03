using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>
/// The usual Korean spreadsheet export - CP949, Korean column names and class words, a dataframe's index in front, a
/// header cell broken over two lines - must be read by the server's table entrances as the table it holds, the way the
/// command line reads it: <c>POST /train</c> takes the file, <c>POST /predict</c> takes rows keyed by the names the
/// model read, <c>POST /evaluate</c> takes a held-out file in the same shape.
/// </summary>
public class KoreanCp949TableApiTests : IClassFixture<TestWebApplicationFactory>
{
    private const string Model = "korean";
    private static readonly string[] HeaderCells = ["", "엑스", "제트", "\"더블\n유\"", "결과"];
    private static readonly string[] Classes = ["예", "아니오"];

    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public KoreanCp949TableApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    /// <summary>x decides the answer (with some noise); the other two are noise. The index is the row number.</summary>
    internal static string Rows(int rows, int seed, bool labelled)
    {
        var csv = new StringBuilder(string.Join(',', labelled ? HeaderCells : HeaderCells.Take(HeaderCells.Length - 1)) + "\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var yes = rng.NextDouble() < (x > 0.5 ? 0.92 : 0.08);
            csv.Append(CultureInfo.InvariantCulture, $"{i},{x:F4},{rng.NextDouble():F4},{rng.NextDouble():F4}");
            if (labelled) csv.Append(',').Append(yes ? Classes[0] : Classes[1]);
            csv.Append('\n');
        }
        return csv.ToString();
    }

    private static Encoding Cp949()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        return Encoding.GetEncoding(949);
    }

    private static async Task<JsonElement> JsonOf(HttpResponseMessage response) =>
        JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.Clone();

    /// <summary>Trains the Korean model once through the server (the class fixture is shared by the tests of this class).</summary>
    private async Task TrainedAsync()
    {
        var train = Path.Combine(_factory.TestProjectRoot, "datasets", "korean-train.csv");
        if (File.Exists(train)) return;
        await File.WriteAllTextAsync(train, Rows(300, 1, labelled: true), Cp949());
        await File.WriteAllTextAsync(Path.Combine(_factory.TestProjectRoot, "datasets", "korean-test.csv"), Rows(150, 2, labelled: true), Cp949());

        var started = await _client.PostAsJsonAsync("/train", new
        {
            dataFile = "datasets/korean-train.csv",
            labelColumn = "결과",
            task = "binary-classification",
            name = Model,
            timeLimitSeconds = 60,
            maxTrials = 10,
        });
        started.StatusCode.Should().Be(HttpStatusCode.Accepted, await started.Content.ReadAsStringAsync());
        var jobId = (await JsonOf(started)).GetProperty("jobId").GetString();

        // the job is background work: wait for it, with a bound that a starved machine does not trip
        JsonElement job = default;
        for (var i = 0; i < 600; i++)
        {
            job = await JsonOf(await _client.GetAsync($"/jobs/{jobId}"));
            var status = job.GetProperty("status");
            var text = status.ValueKind == JsonValueKind.String ? status.GetString()! : status.GetRawText();
            if (text.Equals("Completed", StringComparison.OrdinalIgnoreCase) || text == "2") break;
            text.Should().NotBe("Failed", job.ToString());
            if (text == "3") throw new Xunit.Sdk.XunitException(job.ToString());
            await Task.Delay(500);
        }
        job.GetProperty("promoted").GetBoolean().Should().BeTrue(job.ToString());
    }

    [Fact]
    public async Task ATableInCp949WithKoreanNamesAnIndexAndABrokenHeaderCell_TrainsOverTheServer_AndAnswersRowsKeyedByTheNamesItRead()
    {
        await TrainedAsync();

        var response = await _client.PostAsJsonAsync($"/predict?name={Model}", new object[]
        {
            new Dictionary<string, object> { ["엑스"] = 0.95, ["제트"] = 0.4, ["더블 유"] = 0.5 },
            new Dictionary<string, object> { ["엑스"] = 0.05, ["제트"] = 0.4, ["더블 유"] = 0.5 },
        });

        response.StatusCode.Should().Be(HttpStatusCode.OK, await response.Content.ReadAsStringAsync());
        var body = await JsonOf(response);
        body.GetProperty("count").GetInt32().Should().Be(2);
        var answers = body.GetProperty("predictions").EnumerateArray().Select(p => p.GetProperty("predictedLabel").GetString()).ToArray();
        answers.Should().Equal(Classes[0], Classes[1]);
        body.TryGetProperty("warnings", out var warnings).Should().BeFalse($"nothing to warn about, got {warnings}");
    }

    [Fact]
    public async Task AHeldOutFileInTheSameShape_IsEvaluatedAsTheTableItHolds()
    {
        await TrainedAsync();

        var response = await _client.PostAsJsonAsync("/evaluate", new { name = Model, testDataPath = "datasets/korean-test.csv" });

        response.StatusCode.Should().Be(HttpStatusCode.OK, await response.Content.ReadAsStringAsync());
        var body = await JsonOf(response);
        body.GetProperty("metrics").GetProperty("accuracy").GetDouble().Should().BeGreaterThan(0.7, body.ToString());
    }
}
