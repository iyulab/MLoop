using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using Microsoft.ML;

namespace MLoop.API.Tests;

/// <summary>
/// The response shape <c>docs/PREDICT-RESPONSE.md</c> promises to consumers, asserted against a real
/// scored model over the real endpoint. The document is the contract; these are the sentences of it
/// that a refactor could break silently — the envelope's field names, which row fields a multiclass
/// prediction carries, and the fact that its probability keys are the trained class names paired with
/// the slots they belong to.
/// </summary>
public class PredictResponseContractTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public PredictResponseContractTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    private sealed class LabeledPoint
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public string Label { get; set; } = "";
    }

    /// <summary>
    /// Trains a real multiclass model and seeds it as production, mirroring what `mloop train` +
    /// `mloop promote` leave on disk. The classes occur in the order dog, cat, bird — deliberately
    /// not alphabetical, since the trainer keys them by first occurrence and a vocabulary that
    /// happened to be sorted would otherwise pass while paired with the wrong slots.
    /// </summary>
    private string SeedMulticlassProductionModel()
    {
        const string modelName = "mc";
        var modelsDir = Path.Combine(_factory.TestProjectRoot, "models", modelName);
        var stagingDir = Path.Combine(modelsDir, "staging", "exp-001");
        var productionDir = Path.Combine(modelsDir, "production");
        Directory.CreateDirectory(stagingDir);
        Directory.CreateDirectory(productionDir);

        var ml = new MLContext(seed: 42);
        var points = new List<LabeledPoint>();
        foreach (var (label, cx, cy) in new[] { ("dog", 9f, 9f), ("cat", 5f, 5f), ("bird", 1f, 1f) })
        {
            for (int i = 0; i < 20; i++)
            {
                float jitter = (i % 5 - 2) * 0.1f;
                points.Add(new LabeledPoint { X1 = cx + jitter, X2 = cy - jitter, Label = label });
            }
        }

        var data = ml.Data.LoadFromEnumerable(points);
        var keyed = ml.Transforms.Conversion.MapValueToKey("Label", "Label").Fit(data).Transform(data);
        var featurized = ml.Transforms.Concatenate("Features", "X1", "X2").Fit(keyed).Transform(keyed);
        var model = ml.MulticlassClassification.Trainers
            .SingleThreadSdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
            .Fit(featurized);

        ml.Model.Save(model, featurized.Schema, Path.Combine(productionDir, "model.zip"));

        var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

        File.WriteAllText(Path.Combine(stagingDir, "config.json"), JsonSerializer.Serialize(new
        {
            dataFile = Path.Combine(stagingDir, "train.csv"),
            labelColumn = "Label",
            timeLimitSeconds = 10,
            metric = "MicroAccuracy",
            testSplit = 0.2,
            inputSchema = new
            {
                columns = new[]
                {
                    new { name = "X1", dataType = "Numeric", purpose = "Feature" },
                    new { name = "X2", dataType = "Numeric", purpose = "Feature" },
                    new { name = "Label", dataType = "Categorical", purpose = "Label" },
                },
                capturedAt = DateTime.UtcNow,
            },
        }, json));

        // The row-based /predict path loads the *staging* experiment for its InputSchema, so the
        // experiment's own metadata.json has to be there too — the forecasting endpoint never needed
        // it because a horizon replay bypasses schema loading entirely.
        File.WriteAllText(Path.Combine(stagingDir, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-001",
            timestamp = DateTime.UtcNow,
            status = "Completed",
            task = "multiclass-classification",
            result = new { bestTrainer = "SdcaMaximumEntropy", trainingTimeSeconds = 1.0 },
        }, json));

        File.WriteAllText(Path.Combine(stagingDir, "metrics.json"), JsonSerializer.Serialize(
            new Dictionary<string, double> { ["MicroAccuracy"] = 1.0 }, json));

        File.WriteAllText(Path.Combine(productionDir, "metadata.json"), JsonSerializer.Serialize(new
        {
            modelName,
            experimentId = "exp-001",
            promotedAt = DateTime.UtcNow,
            metrics = new Dictionary<string, double> { ["MicroAccuracy"] = 1.0 },
            task = "multiclass-classification",
            bestTrainer = "SdcaMaximumEntropy",
            labelColumn = "Label",
        }, json));

        return modelName;
    }

    [Fact]
    public async Task Served_predictions_are_logged_and_close_the_feedback_loop()
    {
        // Served traffic used to reach none of logs, feedback or triggers: `/predict` never logged,
        // nothing showed the id feedback needs, and an unknown id came back as a 500.
        var modelName = SeedMulticlassProductionModel();

        var predict = await _client.PostAsJsonAsync($"/predict?name={modelName}", new[] { new { X1 = 1.0, X2 = 1.0 } });
        predict.StatusCode.Should().Be(HttpStatusCode.OK, await predict.Content.ReadAsStringAsync());
        var row = JsonSerializer.Deserialize<JsonElement>(await predict.Content.ReadAsStringAsync())
            .GetProperty("predictions")[0];
        var id = row.GetProperty("predictionId").GetString();
        id.Should().MatchRegex("^[0-9a-f]{16}$");

        var logs = JsonSerializer.Deserialize<JsonElement>(await _client.GetStringAsync($"/logs?name={modelName}"));
        logs.GetProperty("logs").EnumerateArray().Select(l => l.GetProperty("id").GetString())
            .Should().Contain(id, "the id a response hands out is the id the log holds");

        var first = await _client.PostAsJsonAsync("/feedback", new { predictionId = id, actualValue = "bird" });
        first.StatusCode.Should().Be(HttpStatusCode.OK, await first.Content.ReadAsStringAsync());

        // Recording it again is a correction, not an error; the metrics count the prediction once.
        var again = await _client.PostAsJsonAsync("/feedback", new { predictionId = id, actualValue = "bird" });
        again.StatusCode.Should().Be(HttpStatusCode.OK);

        var unknown = await _client.PostAsJsonAsync("/feedback", new { predictionId = "0000000000000000", actualValue = "bird" });
        unknown.StatusCode.Should().Be(HttpStatusCode.NotFound);

        var empty = await _client.PostAsJsonAsync("/feedback", new { });
        empty.StatusCode.Should().Be(HttpStatusCode.BadRequest);

        // A weight travels with the entry (a row reviewed with probability 1/4 counts 4), and one that
        // cannot weight a row is the caller's mistake, not the server's.
        var weighted = await _client.PostAsJsonAsync("/feedback", new { predictionId = id, actualValue = "bird", source = "reviewer", weight = 4.0 });
        weighted.StatusCode.Should().Be(HttpStatusCode.OK, await weighted.Content.ReadAsStringAsync());
        var negative = await _client.PostAsJsonAsync("/feedback", new { predictionId = id, actualValue = "bird", weight = -1.0 });
        negative.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var listed = JsonSerializer.Deserialize<JsonElement>(await _client.GetStringAsync($"/feedback?name={modelName}"));
        listed.GetProperty("feedback").EnumerateArray()
            .Where(f => f.TryGetProperty("weight", out var w) && w.ValueKind == JsonValueKind.Number)
            .Select(f => f.GetProperty("weight").GetDouble())
            .Should().Equal(4.0);

        // The feedback becomes a retraining set over HTTP, as `mloop sample create --model`: the actual value
        // under the model's label, and the weight the feedback carried.
        var sample = await _client.PostAsJsonAsync("/samples", new { name = modelName, rows = 10, strategy = "feedback-priority" });
        sample.StatusCode.Should().Be(HttpStatusCode.OK, await sample.Content.ReadAsStringAsync());
        var sampled = JsonSerializer.Deserialize<JsonElement>(await sample.Content.ReadAsStringAsync());
        sampled.GetProperty("labeledCount").GetInt32().Should().Be(1);
        sampled.GetProperty("labelColumn").GetString().Should().Be("Label");
        sampled.GetProperty("weightColumn").GetString().Should().Be("weight");
        var sampleFile = File.ReadAllLines(Path.Combine(_factory.TestProjectRoot, sampled.GetProperty("path").GetString()!));
        // The client sent x1/x2 (camelCase JSON); the file is in the model's spelling, so it merges with its training data.
        sampleFile[0].Should().Be("X1,X2,Label,weight");
        sampleFile[1].Should().EndWith(",bird,4");

        (await _client.PostAsJsonAsync("/samples", new { name = modelName, strategy = "sideways" }))
            .StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await _client.PostAsJsonAsync("/samples", new { name = "never-predicted" }))
            .StatusCode.Should().BeOneOf(HttpStatusCode.BadRequest, HttpStatusCode.NotFound);

        var trigger = JsonSerializer.Deserialize<JsonElement>(await _client.GetStringAsync($"/trigger?name={modelName}"));
        trigger.GetProperty("conditions").EnumerateArray().Select(c => c.GetProperty("type").GetString())
            .Should().BeEquivalentTo(MLoop.Ops.Services.RetrainingDefaults.All.Select(c => c.Type.ToString()),
                "the API checks the same default conditions as `mloop trigger check`");
    }

    /// <summary>
    /// A shadow prediction over HTTP, as `mloop predict --experiment`: a candidate in staging answers,
    /// the response names it, its logged rows carry its id, and production stays where it was.
    /// </summary>
    [Fact]
    public async Task Predict_WithAnExperiment_AnswersWithTheCandidate_AndLeavesProductionAlone()
    {
        var modelName = SeedMulticlassProductionModel();
        var modelsDir = Path.Combine(_factory.TestProjectRoot, "models", modelName);
        var candidate = Path.Combine(modelsDir, "staging", "exp-002");
        Directory.CreateDirectory(candidate);
        foreach (var file in new[] { "config.json", "metrics.json" })
            File.Copy(Path.Combine(modelsDir, "staging", "exp-001", file), Path.Combine(candidate, file), overwrite: true);
        File.WriteAllText(Path.Combine(candidate, "metadata.json"),
            File.ReadAllText(Path.Combine(modelsDir, "staging", "exp-001", "metadata.json")).Replace("exp-001", "exp-002"));
        File.Copy(Path.Combine(modelsDir, "production", "model.zip"), Path.Combine(candidate, "model.zip"), overwrite: true);

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}&experiment=exp-002", new[] { new { X1 = 9.0, X2 = 9.0 } });
        var raw = await response.Content.ReadAsStringAsync();
        response.StatusCode.Should().Be(HttpStatusCode.OK, raw);
        var body = JsonSerializer.Deserialize<JsonElement>(raw);
        body.GetProperty("experimentId").GetString().Should().Be("exp-002");
        var id = body.GetProperty("predictions")[0].GetProperty("predictionId").GetString();

        var logs = JsonSerializer.Deserialize<JsonElement>(await _client.GetStringAsync($"/logs?name={modelName}"));
        logs.GetProperty("logs").EnumerateArray().Single(l => l.GetProperty("id").GetString() == id)
            .GetProperty("experimentId").GetString().Should().Be("exp-002");

        var production = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(Path.Combine(modelsDir, "production", "metadata.json")));
        production.GetProperty("experimentId").GetString().Should().Be("exp-001");

        var unknown = await _client.PostAsJsonAsync($"/predict?name={modelName}&experiment=exp-099", new[] { new { X1 = 1.0, X2 = 1.0 } });
        unknown.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task Predict_Multiclass_MatchesTheDocumentedResponseContract()
    {
        var modelName = SeedMulticlassProductionModel();

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new[]
        {
            new { X1 = 1.0, X2 = 1.0 },
            new { X1 = 9.0, X2 = 9.0 },
        });

        var raw = await response.Content.ReadAsStringAsync();
        response.StatusCode.Should().Be(HttpStatusCode.OK, "response body was: {0}", raw);
        var body = JsonSerializer.Deserialize<JsonElement>(raw);

        // Envelope — the HTTP response names the experiment as well as the model, and the CLI's
        // `model` is `modelName` here. A consumer written against one and pointed at the other is
        // exactly what the document exists to prevent.
        body.GetProperty("modelName").GetString().Should().Be(modelName);
        body.GetProperty("experimentId").GetString().Should().Be("exp-001");
        body.TryGetProperty("predictedAt", out _).Should().BeTrue();
        body.GetProperty("task").GetString().Should().Be("multiclass-classification");
        body.GetProperty("count").GetInt32().Should().Be(2);

        // A field without a value is absent, here as on the CLI. The server used to write it as
        // null, which gave the same row two shapes depending on the surface asked; the document now
        // promises one shape, and this is the sentence that would break first if the server's
        // serializer options were ever reset to the framework default.
        if (body.TryGetProperty("warnings", out var warnings))
            warnings.ValueKind.Should().NotBe(JsonValueKind.Null, "an empty warnings field is omitted, never written as null");

        var predictions = body.GetProperty("predictions");
        predictions.GetArrayLength().Should().Be(2);
        // The body's keys are camelCase (`x1`) and the schema's columns are not (`X1`): the values reach
        // the model all the same — (1, 1) is a bird and (9, 9) a dog. They used to be dropped and scored
        // as zeros, a bird twice, and the assertion below accepted it.
        predictions.EnumerateArray().Select(p => p.GetProperty("predictedLabel").GetString())
            .Should().Equal("bird", "dog");

        foreach (var row in predictions.EnumerateArray())
        {
            var label = row.GetProperty("predictedLabel").GetString();
            label.Should().BeOneOf("cat", "dog", "bird");

            var probabilities = row.GetProperty("probabilities");
            probabilities.EnumerateObject().Select(p => p.Name)
                .Should().BeEquivalentTo(new[] { "cat", "dog", "bird" });

            // The documented join: a row's own label is the key with the largest probability. Key
            // presence alone would hold under any permutation of the vocabulary.
            var winner = probabilities.EnumerateObject()
                .OrderByDescending(p => p.Value.GetDouble()).First().Name;
            winner.Should().Be(label);

            row.GetProperty("confidence").GetDouble().Should().BeInRange(0.0, 1.0);

            // Multiclass carries no scalar `score`, so the key is not there — the same row the CLI
            // returns. Before 0.32.0 the key was present and null over HTTP, and a consumer testing
            // key presence concluded a multiclass row had a score.
            row.TryGetProperty("score", out _).Should().BeFalse("a field without a value is absent, not null");
            row.TryGetProperty("clusterId", out _).Should().BeFalse("a field without a value is absent, not null");
            row.TryGetProperty("distances", out _).Should().BeFalse("a field without a value is absent, not null");
        }
    }

    // POST /predict takes an image model's images by their contents, as POST /embed does. What it refuses, it refuses
    // before any model work: too many, one that cannot be decoded (named by position), a body that is not JSON — and
    // images sent to a model that does not read images.
    private static string Pattern => Convert.ToBase64String(File.ReadAllBytes(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding", "pattern.png")));

    [Fact]
    public async Task Images_sent_to_a_model_that_does_not_read_them_are_refused()
    {
        var modelName = SeedMulticlassProductionModel();

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new { images = new[] { Pattern } });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("image-classification");
    }

    [Fact]
    public async Task More_images_than_one_request_takes_is_413()
    {
        var modelName = SeedMulticlassProductionModel();

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new { images = Enumerable.Repeat(Pattern, 65) });

        response.StatusCode.Should().Be(HttpStatusCode.RequestEntityTooLarge);
    }

    [Fact]
    public async Task An_image_that_cannot_be_decoded_is_named_by_its_position()
    {
        var modelName = SeedMulticlassProductionModel();

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}",
            new { images = new[] { Pattern, Convert.ToBase64String("not an image"u8.ToArray()) } });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync()).GetProperty("index").GetInt32().Should().Be(1);
    }

    [Fact]
    public async Task A_body_that_is_not_json_is_400()
    {
        var modelName = SeedMulticlassProductionModel();

        var response = await _client.PostAsync($"/predict?name={modelName}",
            new StringContent("{ not json", System.Text.Encoding.UTF8, "application/json"));

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Images_in_a_multipart_form_are_read_like_base64_ones()
    {
        var modelName = SeedMulticlassProductionModel();
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(Convert.FromBase64String(Pattern)), "image", "pattern.png");

        var response = await _client.PostAsync($"/predict?name={modelName}", form);

        // Read, decoded and then refused for the model's task — the same answer as the JSON form.
        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        (await response.Content.ReadAsStringAsync()).Should().Contain("image-classification");
    }

}
