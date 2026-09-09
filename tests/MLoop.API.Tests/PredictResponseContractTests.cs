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
            .SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
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

        // The server writes nulls rather than omitting them, which is the half of the shape that
        // differs from the CLI's; the document tells consumers to treat the two as the same.
        body.TryGetProperty("warnings", out var warnings).Should().BeTrue();
        warnings.ValueKind.Should().Be(JsonValueKind.Null);

        var predictions = body.GetProperty("predictions");
        predictions.GetArrayLength().Should().Be(2);

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

            // Multiclass carries no scalar `score` — but over HTTP the key is still *there*, holding
            // null, because the server writes nulls where the CLI omits them. This is the distinction
            // the document has to state precisely: a consumer testing key presence would conclude this
            // multiclass row has a score, and a consumer testing for a value would not.
            row.GetProperty("score").ValueKind.Should().Be(JsonValueKind.Null);
            row.GetProperty("clusterId").ValueKind.Should().Be(JsonValueKind.Null);
        }
    }
}
