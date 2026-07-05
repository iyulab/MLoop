using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;

namespace MLoop.API.Tests;

/// <summary>
/// D21-A: <c>POST /predict</c> for a forecasting model is horizon-based (stateful SSA replaying its
/// training series), not row-based — accepts an optional <c>{"horizon":N}</c> body (omitted/no body
/// uses the model's trained horizon) and returns the forecast on the same <c>PredictionRow</c> schema
/// every other task uses, so structured consumers never see two different response shapes.
/// </summary>
public class ForecastingApiTests : IClassFixture<TestWebApplicationFactory>
{
    private readonly TestWebApplicationFactory _factory;
    private readonly HttpClient _client;

    public ForecastingApiTests(TestWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    /// <summary>SSA (<c>ForecastBySsa</c>) needs the MKL native library — absent on some CI runners
    /// (see <c>PredictForecastingTests.MklAvailable</c> for the same guard in the CLI test project).</summary>
    private static readonly Lazy<bool> MklAvailable = new(() =>
    {
        try
        {
            var ml = new MLContext(seed: 0);
            var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 20).Select(i => new Point { Value = i }));
            var loader = ml.Data.CreateTextLoader(new TextLoader.Options
            {
                Columns = [new TextLoader.Column("Value", DataKind.Single, 0)],
                HasHeader = true,
            });
            var pipeline = ml.Forecasting.ForecastBySsa("Forecast", "Value", 4, 8, 20, 2);
            pipeline.Fit(data).Transform(data);
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException
                                   || ex.InnerException is DllNotFoundException)
        {
            return false;
        }
    });

    private sealed class Point { public float Value { get; set; } }

    /// <summary>Trains a real SSA forecaster and seeds it as model "fc" production, mirroring what
    /// `mloop train` + `mloop promote` produce on disk.</summary>
    private string SeedForecastingProductionModel(int horizon)
    {
        var modelName = "fc";
        var modelsDir = Path.Combine(_factory.TestProjectRoot, "models", modelName);
        var stagingDir = Path.Combine(modelsDir, "staging", "exp-001");
        var productionDir = Path.Combine(modelsDir, "production");
        Directory.CreateDirectory(stagingDir);
        Directory.CreateDirectory(productionDir);

        var trainCsvPath = Path.Combine(stagingDir, "train.csv");
        var lines = new List<string> { "Value" };
        for (int i = 0; i < 120; i++)
        {
            var v = 10.0 + 0.05 * i + 3.0 * Math.Sin(2 * Math.PI * i / 12.0);
            lines.Add(v.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
        }
        File.WriteAllLines(trainCsvPath, lines);

        var mlContext = new MLContext(seed: 42);
        var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
        {
            Columns = [new TextLoader.Column("Value", DataKind.Single, 0)],
            HasHeader = true,
            Separators = [','],
        });
        var trainData = loader.Load(trainCsvPath);

        var pipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: ForecastOutput.ForecastColumnName,
            inputColumnName: "Value",
            windowSize: 12,
            seriesLength: 36,
            trainSize: 120,
            horizon: horizon,
            confidenceLowerBoundColumn: ForecastOutput.LowerBoundColumnName,
            confidenceUpperBoundColumn: ForecastOutput.UpperBoundColumnName,
            confidenceLevel: (float)ForecastOutput.ConfidenceLevel);
        var model = pipeline.Fit(trainData);

        var modelPath = Path.Combine(productionDir, "model.zip");
        mlContext.Model.Save(model, trainData.Schema, modelPath);

        File.WriteAllText(Path.Combine(stagingDir, "config.json"),
            JsonSerializer.Serialize(new { dataFile = trainCsvPath, labelColumn = "Value" }));

        File.WriteAllText(Path.Combine(productionDir, "metadata.json"),
            JsonSerializer.Serialize(new
            {
                modelName,
                experimentId = "exp-001",
                promotedAt = DateTime.UtcNow,
                metrics = new Dictionary<string, double> { ["horizon"] = horizon },
                task = "forecasting",
                bestTrainer = "SsaForecasting",
                labelColumn = "Value",
            }));

        return modelName;
    }

    [Fact]
    public async Task Predict_Forecasting_EmptyBody_UsesTrainedHorizon()
    {
        if (!MklAvailable.Value) return;

        var modelName = SeedForecastingProductionModel(horizon: 5);

        // "horizon omitted" means an empty JSON object, not a bodyless POST — /predict always
        // requires a JSON body (every other task posts a row array), and ASP.NET's minimal-API
        // JsonElement binding rejects a truly empty/no-content-type request before the handler runs.
        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new { });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());
        body.GetProperty("task").GetString().Should().Be("forecasting");
        body.GetProperty("count").GetInt32().Should().Be(5);
        var predictions = body.GetProperty("predictions");
        predictions.GetArrayLength().Should().Be(5);
        predictions[0].GetProperty("score").GetSingle().Should().NotBe(0f);
    }

    [Fact]
    public async Task Predict_Forecasting_MatchingHorizon_Succeeds()
    {
        if (!MklAvailable.Value) return;

        var modelName = SeedForecastingProductionModel(horizon: 5);

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new { horizon = 5 });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = JsonSerializer.Deserialize<JsonElement>(await response.Content.ReadAsStringAsync());
        body.GetProperty("count").GetInt32().Should().Be(5);
    }

    [Fact]
    public async Task Predict_Forecasting_MismatchedHorizon_ReturnsActionableBadRequest()
    {
        if (!MklAvailable.Value) return;

        var modelName = SeedForecastingProductionModel(horizon: 5);

        var response = await _client.PostAsJsonAsync($"/predict?name={modelName}", new { horizon = 99 });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("fixed horizon of 5");
    }
}
