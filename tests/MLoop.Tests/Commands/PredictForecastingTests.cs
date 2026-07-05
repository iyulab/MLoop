using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.CLI.Commands;
using MLoop.Core.AutoML;

namespace MLoop.Tests.Commands;

/// <summary>
/// D21 CLI-side coverage: the forecasting predict path is horizon-based (stateful SSA replaying
/// the training series). These tests pin the extracted computation helper and the --json row
/// mapping so the structured output contract (PredictionRow reuse, native band, coverage level)
/// cannot silently drift from the CSV path.
/// </summary>
[Collection("FileSystem")]
public class PredictForecastingTests : IDisposable
{
    private readonly string _tempDir;

    public PredictForecastingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"mloop-forecast-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { }
    }

    /// <summary>
    /// Trains a tiny SSA forecaster on a deterministic series, saves model.zip + the sibling
    /// config.json ComputeForecastAsync resolves the training data from, and returns the model path.
    /// </summary>
    private (string ModelPath, string TrainCsvPath) CreateForecastingFixture(int horizon = 5)
    {
        var trainCsvPath = Path.Combine(_tempDir, "train.csv");
        var lines = new List<string> { "Value" };
        for (int i = 0; i < 120; i++)
        {
            // Deterministic seasonal series: trend + sine.
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

        var modelDir = Path.Combine(_tempDir, "production");
        Directory.CreateDirectory(modelDir);
        var modelPath = Path.Combine(modelDir, "model.zip");
        mlContext.Model.Save(model, trainData.Schema, modelPath);

        // config.json next to model.zip (the experimentId == null resolution path).
        var config = System.Text.Json.JsonSerializer.Serialize(new
        {
            dataFile = trainCsvPath,
            labelColumn = "Value",
        });
        File.WriteAllText(Path.Combine(modelDir, "config.json"), config);

        return (modelPath, trainCsvPath);
    }

    [Fact]
    public async Task ComputeForecastAsync_ProducesHorizonForecastWithOrderedBounds()
    {
        var (modelPath, _) = CreateForecastingFixture(horizon: 5);

        var (forecast, error) = await PredictCommand.ComputeForecastAsync(modelPath, experimentId: null);

        Assert.Null(error);
        Assert.NotNull(forecast);
        Assert.Equal(5, forecast!.ForecastedValues.Length);
        Assert.Equal(5, forecast.LowerBound.Length);
        Assert.Equal(5, forecast.UpperBound.Length);
        for (int i = 0; i < 5; i++)
        {
            Assert.True(forecast.LowerBound[i] <= forecast.ForecastedValues[i],
                $"step {i + 1}: lower {forecast.LowerBound[i]} > forecast {forecast.ForecastedValues[i]}");
            Assert.True(forecast.ForecastedValues[i] <= forecast.UpperBound[i],
                $"step {i + 1}: forecast {forecast.ForecastedValues[i]} > upper {forecast.UpperBound[i]}");
        }
    }

    [Fact]
    public async Task ComputeForecastAsync_MissingTrainingData_ReturnsActionableError()
    {
        var (modelPath, trainCsvPath) = CreateForecastingFixture();
        File.Delete(trainCsvPath);

        var (forecast, error) = await PredictCommand.ComputeForecastAsync(modelPath, experimentId: null);

        Assert.Null(forecast);
        Assert.NotNull(error);
        Assert.Contains("training data", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BuildForecastRows_MapsStepsOntoSharedPredictionRowSchema()
    {
        var forecast = new ForecastOutput
        {
            ForecastedValues = [43.65f, 44.10f],
            LowerBound = [41.09f, 41.50f],
            UpperBound = [46.22f, 46.70f],
        };

        var rows = PredictCommand.BuildForecastRows(forecast);

        Assert.Equal(2, rows.Count);
        // Row order == step order (step = index + 1, matching the CSV Step column).
        Assert.Equal(43.65f, rows[0].Score!.Value, 3);
        Assert.Equal(41.09f, rows[0].ScoreLowerBound!.Value, 3);
        Assert.Equal(46.22f, rows[0].ScoreUpperBound!.Value, 3);
        Assert.Equal(ForecastOutput.ConfidenceLevel, rows[0].IntervalConfidence);
        Assert.Equal(44.10f, rows[1].Score!.Value, 3);
        // Fields the shared schema leaves null for forecasting stay null (omitted in JSON).
        Assert.Null(rows[0].PredictedLabel);
        Assert.Null(rows[0].Confidence);
    }

    // D23: for time-series tasks the config "label" is the monitored series value column — it must
    // NOT be stripped from the structured-predict input rows (stripping defaulted the series to zero
    // and the detector fabricated an all-normal answer). Tabular tasks keep the exclusion (leakage).
    [Theory]
    [InlineData("time-series-anomaly", null)]
    [InlineData("forecasting", null)]
    [InlineData("regression", "Temp")]
    [InlineData("binary-classification", "Temp")]
    public void LabelColumnToExcludeFromRows_KeepsSeriesValueForTimeSeries(string taskType, string? expected)
    {
        Assert.Equal(expected, PredictCommand.LabelColumnToExcludeFromRows(taskType, "Temp"));
    }

    [Fact]
    public void BuildForecastRows_MissingBounds_LeavesBoundsNull()
    {
        var forecast = new ForecastOutput { ForecastedValues = [1.0f] };

        var rows = PredictCommand.BuildForecastRows(forecast);

        Assert.Single(rows);
        Assert.Equal(1.0f, rows[0].Score!.Value, 3);
        Assert.Null(rows[0].ScoreLowerBound);
        Assert.Null(rows[0].ScoreUpperBound);
    }
}
