using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;
using MLoop.Core.Storage;

namespace MLoop.Core.Prediction;

/// <summary>
/// D21: single source for replaying an SSA forecasting model's training series and extracting its
/// horizon forecast — shared by the CLI (<c>mloop predict</c>/<c>--json</c>) and the API
/// (<c>POST /predict</c>) so the two surfaces can never drift on how the forecast is computed
/// (the same lesson as PRED-1/D-series). SSA forecasting is stateful (it forecasts a fixed horizon
/// ahead of its training series), so — unlike every other task — a row-based <c>Transform</c> extracts
/// nothing; this class replays the original training series read from the experiment's config
/// (<c>dataFile</c>) and reads the model's native forecast/confidence-band columns instead.
/// </summary>
public static class ForecastReplayService
{
    /// <summary>
    /// Runs the stateful SSA forecast: loads the model, replays the original training series
    /// (resolved from the experiment config), and extracts the horizon forecast with its native
    /// confidence bounds. Pure computation shared by the CSV, --json, and serve presenters —
    /// returns either the forecast or an actionable error message, never both.
    /// </summary>
    /// <param name="mlContext">Context used to load the model and replay the training series.</param>
    /// <param name="modelPath">Path to the saved SSA model (<c>model.zip</c>).</param>
    /// <param name="experimentId">
    /// The staging experiment id whose <c>config.json</c> carries the original <c>dataFile</c>/
    /// <c>labelColumn</c>; <c>null</c> resolves the config sitting next to <paramref name="modelPath"/>
    /// instead (e.g. a promoted model's own <c>production/config.json</c>).
    /// </param>
    /// <param name="requestedHorizon">
    /// Optional horizon override (D21-A, <c>POST /predict</c> body <c>{"horizon":N}</c>). The saved
    /// SSA model's horizon is fixed at train time (<c>variableHorizon</c> is not enabled), so a
    /// mismatched override cannot be honored — it fails fast with an actionable error naming the
    /// trained horizon rather than silently ignoring the request or truncating/padding the result.
    /// <c>null</c> (the field omitted from the request body) always uses the trained horizon.
    /// </param>
    public static async Task<(ForecastOutput? Forecast, string? Error)> ComputeForecastAsync(
        MLContext mlContext, string modelPath, string? experimentId, int? requestedHorizon = null)
    {
        var model = mlContext.Model.Load(modelPath, out var modelSchema);

        // SSA model needs data with the correct value column to transform.
        // Read config to find original training data and value column.
        // Config is in staging/{experimentId}/config.json
        var modelBaseDir = Path.GetDirectoryName(Path.GetDirectoryName(modelPath))!; // models/default/
        var configPath = experimentId != null
            ? Path.Combine(modelBaseDir, ExperimentLayout.StagingDirectory, experimentId, ExperimentLayout.ConfigFileName)
            : Path.Combine(Path.GetDirectoryName(modelPath)!, ExperimentLayout.ConfigFileName);
        string? trainDataPath = null;
        string? valueColName = null;

        if (File.Exists(configPath))
        {
            var configJson = await File.ReadAllTextAsync(configPath).ConfigureAwait(false);
            var configDoc = System.Text.Json.JsonDocument.Parse(configJson);
            trainDataPath = configDoc.RootElement.TryGetProperty("dataFile", out var df) ? df.GetString() : null;
            valueColName = configDoc.RootElement.TryGetProperty("labelColumn", out var lc) ? lc.GetString() : null;
        }

        if (string.IsNullOrEmpty(valueColName))
        {
            // Fallback: find from model schema
            valueColName = modelSchema
                .Where(c => !c.IsHidden && c.Type == NumberDataViewType.Single)
                .Select(c => c.Name)
                .FirstOrDefault(n => n != ForecastOutput.ForecastColumnName
                                  && n != ForecastOutput.LowerBoundColumnName
                                  && n != ForecastOutput.UpperBoundColumnName)
                ?? "Value";
        }

        if (string.IsNullOrEmpty(trainDataPath) || !File.Exists(trainDataPath))
        {
            return (null, "Original training data not found for forecasting predict. " +
                          "Forecasting models need the training data to generate forecasts.");
        }

        // Load training data with just the value column — find its index in the CSV header first.
        var headerLine = File.ReadLines(trainDataPath, System.Text.Encoding.UTF8).First();
        var headers = CsvFieldParser.ParseFields(headerLine);
        var colIdx = Array.FindIndex(headers, h => h.Equals(valueColName, StringComparison.OrdinalIgnoreCase));
        if (colIdx < 0) colIdx = headers.Length - 1; // fallback to last column

        var columnOptions = new TextLoader.Options
        {
            Columns = [new TextLoader.Column(valueColName, DataKind.Single, colIdx)],
            HasHeader = true,
            Separators = [','],
            AllowQuoting = true
        };

        var textLoader = mlContext.Data.CreateTextLoader(columnOptions);
        var trainData = textLoader.Load(trainDataPath);

        var predictions = model.Transform(trainData);
        var forecastCol = predictions.Schema.GetColumnOrNull(ForecastOutput.ForecastColumnName);
        var lowerCol = predictions.Schema.GetColumnOrNull(ForecastOutput.LowerBoundColumnName);
        var upperCol = predictions.Schema.GetColumnOrNull(ForecastOutput.UpperBoundColumnName);

        if (!forecastCol.HasValue)
        {
            return (null, $"Model does not produce {ForecastOutput.ForecastColumnName} column.");
        }

        using var cursor = predictions.GetRowCursor(predictions.Schema);
        var forecastGetter = cursor.GetGetter<VBuffer<float>>(forecastCol.Value);
        var lowerGetter = lowerCol.HasValue ? cursor.GetGetter<VBuffer<float>>(lowerCol.Value) : null;
        var upperGetter = upperCol.HasValue ? cursor.GetGetter<VBuffer<float>>(upperCol.Value) : null;

        VBuffer<float> forecastBuf = default, lowerBuf = default, upperBuf = default;
        if (cursor.MoveNext())
        {
            forecastGetter(ref forecastBuf);
            lowerGetter?.Invoke(ref lowerBuf);
            upperGetter?.Invoke(ref upperBuf);
        }

        var forecast = new ForecastOutput
        {
            ForecastedValues = forecastBuf.DenseValues().ToArray(),
            LowerBound = lowerBuf.DenseValues().ToArray(),
            UpperBound = upperBuf.DenseValues().ToArray(),
        };

        if (forecast.ForecastedValues.Length == 0)
        {
            return (null, "Forecast produced 0 values.");
        }

        if (requestedHorizon.HasValue && requestedHorizon.Value != forecast.ForecastedValues.Length)
        {
            return (null,
                $"This model was trained with a fixed horizon of {forecast.ForecastedValues.Length} and does not " +
                $"support a different runtime horizon ({requestedHorizon.Value}). Omit 'horizon' to use the " +
                "trained default, or retrain the model with the desired horizon.");
        }

        return (forecast, null);
    }

    /// <summary>
    /// Maps a horizon forecast onto the shared structured-prediction row schema (the same
    /// PredictionRow the /predict API and tabular --json emit): Score = forecasted value,
    /// ScoreLowerBound/Upper = SSA native band, IntervalConfidence = its coverage level.
    /// Row order is the step order — step = index + 1, matching the CSV output's Step column.
    /// </summary>
    public static List<PredictionRow> BuildForecastRows(ForecastOutput forecast)
    {
        var rows = new List<PredictionRow>(forecast.ForecastedValues.Length);
        for (int i = 0; i < forecast.ForecastedValues.Length; i++)
        {
            rows.Add(new PredictionRow
            {
                Score = forecast.ForecastedValues[i],
                ScoreLowerBound = i < forecast.LowerBound.Length ? forecast.LowerBound[i] : null,
                ScoreUpperBound = i < forecast.UpperBound.Length ? forecast.UpperBound[i] : null,
                IntervalConfidence = ForecastOutput.ConfidenceLevel,
            });
        }
        return rows;
    }
}
