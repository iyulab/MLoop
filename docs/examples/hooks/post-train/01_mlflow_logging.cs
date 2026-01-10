// Example PostTrain Hook: MLflow Experiment Tracking
//
// Purpose:
//   Logs training experiments to MLflow tracking server.
//   Records metrics, parameters, and model artifacts for experiment tracking.
//
// Installation:
//   1. Install MLflow: pip install mlflow
//   2. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000
//   3. Copy to: .mloop/scripts/hooks/post-train/01_mlflow_logging.cs
//   4. Set MLFLOW_TRACKING_URI environment variable (default: http://localhost:5000)
//
// Dependencies:
//   Requires MLflow REST API or Python client via subprocess

using System.Threading.Tasks;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using MLoop.Extensibility.Hooks;

public class MLflowLoggingHook : IMLoopHook
{
    public string Name => "MLflow Experiment Tracking";

    private static readonly HttpClient _httpClient = new HttpClient();
    private readonly string _trackingUri;

    public MLflowLoggingHook()
    {
        _trackingUri = Environment.GetEnvironmentVariable("MLFLOW_TRACKING_URI")
            ?? "http://localhost:5000";
    }

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            var modelName = ctx.GetMetadata<string>("ModelName", "default");
            var experimentId = ctx.GetMetadata<string>("ExperimentId", "unknown");
            var taskType = ctx.GetMetadata<string>("TaskType", "unknown");
            var labelColumn = ctx.GetMetadata<string>("LabelColumn", "Label");
            var timeLimit = ctx.GetMetadata<int>("TimeLimit", 60);
            var bestTrainer = ctx.GetMetadata<string>("BestTrainer", "unknown");

            ctx.Logger.Info($"📊 Logging to MLflow: {_trackingUri}");

            // Create or get experiment
            var mlflowExperimentId = await GetOrCreateExperiment(modelName);

            // Start run
            var runId = await CreateRun(mlflowExperimentId, $"{modelName}-{experimentId}");

            // Log parameters
            await LogParameter(runId, "model_name", modelName);
            await LogParameter(runId, "task_type", taskType);
            await LogParameter(runId, "label_column", labelColumn);
            await LogParameter(runId, "time_limit_seconds", timeLimit.ToString());
            await LogParameter(runId, "best_trainer", bestTrainer);

            // Log metrics
            if (ctx.Metrics != null)
            {
                var metricsDict = ctx.Metrics as IDictionary<string, double>;
                if (metricsDict != null)
                {
                    foreach (var (metricName, metricValue) in metricsDict)
                    {
                        await LogMetric(runId, metricName, metricValue);
                        ctx.Logger.Info($"   {metricName}: {metricValue:F4}");
                    }
                }
            }

            // Log model artifact path
            var modelPath = ctx.GetMetadata<string>("ModelPath", "");
            if (!string.IsNullOrEmpty(modelPath))
            {
                await LogParameter(runId, "model_path", modelPath);
            }

            // End run
            await EndRun(runId);

            ctx.Logger.Info($"✅ MLflow tracking complete: {_trackingUri}/#/experiments/{mlflowExperimentId}/runs/{runId}");
            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Warning($"⚠️ MLflow logging failed: {ex.Message}");
            ctx.Logger.Warning("   Training will continue despite logging failure");
            return HookResult.Continue(); // Non-critical failure
        }
    }

    private async Task<string> GetOrCreateExperiment(string experimentName)
    {
        // Try to get existing experiment
        var getUrl = $"{_trackingUri}/api/2.0/mlflow/experiments/get-by-name?experiment_name={Uri.EscapeDataString(experimentName)}";
        var getResponse = await _httpClient.GetAsync(getUrl);

        if (getResponse.IsSuccessStatusCode)
        {
            var getContent = await getResponse.Content.ReadAsStringAsync();
            var getDoc = JsonDocument.Parse(getContent);
            return getDoc.RootElement.GetProperty("experiment").GetProperty("experiment_id").GetString()!;
        }

        // Create new experiment
        var createUrl = $"{_trackingUri}/api/2.0/mlflow/experiments/create";
        var createPayload = JsonSerializer.Serialize(new { name = experimentName });
        var createContent = new StringContent(createPayload, Encoding.UTF8, "application/json");
        var createResponse = await _httpClient.PostAsync(createUrl, createContent);
        createResponse.EnsureSuccessStatusCode();

        var createResponseContent = await createResponse.Content.ReadAsStringAsync();
        var createDoc = JsonDocument.Parse(createResponseContent);
        return createDoc.RootElement.GetProperty("experiment_id").GetString()!;
    }

    private async Task<string> CreateRun(string experimentId, string runName)
    {
        var url = $"{_trackingUri}/api/2.0/mlflow/runs/create";
        var payload = JsonSerializer.Serialize(new
        {
            experiment_id = experimentId,
            run_name = runName,
            start_time = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        });

        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync(url, content);
        response.EnsureSuccessStatusCode();

        var responseContent = await response.Content.ReadAsStringAsync();
        var doc = JsonDocument.Parse(responseContent);
        return doc.RootElement.GetProperty("run").GetProperty("info").GetProperty("run_id").GetString()!;
    }

    private async Task LogParameter(string runId, string key, string value)
    {
        var url = $"{_trackingUri}/api/2.0/mlflow/runs/log-parameter";
        var payload = JsonSerializer.Serialize(new { run_id = runId, key, value });
        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        await _httpClient.PostAsync(url, content);
    }

    private async Task LogMetric(string runId, string key, double value)
    {
        var url = $"{_trackingUri}/api/2.0/mlflow/runs/log-metric";
        var payload = JsonSerializer.Serialize(new
        {
            run_id = runId,
            key,
            value,
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        });
        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        await _httpClient.PostAsync(url, content);
    }

    private async Task EndRun(string runId)
    {
        var url = $"{_trackingUri}/api/2.0/mlflow/runs/update";
        var payload = JsonSerializer.Serialize(new
        {
            run_id = runId,
            status = "FINISHED",
            end_time = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        });
        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        await _httpClient.PostAsync(url, content);
    }
}
