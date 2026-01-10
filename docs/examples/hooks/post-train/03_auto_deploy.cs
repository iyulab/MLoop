// Example PostTrain Hook: Automated Deployment Trigger
//
// Purpose:
//   Automatically triggers model deployment when performance exceeds threshold.
//   Supports multiple deployment targets: REST API, Azure ML, AWS SageMaker, etc.
//
// Installation:
//   Copy to: .mloop/scripts/hooks/post-train/03_auto_deploy.cs
//
// Configuration:
//   Set DEPLOY_THRESHOLD, DEPLOY_TARGET, and deployment credentials as needed.
//   Customize deployment logic for your infrastructure.

using System.Threading.Tasks;
using System.Diagnostics;
using Microsoft.ML.Data;
using MLoop.Extensibility.Hooks;

public class AutoDeployHook : IMLoopHook
{
    public string Name => "Automated Deployment Trigger";

    // Configuration
    private const double DEPLOY_THRESHOLD = 0.90;  // 90% accuracy for auto-deployment
    private readonly string _deployTarget;
    private readonly string _deploymentScript;

    public AutoDeployHook()
    {
        _deployTarget = Environment.GetEnvironmentVariable("DEPLOY_TARGET") ?? "local";
        _deploymentScript = Environment.GetEnvironmentVariable("DEPLOY_SCRIPT") ?? "";
    }

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            if (ctx.Metrics == null)
            {
                ctx.Logger.Info("ℹ️  No metrics available for deployment decision");
                return HookResult.Continue();
            }

            var modelName = ctx.GetMetadata<string>("ModelName", "default");
            var taskType = ctx.GetMetadata<string>("TaskType", "unknown");
            var modelPath = ctx.GetMetadata<string>("ModelPath", "");

            // Extract primary metric based on task type
            double primaryMetric = ExtractPrimaryMetric(ctx.Metrics, taskType);

            ctx.Logger.Info($"🎯 Primary metric: {primaryMetric:P2} (deploy threshold: {DEPLOY_THRESHOLD:P2})");

            if (primaryMetric >= DEPLOY_THRESHOLD)
            {
                ctx.Logger.Info($"🚀 Metric exceeds threshold! Triggering deployment to {_deployTarget}...");

                // Execute deployment
                bool deploymentSuccess = await ExecuteDeployment(
                    modelName,
                    modelPath,
                    primaryMetric,
                    ctx);

                if (deploymentSuccess)
                {
                    ctx.Logger.Info("✅ Deployment triggered successfully");

                    // Optionally modify metadata to track deployment
                    return HookResult.ModifyConfig(
                        new Dictionary<string, object>
                        {
                            ["DeploymentTriggered"] = true,
                            ["DeploymentTarget"] = _deployTarget,
                            ["DeploymentMetric"] = primaryMetric
                        },
                        $"Model deployed to {_deployTarget}");
                }
                else
                {
                    ctx.Logger.Warning("⚠️ Deployment trigger failed, but training completed successfully");
                    return HookResult.Continue();
                }
            }
            else
            {
                ctx.Logger.Info($"ℹ️  Metric {primaryMetric:P2} below deployment threshold, skipping auto-deploy");
                return HookResult.Continue();
            }
        }
        catch (Exception ex)
        {
            ctx.Logger.Warning($"⚠️ Deployment trigger failed: {ex.Message}");
            ctx.Logger.Warning("   Training will continue despite deployment failure");
            return HookResult.Continue(); // Non-critical failure
        }
    }

    private double ExtractPrimaryMetric(object metrics, string taskType)
    {
        // Try typed metrics first
        if (metrics is BinaryClassificationMetrics binaryMetrics)
        {
            return binaryMetrics.Accuracy;
        }
        else if (metrics is MulticlassClassificationMetrics multiclassMetrics)
        {
            return multiclassMetrics.MacroAccuracy;
        }
        else if (metrics is RegressionMetrics regressionMetrics)
        {
            return regressionMetrics.RSquared;
        }

        // Fallback to dictionary
        var metricsDict = metrics as IDictionary<string, double>;
        if (metricsDict != null)
        {
            // Try common metric names
            if (metricsDict.TryGetValue("Accuracy", out var accuracy)) return accuracy;
            if (metricsDict.TryGetValue("MacroAccuracy", out var macroAccuracy)) return macroAccuracy;
            if (metricsDict.TryGetValue("RSquared", out var rSquared)) return rSquared;
            if (metricsDict.TryGetValue("AreaUnderRocCurve", out var auc)) return auc;

            // Return first metric if no known metric found
            return metricsDict.Values.FirstOrDefault();
        }

        return 0.0;
    }

    private async Task<bool> ExecuteDeployment(
        string modelName,
        string modelPath,
        double metric,
        HookContext ctx)
    {
        switch (_deployTarget.ToLowerInvariant())
        {
            case "local":
                return await DeployLocal(modelName, modelPath, ctx);

            case "docker":
                return await DeployDocker(modelName, modelPath, metric, ctx);

            case "kubernetes":
                return await DeployKubernetes(modelName, modelPath, metric, ctx);

            case "azureml":
                return await DeployAzureML(modelName, modelPath, metric, ctx);

            case "sagemaker":
                return await DeploySageMaker(modelName, modelPath, metric, ctx);

            case "custom":
                return await DeployCustom(modelName, modelPath, metric, ctx);

            default:
                ctx.Logger.Warning($"⚠️ Unknown deployment target: {_deployTarget}");
                return false;
        }
    }

    private async Task<bool> DeployLocal(string modelName, string modelPath, HookContext ctx)
    {
        ctx.Logger.Info("📦 Deploying to local API server...");

        // Example: Promote to production slot
        var productionPath = Path.Combine(
            ctx.ProjectRoot,
            "models",
            modelName,
            "production",
            "model.zip");

        Directory.CreateDirectory(Path.GetDirectoryName(productionPath)!);
        File.Copy(modelPath, productionPath, overwrite: true);

        ctx.Logger.Info($"   Model promoted to production: {productionPath}");

        // Optional: Restart API server
        await RestartApiServer(ctx);

        return true;
    }

    private async Task<bool> DeployDocker(string modelName, string modelPath, double metric, HookContext ctx)
    {
        ctx.Logger.Info("🐳 Building and pushing Docker image...");

        var imageName = $"{modelName}:v{DateTime.UtcNow:yyyyMMddHHmmss}";
        var buildCommand = $"docker build -t {imageName} -f Dockerfile --build-arg MODEL_PATH={modelPath} .";

        var buildResult = await RunCommand("docker", buildCommand, ctx);
        if (!buildResult)
        {
            return false;
        }

        // Push to registry
        var registry = Environment.GetEnvironmentVariable("DOCKER_REGISTRY") ?? "localhost:5000";
        var pushCommand = $"docker push {registry}/{imageName}";

        return await RunCommand("docker", pushCommand, ctx);
    }

    private async Task<bool> DeployKubernetes(string modelName, string modelPath, double metric, HookContext ctx)
    {
        ctx.Logger.Info("☸️  Deploying to Kubernetes...");

        // Update deployment YAML with new model path
        var deploymentYaml = $@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {modelName}-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {modelName}
  template:
    metadata:
      labels:
        app: {modelName}
        version: {DateTime.UtcNow:yyyyMMddHHmmss}
    spec:
      containers:
      - name: {modelName}
        image: mloop-api:latest
        env:
        - name: MODEL_PATH
          value: {modelPath}
        - name: MODEL_METRIC
          value: {metric:F4}
";

        var yamlPath = Path.Combine(ctx.ProjectRoot, "deployment.yaml");
        await File.WriteAllTextAsync(yamlPath, deploymentYaml);

        // Apply deployment
        return await RunCommand("kubectl", $"apply -f {yamlPath}", ctx);
    }

    private async Task<bool> DeployAzureML(string modelName, string modelPath, double metric, HookContext ctx)
    {
        ctx.Logger.Info("☁️  Deploying to Azure ML...");

        // Example: Use Azure CLI to register and deploy model
        var registerCommand = $"az ml model register --name {modelName} --model-path {modelPath} --tags metric={metric:F4}";
        var registerResult = await RunCommand("az", registerCommand, ctx);

        if (!registerResult)
        {
            return false;
        }

        // Deploy to endpoint
        var deployCommand = $"az ml model deploy --name {modelName}-service --model {modelName}:1";
        return await RunCommand("az", deployCommand, ctx);
    }

    private async Task<bool> DeploySageMaker(string modelName, string modelPath, double metric, HookContext ctx)
    {
        ctx.Logger.Info("☁️  Deploying to AWS SageMaker...");

        // Example: Use AWS CLI to upload and deploy model
        var s3Bucket = Environment.GetEnvironmentVariable("S3_BUCKET") ?? "mloop-models";
        var s3Key = $"{modelName}/{DateTime.UtcNow:yyyyMMddHHmmss}/model.tar.gz";

        // Upload to S3
        var uploadCommand = $"aws s3 cp {modelPath} s3://{s3Bucket}/{s3Key}";
        var uploadResult = await RunCommand("aws", uploadCommand, ctx);

        if (!uploadResult)
        {
            return false;
        }

        // Create SageMaker model
        var createModelCommand = $"aws sagemaker create-model --model-name {modelName} --primary-container Image=...,ModelDataUrl=s3://{s3Bucket}/{s3Key}";
        return await RunCommand("aws", createModelCommand, ctx);
    }

    private async Task<bool> DeployCustom(string modelName, string modelPath, double metric, HookContext ctx)
    {
        if (string.IsNullOrEmpty(_deploymentScript))
        {
            ctx.Logger.Warning("⚠️ DEPLOY_SCRIPT environment variable not set");
            return false;
        }

        ctx.Logger.Info($"🔧 Running custom deployment script: {_deploymentScript}");

        // Pass deployment info as environment variables
        var envVars = new Dictionary<string, string>
        {
            ["MODEL_NAME"] = modelName,
            ["MODEL_PATH"] = modelPath,
            ["MODEL_METRIC"] = metric.ToString("F4"),
            ["DEPLOY_TARGET"] = _deployTarget
        };

        return await RunScript(_deploymentScript, envVars, ctx);
    }

    private async Task<bool> RestartApiServer(HookContext ctx)
    {
        // Example: Send signal to restart API process
        ctx.Logger.Info("🔄 Restarting API server...");

        // Implementation depends on how API is deployed
        // For systemd: systemctl restart mloop-api
        // For Docker: docker restart mloop-api
        // For process manager: pm2 restart mloop-api

        await Task.Delay(100); // Placeholder
        return true;
    }

    private async Task<bool> RunCommand(string command, string arguments, HookContext ctx)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = command,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null)
            {
                ctx.Logger.Error($"Failed to start process: {command}");
                return false;
            }

            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                var error = await process.StandardError.ReadToEndAsync();
                ctx.Logger.Error($"Command failed: {command} {arguments}");
                ctx.Logger.Error($"   {error}");
                return false;
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            ctx.Logger.Info($"   {output}");
            return true;
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Failed to run command: {ex.Message}");
            return false;
        }
    }

    private async Task<bool> RunScript(string scriptPath, Dictionary<string, string> envVars, HookContext ctx)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = scriptPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            foreach (var (key, value) in envVars)
            {
                startInfo.Environment[key] = value;
            }

            using var process = Process.Start(startInfo);
            if (process == null)
            {
                ctx.Logger.Error($"Failed to start script: {scriptPath}");
                return false;
            }

            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                var error = await process.StandardError.ReadToEndAsync();
                ctx.Logger.Error($"Script failed: {scriptPath}");
                ctx.Logger.Error($"   {error}");
                return false;
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            ctx.Logger.Info($"   {output}");
            return true;
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Failed to run script: {ex.Message}");
            return false;
        }
    }
}
