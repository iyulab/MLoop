// Example PostTrain Hook: Model Performance Gate
//
// Purpose:
//   Validates trained model meets minimum performance thresholds.
//   Prevents deployment of underperforming models.
//
// Installation:
//   Copy to: .mloop/scripts/hooks/post-train/02_performance_gate.cs
//
// Configuration:
//   Customize MIN_ACCURACY, MIN_AUC, MIN_F1_SCORE as needed.
//   Adjust thresholds based on your use case requirements.

using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility.Hooks;

public class PerformanceGateHook : IMLoopHook
{
    public string Name => "Model Performance Gate";

    // Configuration: Minimum performance thresholds
    private const double MIN_ACCURACY = 0.70;  // 70% minimum accuracy
    private const double MIN_AUC = 0.75;       // 75% minimum AUC (for binary classification)
    private const double MIN_F1_SCORE = 0.65;  // 65% minimum F1 score
    private const double MIN_R_SQUARED = 0.60; // 60% minimum R² (for regression)

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            if (ctx.Metrics == null)
            {
                ctx.Logger.Warning("⚠️ No metrics available for performance validation");
                return Task.FromResult(HookResult.Continue());
            }

            var taskType = ctx.GetMetadata<string>("TaskType", "unknown");
            ctx.Logger.Info($"🎯 Validating {taskType} model performance...");

            // Type-specific validation
            if (taskType.Contains("BinaryClassification", StringComparison.OrdinalIgnoreCase))
            {
                return ValidateBinaryClassification(ctx);
            }
            else if (taskType.Contains("MulticlassClassification", StringComparison.OrdinalIgnoreCase))
            {
                return ValidateMulticlassClassification(ctx);
            }
            else if (taskType.Contains("Regression", StringComparison.OrdinalIgnoreCase))
            {
                return ValidateRegression(ctx);
            }

            ctx.Logger.Info("✅ Performance validation skipped for unsupported task type");
            return Task.FromResult(HookResult.Continue());
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Performance gate validation failed: {ex.Message}");
            return Task.FromResult(HookResult.Abort($"Validation error: {ex.Message}"));
        }
    }

    private Task<HookResult> ValidateBinaryClassification(HookContext ctx)
    {
        var metrics = ctx.Metrics as BinaryClassificationMetrics;
        if (metrics == null)
        {
            // Try to extract from IDictionary
            var metricsDict = ctx.Metrics as IDictionary<string, double>;
            if (metricsDict == null)
            {
                ctx.Logger.Warning("⚠️ Unable to parse binary classification metrics");
                return Task.FromResult(HookResult.Continue());
            }

            return ValidateFromDictionary(ctx, metricsDict, new Dictionary<string, double>
            {
                ["Accuracy"] = MIN_ACCURACY,
                ["AreaUnderRocCurve"] = MIN_AUC,
                ["F1Score"] = MIN_F1_SCORE
            });
        }

        ctx.Logger.Info($"   Accuracy: {metrics.Accuracy:P2} (min: {MIN_ACCURACY:P2})");
        ctx.Logger.Info($"   AUC: {metrics.AreaUnderRocCurve:P2} (min: {MIN_AUC:P2})");
        ctx.Logger.Info($"   F1 Score: {metrics.F1Score:P2} (min: {MIN_F1_SCORE:P2})");

        // Check thresholds
        var failures = new List<string>();

        if (metrics.Accuracy < MIN_ACCURACY)
        {
            failures.Add($"Accuracy {metrics.Accuracy:P2} < {MIN_ACCURACY:P2}");
        }

        if (metrics.AreaUnderRocCurve < MIN_AUC)
        {
            failures.Add($"AUC {metrics.AreaUnderRocCurve:P2} < {MIN_AUC:P2}");
        }

        if (metrics.F1Score < MIN_F1_SCORE)
        {
            failures.Add($"F1 Score {metrics.F1Score:P2} < {MIN_F1_SCORE:P2}");
        }

        if (failures.Any())
        {
            ctx.Logger.Error("❌ Model failed performance gate:");
            foreach (var failure in failures)
            {
                ctx.Logger.Error($"   - {failure}");
            }

            return Task.FromResult(HookResult.Abort(
                "Model performance below minimum thresholds. " +
                "Consider: (1) More training data, (2) Feature engineering, (3) Longer training time"));
        }

        ctx.Logger.Info("✅ Model passed performance gate");
        return Task.FromResult(HookResult.Continue());
    }

    private Task<HookResult> ValidateMulticlassClassification(HookContext ctx)
    {
        var metrics = ctx.Metrics as MulticlassClassificationMetrics;
        if (metrics == null)
        {
            var metricsDict = ctx.Metrics as IDictionary<string, double>;
            if (metricsDict == null)
            {
                ctx.Logger.Warning("⚠️ Unable to parse multiclass classification metrics");
                return Task.FromResult(HookResult.Continue());
            }

            return ValidateFromDictionary(ctx, metricsDict, new Dictionary<string, double>
            {
                ["MacroAccuracy"] = MIN_ACCURACY,
                ["MicroAccuracy"] = MIN_ACCURACY
            });
        }

        ctx.Logger.Info($"   Macro Accuracy: {metrics.MacroAccuracy:P2} (min: {MIN_ACCURACY:P2})");
        ctx.Logger.Info($"   Micro Accuracy: {metrics.MicroAccuracy:P2} (min: {MIN_ACCURACY:P2})");

        if (metrics.MacroAccuracy < MIN_ACCURACY || metrics.MicroAccuracy < MIN_ACCURACY)
        {
            return Task.FromResult(HookResult.Abort(
                $"Model accuracy below threshold: " +
                $"Macro={metrics.MacroAccuracy:P2}, Micro={metrics.MicroAccuracy:P2} " +
                $"(min: {MIN_ACCURACY:P2})"));
        }

        ctx.Logger.Info("✅ Model passed performance gate");
        return Task.FromResult(HookResult.Continue());
    }

    private Task<HookResult> ValidateRegression(HookContext ctx)
    {
        var metrics = ctx.Metrics as RegressionMetrics;
        if (metrics == null)
        {
            var metricsDict = ctx.Metrics as IDictionary<string, double>;
            if (metricsDict == null)
            {
                ctx.Logger.Warning("⚠️ Unable to parse regression metrics");
                return Task.FromResult(HookResult.Continue());
            }

            return ValidateFromDictionary(ctx, metricsDict, new Dictionary<string, double>
            {
                ["RSquared"] = MIN_R_SQUARED
            });
        }

        ctx.Logger.Info($"   R²: {metrics.RSquared:P2} (min: {MIN_R_SQUARED:P2})");
        ctx.Logger.Info($"   RMSE: {metrics.RootMeanSquaredError:F4}");
        ctx.Logger.Info($"   MAE: {metrics.MeanAbsoluteError:F4}");

        if (metrics.RSquared < MIN_R_SQUARED)
        {
            return Task.FromResult(HookResult.Abort(
                $"Model R² below threshold: {metrics.RSquared:P2} < {MIN_R_SQUARED:P2}"));
        }

        ctx.Logger.Info("✅ Model passed performance gate");
        return Task.FromResult(HookResult.Continue());
    }

    private Task<HookResult> ValidateFromDictionary(
        HookContext ctx,
        IDictionary<string, double> metrics,
        Dictionary<string, double> thresholds)
    {
        var failures = new List<string>();

        foreach (var (metricName, threshold) in thresholds)
        {
            if (metrics.TryGetValue(metricName, out var value))
            {
                ctx.Logger.Info($"   {metricName}: {value:P2} (min: {threshold:P2})");

                if (value < threshold)
                {
                    failures.Add($"{metricName} {value:P2} < {threshold:P2}");
                }
            }
        }

        if (failures.Any())
        {
            ctx.Logger.Error("❌ Model failed performance gate:");
            foreach (var failure in failures)
            {
                ctx.Logger.Error($"   - {failure}");
            }

            return Task.FromResult(HookResult.Abort(
                "Model performance below minimum thresholds"));
        }

        ctx.Logger.Info("✅ Model passed performance gate");
        return Task.FromResult(HookResult.Continue());
    }
}
