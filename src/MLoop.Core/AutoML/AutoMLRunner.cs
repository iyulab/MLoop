using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.AutoML;

/// <summary>
/// AutoML runner implementation using ML.NET AutoML with extensibility support
/// </summary>
public class AutoMLRunner
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;
    private readonly string? _projectRoot;
    private readonly ScriptDiscovery _scriptDiscovery;
    private readonly ConsoleLogger _logger;

    public AutoMLRunner(MLContext mlContext, IDataProvider dataLoader, string? projectRoot = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
        _projectRoot = projectRoot ?? Directory.GetCurrentDirectory();
        _scriptDiscovery = new ScriptDiscovery(_projectRoot);
        _logger = new ConsoleLogger();
    }

    public async Task<AutoMLResult> RunAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Load and split data
        IDataView trainSet, testSet;
        if (!string.IsNullOrEmpty(config.TestDataFile))
        {
            // Pre-split data (e.g. balanced training with separate test set)
            trainSet = _dataLoader.LoadData(config.DataFile, config.LabelColumn, config.Task);
            testSet = _dataLoader.LoadData(config.TestDataFile, config.LabelColumn, config.Task);
        }
        else
        {
            var dataView = _dataLoader.LoadData(config.DataFile, config.LabelColumn, config.Task);
            (trainSet, testSet) = _dataLoader.SplitData(dataView, config.TestSplit);
        }

        // Discover hooks (zero-overhead if .mloop/scripts/hooks/ doesn't exist)
        var hooks = await _scriptDiscovery.DiscoverHooksAsync();

        // Execute pre-train hooks
        if (hooks.Count > 0)
        {
            var preTrainContext = new HookContext
            {
                HookType = HookType.PreTrain,
                HookName = "pre-train",
                MLContext = _mlContext,
                DataView = trainSet,
                ProjectRoot = _projectRoot!,
                Logger = _logger,
                Metadata = new Dictionary<string, object>
                {
                    ["LabelColumn"] = config.LabelColumn,
                    ["TaskType"] = config.Task,
                    ["TimeLimitSeconds"] = config.TimeLimitSeconds
                }
            };

            foreach (var hook in hooks)
            {
                var hookResult = await hook.ExecuteAsync(preTrainContext);
                if (hookResult.Action == HookAction.Abort)
                {
                    throw new InvalidOperationException(
                        $"Hook '{hook.Name}' aborted training: {hookResult.Message}");
                }
            }
        }

        // Run AutoML based on task type
        var result = config.Task.ToLowerInvariant() switch
        {
            "binary-classification" => await RunBinaryClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken),
            "multiclass-classification" => await RunMulticlassClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken),
            "regression" => await RunRegressionAsync(
                trainSet, testSet, config, progress, cancellationToken),
            _ => throw new NotSupportedException($"Task type '{config.Task}' is not supported")
        };

        // Execute post-train hooks
        if (hooks.Count > 0)
        {
            var postTrainContext = new HookContext
            {
                HookType = HookType.PostTrain,
                HookName = "post-train",
                MLContext = _mlContext,
                DataView = testSet,
                Model = result.Model,
                ProjectRoot = _projectRoot!,
                Logger = _logger,
                Metadata = new Dictionary<string, object>
                {
                    ["LabelColumn"] = config.LabelColumn,
                    ["TaskType"] = config.Task,
                    ["BestTrainer"] = result.BestTrainer,
                    ["Metrics"] = result.Metrics
                }
            };

            foreach (var hook in hooks)
            {
                try
                {
                    await hook.ExecuteAsync(postTrainContext);
                }
                catch (Exception ex)
                {
                    _logger.Warning($"Post-train hook '{hook.Name}' failed: {ex.Message}");
                }
            }
        }

        return result;
    }

    private async Task<AutoMLResult> RunBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var optimizingMetric = GetBinaryMetric(config.Metric);
        string? metricFallbackNote = null;

        try
        {
            return await RunBinaryClassificationCoreAsync(
                trainSet, testSet, config, optimizingMetric, cancellationToken);
        }
        catch (InvalidOperationException ex) when (
            optimizingMetric == BinaryClassificationMetric.AreaUnderRocCurve &&
            (ex.Message.Contains("AUC") || ex.Message.Contains("positive class")))
        {
            // AUC requires both positive and negative samples in the test set.
            // With extreme class imbalance, the test split may have only one class.
            // Fall back to F1Score which is more robust for imbalanced data.
            Console.WriteLine("[Warning] AUC metric failed (extreme class imbalance). Falling back to F1Score.");
            metricFallbackNote = "AUC→F1Score (extreme imbalance)";

            return await RunBinaryClassificationCoreAsync(
                trainSet, testSet, config, BinaryClassificationMetric.F1Score, cancellationToken,
                metricFallbackNote);
        }
    }

    private async Task<AutoMLResult> RunBinaryClassificationCoreAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        BinaryClassificationMetric optimizingMetric,
        CancellationToken cancellationToken,
        string? metricFallbackNote = null)
    {
        var settings = new BinaryExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = optimizingMetric,
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateBinaryClassificationExperiment(settings);

        // Execute AutoML (progress tracking via settings callback)
        var experimentResult = await Task.Run(
            () => experiment.Execute(trainSet, config.LabelColumn),
            cancellationToken);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, config.LabelColumn);

        var metricsDict = new Dictionary<string, double>
        {
            ["accuracy"] = metrics.Accuracy,
            ["f1_score"] = metrics.F1Score,
            ["precision"] = metrics.PositivePrecision,
            ["recall"] = metrics.PositiveRecall
        };

        // Only include AUC if it's a valid number (may be NaN for imbalanced data)
        if (!double.IsNaN(metrics.AreaUnderRocCurve))
        {
            metricsDict["auc"] = metrics.AreaUnderRocCurve;
        }

        var trainerName = experimentResult.BestRun.TrainerName;
        if (metricFallbackNote != null)
        {
            trainerName += $" [metric fallback: {metricFallbackNote}]";
        }

        return new AutoMLResult
        {
            BestTrainer = trainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private async Task<AutoMLResult> RunMulticlassClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var settings = new MulticlassExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = GetMulticlassMetric(config.Metric),
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateMulticlassClassificationExperiment(settings);

        // Execute AutoML (progress tracking via settings callback)
        var experimentResult = await Task.Run(
            () => experiment.Execute(trainSet, config.LabelColumn),
            cancellationToken);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, config.LabelColumn);

        var metricsDict = new Dictionary<string, double>
        {
            ["macro_accuracy"] = metrics.MacroAccuracy,
            ["micro_accuracy"] = metrics.MicroAccuracy,
            ["log_loss"] = metrics.LogLoss
        };

        // Calculate Macro F1 from confusion matrix per-class precision/recall
        try
        {
            var cm = metrics.ConfusionMatrix;
            var classCount = cm.PerClassPrecision.Count;
            if (classCount > 0)
            {
                double f1Sum = 0;
                for (int i = 0; i < classCount; i++)
                {
                    var p = cm.PerClassPrecision[i];
                    var r = cm.PerClassRecall[i];
                    f1Sum += (p + r) > 0 ? 2 * p * r / (p + r) : 0;
                }
                metricsDict["macro_f1"] = f1Sum / classCount;
            }
        }
        catch
        {
            // Non-critical: skip if confusion matrix unavailable
        }

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private async Task<AutoMLResult> RunRegressionAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var settings = new RegressionExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = GetRegressionMetric(config.Metric),
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateRegressionExperiment(settings);

        // Execute AutoML (progress tracking via settings callback)
        var experimentResult = await Task.Run(
            () => experiment.Execute(trainSet, config.LabelColumn),
            cancellationToken);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);
        var metrics = _mlContext.Regression.Evaluate(predictions, config.LabelColumn);

        var metricsDict = new Dictionary<string, double>
        {
            ["r_squared"] = metrics.RSquared,
            ["rmse"] = metrics.RootMeanSquaredError,
            ["mae"] = metrics.MeanAbsoluteError,
            ["mse"] = metrics.MeanSquaredError
        };

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private BinaryClassificationMetric GetBinaryMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" => BinaryClassificationMetric.Accuracy,
            "auc" => BinaryClassificationMetric.AreaUnderRocCurve,
            "f1" or "f1_score" => BinaryClassificationMetric.F1Score,
            "auprc" => BinaryClassificationMetric.AreaUnderPrecisionRecallCurve,
            _ => BinaryClassificationMetric.Accuracy
        };
    }

    private MulticlassClassificationMetric GetMulticlassMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" or "macro_accuracy" => MulticlassClassificationMetric.MacroAccuracy,
            "micro_accuracy" => MulticlassClassificationMetric.MicroAccuracy,
            "log_loss" => MulticlassClassificationMetric.LogLoss,
            _ => MulticlassClassificationMetric.MacroAccuracy
        };
    }

    private RegressionMetric GetRegressionMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => RegressionMetric.RSquared,
            "rmse" => RegressionMetric.RootMeanSquaredError,
            "mae" => RegressionMetric.MeanAbsoluteError,
            "mse" => RegressionMetric.MeanSquaredError,
            _ => RegressionMetric.RSquared
        };
    }

    private double GetMetricValue(BinaryClassificationMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" => metrics.Accuracy,
            "auc" => metrics.AreaUnderRocCurve,
            "f1" or "f1_score" => metrics.F1Score,
            _ => metrics.Accuracy
        };
    }

    private double GetMetricValue(MulticlassClassificationMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" or "macro_accuracy" => metrics.MacroAccuracy,
            "micro_accuracy" => metrics.MicroAccuracy,
            "log_loss" => metrics.LogLoss,
            _ => metrics.MacroAccuracy
        };
    }

    private double GetMetricValue(RegressionMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => metrics.RSquared,
            "rmse" => metrics.RootMeanSquaredError,
            "mae" => metrics.MeanAbsoluteError,
            "mse" => metrics.MeanSquaredError,
            _ => metrics.RSquared
        };
    }
}

/// <summary>
/// Result from AutoML execution
/// </summary>
public class AutoMLResult
{
    public required string BestTrainer { get; init; }
    public required ITransformer Model { get; init; }
    public required Dictionary<string, double> Metrics { get; init; }

    /// <summary>
    /// Number of rows in the training dataset (for memory collection)
    /// </summary>
    public long RowCount { get; init; }
}

/// <summary>
/// Simple console logger implementation for preprocessing scripts
/// </summary>
internal class ConsoleLogger : ILogger
{
    public void Info(string message) => Console.WriteLine($"ℹ️  {message}");
    public void Warning(string message) => Console.WriteLine($"⚠️  {message}");
    public void Error(string message) => Console.WriteLine($"❌ {message}");
    public void Error(string message, Exception exception) => Console.WriteLine($"❌ {message}{Environment.NewLine}{exception}");
    public void Debug(string message) => Console.WriteLine($"🔍 {message}");
}
