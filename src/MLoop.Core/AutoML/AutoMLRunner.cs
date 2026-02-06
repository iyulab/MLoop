using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
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
        // Load data
        var dataView = _dataLoader.LoadData(config.DataFile, config.LabelColumn, config.Task);

        // Split data
        var (trainSet, testSet) = _dataLoader.SplitData(dataView, config.TestSplit);

        // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
        // TODO: Re-enable when implementing Phase 1

        //// Discover hooks and metrics (zero-overhead if not present)
        //var hooks = await _scriptDiscovery.DiscoverHooksAsync();
        //var customMetrics = await _scriptDiscovery.DiscoverMetricsAsync();

        //// Execute pre-train hooks
        //var preTrainContext = new HookContext
        //{
        //    MLContext = _mlContext,
        //    DataView = trainSet,
        //    Logger = _logger
        //};
        //preTrainContext.InitializeMetadata(new Dictionary<string, object>
        //{
        //    ["ExperimentId"] = Guid.NewGuid().ToString(),
        //    ["LabelColumn"] = config.LabelColumn,
        //    ["Task"] = config.Task,
        //    ["TimeLimitSeconds"] = config.TimeLimitSeconds
        //});

        //foreach (var hook in hooks)
        //{
        //    var hookResult = await hook.ExecuteAsync(preTrainContext);
        //    if (!hookResult.ShouldContinue)
        //    {
        //        throw new InvalidOperationException($"Hook '{hook.Name}' aborted training: {hookResult.Message}");
        //    }
        //}

        // Phase 0: No hooks/metrics yet - use empty lists
        var customMetrics = new List<object>(); // Will be List<IMLoopMetric> in Phase 1

        // Run AutoML based on task type
        var result = config.Task.ToLowerInvariant() switch
        {
            "binary-classification" => await RunBinaryClassificationAsync(
                trainSet, testSet, config, customMetrics, progress, cancellationToken),
            "multiclass-classification" => await RunMulticlassClassificationAsync(
                trainSet, testSet, config, customMetrics, progress, cancellationToken),
            "regression" => await RunRegressionAsync(
                trainSet, testSet, config, customMetrics, progress, cancellationToken),
            _ => throw new NotSupportedException($"Task type '{config.Task}' is not supported")
        };

        // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
        // TODO: Re-enable when implementing Phase 1

        //// Execute post-train hooks
        //var postTrainContext = new HookContext
        //{
        //    MLContext = _mlContext,
        //    DataView = testSet,
        //    Logger = _logger
        //};
        //postTrainContext.InitializeMetadata(new Dictionary<string, object>
        //{
        //    ["ExperimentId"] = preTrainContext.GetMetadata<string>("ExperimentId")!,
        //    ["LabelColumn"] = config.LabelColumn,
        //    ["Task"] = config.Task,
        //    ["BestTrainer"] = result.BestTrainer,
        //    ["Metrics"] = result.Metrics
        //});

        //foreach (var hook in hooks)
        //{
        //    await hook.ExecuteAsync(postTrainContext);
        //}

        return result;
    }

    private async Task<AutoMLResult> RunBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        List<object> customMetrics,  // Phase 0: Changed from List<IMLoopMetric>
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var settings = new BinaryExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = GetBinaryMetric(config.Metric),
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

        // Evaluate custom metrics
        var metricsDict = new Dictionary<string, double>
        {
            ["accuracy"] = metrics.Accuracy,
            ["auc"] = metrics.AreaUnderRocCurve,
            ["f1_score"] = metrics.F1Score,
            ["precision"] = metrics.PositivePrecision,
            ["recall"] = metrics.PositiveRecall
        };

        // NOTE: Phase 1 (Custom Metrics) - Disabled for Phase 0
        // TODO: Re-enable when implementing Phase 1
        //if (customMetrics.Count > 0)
        //{
        //    var metricContext = new MetricContext
        //    {
        //        MLContext = _mlContext,
        //        Predictions = predictions,
        //        LabelColumn = config.LabelColumn,
        //        ScoreColumn = "Score",
        //        Logger = _logger
        //    };

        //    foreach (var customMetric in customMetrics)
        //    {
        //        try
        //        {
        //            var value = await customMetric.CalculateAsync(metricContext);
        //            metricsDict[$"custom_{customMetric.Name.ToLowerInvariant().Replace(" ", "_")}"] = value;
        //        }
        //        catch (Exception ex)
        //        {
        //            _logger.Warning($"Custom metric '{customMetric.Name}' failed: {ex.Message}");
        //        }
        //    }
        //}

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private async Task<AutoMLResult> RunMulticlassClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        List<object> customMetrics,  // Phase 0: Changed from List<IMLoopMetric>
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

        // Evaluate custom metrics
        var metricsDict = new Dictionary<string, double>
        {
            ["macro_accuracy"] = metrics.MacroAccuracy,
            ["micro_accuracy"] = metrics.MicroAccuracy,
            ["log_loss"] = metrics.LogLoss
        };

        // NOTE: Phase 1 (Custom Metrics) - Disabled for Phase 0
        // TODO: Re-enable when implementing Phase 1
        //if (customMetrics.Count > 0)
        //{
        //    var metricContext = new MetricContext
        //    {
        //        MLContext = _mlContext,
        //        Predictions = predictions,
        //        LabelColumn = config.LabelColumn,
        //        ScoreColumn = "Score",
        //        Logger = _logger
        //    };

        //    foreach (var customMetric in customMetrics)
        //    {
        //        try
        //        {
        //            var value = await customMetric.CalculateAsync(metricContext);
        //            metricsDict[$"custom_{customMetric.Name.ToLowerInvariant().Replace(" ", "_")}"] = value;
        //        }
        //        catch (Exception ex)
        //        {
        //            _logger.Warning($"Custom metric '{customMetric.Name}' failed: {ex.Message}");
        //        }
        //    }
        //}

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
        List<object> customMetrics,  // Phase 0: Changed from List<IMLoopMetric>
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

        // Evaluate custom metrics
        var metricsDict = new Dictionary<string, double>
        {
            ["r_squared"] = metrics.RSquared,
            ["rmse"] = metrics.RootMeanSquaredError,
            ["mae"] = metrics.MeanAbsoluteError,
            ["mse"] = metrics.MeanSquaredError
        };

        // NOTE: Phase 1 (Custom Metrics) - Disabled for Phase 0
        // TODO: Re-enable when implementing Phase 1
        //if (customMetrics.Count > 0)
        //{
        //    var metricContext = new MetricContext
        //    {
        //        MLContext = _mlContext,
        //        Predictions = predictions,
        //        LabelColumn = config.LabelColumn,
        //        ScoreColumn = "Score",
        //        Logger = _logger
        //    };

        //    foreach (var customMetric in customMetrics)
        //    {
        //        try
        //        {
        //            var value = await customMetric.CalculateAsync(metricContext);
        //            metricsDict[$"custom_{customMetric.Name.ToLowerInvariant().Replace(" ", "_")}"] = value;
        //        }
        //        catch (Exception ex)
        //        {
        //            _logger.Warning($"Custom metric '{customMetric.Name}' failed: {ex.Message}");
        //        }
        //    }
        //}

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
