using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.AutoML;

/// <summary>
/// AutoML runner implementation using ML.NET AutoML
/// </summary>
public class AutoMLRunner
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;

    public AutoMLRunner(MLContext mlContext, IDataProvider dataLoader)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
    }

    public async Task<AutoMLResult> RunAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Load data
        var dataView = _dataLoader.LoadData(config.DataFile, config.LabelColumn);

        // Split data
        var (trainSet, testSet) = _dataLoader.SplitData(dataView, config.TestSplit);

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

        return result;
    }

    private async Task<AutoMLResult> RunBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
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

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = new Dictionary<string, double>
            {
                ["accuracy"] = metrics.Accuracy,
                ["auc"] = metrics.AreaUnderRocCurve,
                ["f1_score"] = metrics.F1Score,
                ["precision"] = metrics.PositivePrecision,
                ["recall"] = metrics.PositiveRecall
            }
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

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = new Dictionary<string, double>
            {
                ["accuracy"] = metrics.MacroAccuracy,
                ["micro_accuracy"] = metrics.MicroAccuracy,
                ["log_loss"] = metrics.LogLoss
            }
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

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = new Dictionary<string, double>
            {
                ["r_squared"] = metrics.RSquared,
                ["rmse"] = metrics.RootMeanSquaredError,
                ["mae"] = metrics.MeanAbsoluteError,
                ["mse"] = metrics.MeanSquaredError
            }
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
}
