namespace MLoop.Core.Models;

/// <summary>
/// Result of a training operation
/// </summary>
public class TrainingResult
{
    public required string ExperimentId { get; init; }
    public required string BestTrainer { get; init; }
    public required Dictionary<string, double> Metrics { get; init; }
    public required double TrainingTimeSeconds { get; init; }
    public required string ModelPath { get; init; }

    /// <summary>
    /// Number of rows in the training dataset (for memory collection)
    /// </summary>
    public long RowCount { get; init; }

    /// <summary>
    /// Comma-separated column names from schema (for memory collection)
    /// </summary>
    public string? SchemaInfo { get; init; }
}

/// <summary>
/// Progress information during training
/// </summary>
public class TrainingProgress
{
    public required int TrialNumber { get; init; }
    public required string TrainerName { get; init; }
    public required double Metric { get; init; }
    public required string MetricName { get; init; }
    public required double ElapsedSeconds { get; init; }
}
