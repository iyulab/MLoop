namespace MLoop.Core.Prediction;

public class PredictionResult
{
    public required string TaskType { get; init; }
    public required List<PredictionRow> Rows { get; init; }
    public PredictionMetadata? Metadata { get; init; }
    public List<string>? Warnings { get; init; }
}

public class PredictionRow
{
    public string? PredictedLabel { get; init; }
    public Dictionary<string, double>? Probabilities { get; init; }
    public double? Score { get; init; }
    public int? ClusterId { get; init; }
    public double[]? Distances { get; init; }
    public bool? IsAnomaly { get; init; }
    public double? AnomalyScore { get; init; }
}

public record PredictionMetadata(string ModelName, string ExperimentId, DateTime Timestamp);
