namespace MLoop.Core.Models;

/// <summary>
/// Configuration for model training
/// </summary>
public class TrainingConfig
{
    public required string DataFile { get; init; }
    public required string LabelColumn { get; init; }
    public required string Task { get; init; }
    public int TimeLimitSeconds { get; init; } = 300;
    public string Metric { get; init; } = "accuracy";
    public double TestSplit { get; init; } = 0.2;
}
