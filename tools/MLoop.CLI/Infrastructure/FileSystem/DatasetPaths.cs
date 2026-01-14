namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Represents the discovered dataset file paths following convention
/// </summary>
public class DatasetPaths
{
    /// <summary>
    /// Path to training data file (required)
    /// </summary>
    public required string TrainPath { get; set; }

    /// <summary>
    /// Path to validation data file (optional)
    /// </summary>
    public string? ValidationPath { get; set; }

    /// <summary>
    /// Path to test data file (optional)
    /// </summary>
    public string? TestPath { get; set; }

    /// <summary>
    /// Path to prediction data file (optional)
    /// </summary>
    public string? PredictPath { get; set; }
}
