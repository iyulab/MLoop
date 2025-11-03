namespace MLoop.Core.Contracts;

/// <summary>
/// Interface for making predictions with trained models
/// </summary>
public interface IPredictionEngine
{
    /// <summary>
    /// Make predictions on input data
    /// </summary>
    /// <param name="modelPath">Path to the trained model file (.zip)</param>
    /// <param name="inputDataPath">Path to input CSV data</param>
    /// <param name="outputPath">Path to save predictions CSV</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Number of predictions made</returns>
    Task<int> PredictAsync(
        string modelPath,
        string inputDataPath,
        string outputPath,
        CancellationToken cancellationToken = default);
}
