namespace MLoop.DataStore.Interfaces;

/// <summary>
/// Feedback named a prediction id that no prediction log holds.
/// </summary>
public sealed class PredictionNotFoundException(string predictionId)
    : InvalidOperationException(
        $"Prediction with ID '{predictionId}' not found in logs. " +
        "Ensure the prediction was logged and try again.")
{
    /// <summary>The id that was not found.</summary>
    public string PredictionId { get; } = predictionId;
}
