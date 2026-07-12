namespace MLoop.Core.Prediction;

/// <summary>
/// Thrown by the all-degenerate output guard (<see cref="PredictionService"/>) when a scored view
/// produced a row per input but EVERY row's defining output field is null/non-finite — a fabricated
/// "nothing to report" result (the D20~D26 silent-failure family: task/model mismatch, or a degenerate
/// model that scores NaN for every input). A dedicated type (rather than a bare
/// <see cref="InvalidOperationException"/>) lets call sites that add the guard to previously
/// guard-free paths (e.g. the CLI CSV writer) rethrow the intended verdict while treating any other
/// materialization failure as "cannot assess" — without sniffing exception messages. Derives from
/// <see cref="InvalidOperationException"/> so existing handlers keep working.
/// </summary>
public sealed class DegeneratePredictionException : InvalidOperationException
{
    public DegeneratePredictionException(string message) : base(message)
    {
    }
}
