namespace MLoop.Ops.Interfaces;

/// <summary>
/// Compares model performance for promotion decisions.
/// </summary>
public interface IModelComparer
{
    /// <summary>
    /// Compares two experiments and recommends which to promote.
    /// </summary>
    /// <param name="modelName">Model name</param>
    /// <param name="candidateExpId">Candidate experiment to evaluate</param>
    /// <param name="baselineExpId">Baseline experiment (current production)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<ComparisonResult> CompareAsync(
        string modelName,
        string candidateExpId,
        string baselineExpId,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Result of comparing two models.
/// </summary>
public record ComparisonResult(
    string CandidateExpId,
    string BaselineExpId,
    bool CandidateIsBetter,
    double CandidateScore,
    double BaselineScore,
    double Improvement,
    IReadOnlyDictionary<string, MetricComparison> MetricDetails,
    string Recommendation);

/// <summary>
/// Comparison of a single metric.
/// </summary>
public record MetricComparison(
    string MetricName,
    double CandidateValue,
    double BaselineValue,
    double Difference,
    bool IsBetter);
