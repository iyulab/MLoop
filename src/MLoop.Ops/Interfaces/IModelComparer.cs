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
/// Result of comparing two models — judged on one metric, the one the candidate's run optimized
/// (<see cref="Metric"/>), in its direction; the other metrics both experiments recorded are listed, not counted.
/// </summary>
/// <param name="CandidateScore">The candidate's value of <see cref="Metric"/>.</param>
/// <param name="BaselineScore">The baseline's value of <see cref="Metric"/>.</param>
/// <param name="Improvement">How much better the candidate is on <see cref="Metric"/>, in percent of the baseline — positive is better whichever way the metric points.</param>
/// <param name="MetricDetails">Every metric both experiments recorded.</param>
/// <param name="Metric">The metric judged on; <see langword="null"/> when the two share none to judge on.</param>
public record ComparisonResult(
    string CandidateExpId,
    string BaselineExpId,
    bool CandidateIsBetter,
    double CandidateScore,
    double BaselineScore,
    double Improvement,
    IReadOnlyDictionary<string, MetricComparison> MetricDetails,
    string Recommendation,
    string? Metric = null);

/// <summary>
/// Comparison of a single metric.
/// </summary>
public record MetricComparison(
    string MetricName,
    double CandidateValue,
    double BaselineValue,
    double Difference,
    bool IsBetter);
