using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Core.Memory;
using MLoop.AIAgent.Core.Memory.Models;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Enhanced data analyzer that combines structural analysis with memory-based recommendations.
/// Uses composition to wrap DataAnalyzer and enrich results with pattern memory insights.
/// </summary>
public class IntelligentDataAnalyzer
{
    private readonly DataAnalyzer _analyzer;
    private readonly DatasetPatternMemoryService _patternMemory;
    private readonly FailureCaseLearningService _failureLearning;
    private readonly ILogger<IntelligentDataAnalyzer> _logger;

    public IntelligentDataAnalyzer(
        DatasetPatternMemoryService patternMemory,
        FailureCaseLearningService failureLearning,
        ILogger<IntelligentDataAnalyzer> logger)
    {
        _analyzer = new DataAnalyzer();
        _patternMemory = patternMemory ?? throw new ArgumentNullException(nameof(patternMemory));
        _failureLearning = failureLearning ?? throw new ArgumentNullException(nameof(failureLearning));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Analyzes a data file and enriches results with memory-based recommendations.
    /// </summary>
    /// <param name="filePath">Path to the data file.</param>
    /// <param name="labelColumn">Optional label column name for supervised learning tasks.</param>
    /// <param name="taskType">Optional ML task type (regression, binary-classification, etc.).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Intelligent analysis result with recommendations and warnings.</returns>
    public async Task<IntelligentAnalysisResult> AnalyzeWithMemoryAsync(
        string filePath,
        string? labelColumn = null,
        string? taskType = null,
        CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Starting intelligent analysis for {FilePath}", filePath);

        // Step 1: Perform basic structural analysis
        var report = await _analyzer.AnalyzeAsync(filePath);

        // Step 2: Create fingerprint from analysis report
        var fingerprint = DatasetFingerprint.FromAnalysisReport(report, labelColumn);
        _logger.LogDebug(
            "Created fingerprint: {Columns} columns, {Rows} rows, {SizeCategory}",
            fingerprint.ColumnNames.Count,
            fingerprint.RowCount,
            fingerprint.SizeCategory);

        // Step 3: Check dataset compatibility for ML training
        var compatibilityChecker = new DatasetCompatibilityChecker();
        var compatibility = compatibilityChecker.Check(report, labelColumn, taskType);
        
        if (!compatibility.IsCompatible)
        {
            _logger.LogWarning(
                "Dataset has {Count} critical compatibility issues",
                compatibility.CriticalIssues.Count());
        }

        // Step 4: Query pattern memory for similar datasets
        var recommendations = await _patternMemory.FindSimilarPatternsAsync(
            fingerprint,
            topK: 3,
            minScore: 0.5f,
            cancellationToken);

        _logger.LogDebug("Found {Count} similar pattern recommendations", recommendations.Count);

        // Step 5: Check for potential failure risks
        var datasetInfo = new DatasetInfo
        {
            Fingerprint = fingerprint,
            CurrentPhase = "analysis"
        };

        var warnings = await _failureLearning.CheckForSimilarFailuresAsync(
            datasetInfo,
            topK: 3,
            minScore: 0.6f,
            cancellationToken);

        if (warnings.Count > 0)
        {
            _logger.LogWarning(
                "Found {Count} potential failure risks for dataset",
                warnings.Count);
        }

        // Step 6: Build and return enriched result
        return new IntelligentAnalysisResult
        {
            Report = report,
            Fingerprint = fingerprint,
            Recommendations = recommendations,
            Warnings = warnings,
            Compatibility = compatibility,
            HasMemoryInsights = recommendations.Count > 0 || warnings.Count > 0
        };
    }

    /// <summary>
    /// Stores a successful processing outcome for future recommendations.
    /// </summary>
    /// <param name="fingerprint">Dataset fingerprint.</param>
    /// <param name="outcome">Processing outcome to store.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StoreSuccessfulOutcomeAsync(
        DatasetFingerprint fingerprint,
        ProcessingOutcome outcome,
        CancellationToken cancellationToken = default)
    {
        await _patternMemory.StorePatternAsync(fingerprint, outcome, cancellationToken);
        _logger.LogInformation(
            "Stored successful processing pattern for dataset with {Columns} columns",
            fingerprint.ColumnNames.Count);
    }
}

/// <summary>
/// Result of intelligent data analysis with memory-based insights.
/// </summary>
public sealed class IntelligentAnalysisResult
{
    /// <summary>
    /// Basic data analysis report.
    /// </summary>
    public required DataAnalysisReport Report { get; init; }

    /// <summary>
    /// Dataset fingerprint for pattern matching.
    /// </summary>
    public required DatasetFingerprint Fingerprint { get; init; }

    /// <summary>
    /// Recommendations from similar dataset processing patterns.
    /// </summary>
    public List<ProcessingRecommendation> Recommendations { get; init; } = [];

    /// <summary>
    /// Warnings based on historical failure cases.
    /// </summary>
    public List<FailureWarning> Warnings { get; init; } = [];

    /// <summary>
    /// Compatibility check result for ML training.
    /// </summary>
    public CompatibilityResult? Compatibility { get; init; }

    /// <summary>
    /// Whether any memory-based insights were found.
    /// </summary>
    public bool HasMemoryInsights { get; init; }

    /// <summary>
    /// Whether the dataset is ready for ML training.
    /// </summary>
    public bool IsMLReady => Compatibility?.IsCompatible ?? true;

    /// <summary>
    /// Gets the top recommendation if available.
    /// </summary>
    public ProcessingRecommendation? TopRecommendation =>
        Recommendations.OrderByDescending(r => r.SimilarityScore).FirstOrDefault();

    /// <summary>
    /// Gets high-priority warnings (>= 70% similarity).
    /// </summary>
    public IEnumerable<FailureWarning> HighPriorityWarnings =>
        Warnings.Where(w => w.SimilarityScore >= 0.7f);

    /// <summary>
    /// Generates a summary of memory-based insights.
    /// </summary>
    public string GetInsightsSummary()
    {
        var parts = new List<string>();

        if (Recommendations.Count > 0)
        {
            var top = TopRecommendation;
            if (top != null)
            {
                parts.Add($"Found {Recommendations.Count} similar patterns (best match: {top.SimilarityScore:P0})");
                if (top.RecommendedSteps.Count > 0)
                {
                    parts.Add($"Suggested steps: {string.Join(", ", top.RecommendedSteps.Select(s => s.Type))}");
                }
            }
        }

        if (Warnings.Count > 0)
        {
            var highPriority = HighPriorityWarnings.ToList();
            if (highPriority.Count > 0)
            {
                parts.Add($"Warning: {highPriority.Count} high-priority failure risks detected");
            }
        }

        return parts.Count > 0 
            ? string.Join(". ", parts) 
            : "No memory-based insights available";
    }
}
