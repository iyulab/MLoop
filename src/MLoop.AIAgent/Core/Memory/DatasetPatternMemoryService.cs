using MemoryIndexer.Interfaces;
using MemoryIndexer.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Core.Memory.Models;
using System.Text.Json;

namespace MLoop.AIAgent.Core.Memory;

/// <summary>
/// T5.1: Dataset Pattern Memory Service.
/// Stores and retrieves successful dataset processing patterns using procedural memory.
/// </summary>
public sealed class DatasetPatternMemoryService
{
    private const string PatternUserId = "mloop:patterns";
    private readonly IMemoryStore _store;
    private readonly IEmbeddingService _embedding;
    private readonly ILogger<DatasetPatternMemoryService> _logger;

    public DatasetPatternMemoryService(
        IMemoryStore store,
        IEmbeddingService embedding,
        ILogger<DatasetPatternMemoryService>? logger = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
        _logger = logger ?? NullLogger<DatasetPatternMemoryService>.Instance;
    }

    /// <summary>
    /// Stores a successful dataset processing pattern.
    /// </summary>
    /// <param name="fingerprint">Dataset fingerprint.</param>
    /// <param name="outcome">Processing outcome with steps and metrics.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StorePatternAsync(
        DatasetFingerprint fingerprint,
        ProcessingOutcome outcome,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(fingerprint);
        ArgumentNullException.ThrowIfNull(outcome);

        if (!outcome.Success)
        {
            _logger.LogWarning("Skipping storage of failed processing outcome");
            return;
        }

        var content = $"{fingerprint.Describe()} {outcome.Describe()}";

        var memory = new MemoryUnit
        {
            UserId = PatternUserId,
            Type = MemoryType.Procedural,
            Scope = Scope.User,
            Tier = Tier.Long,
            Content = content,
            ImportanceScore = CalculateImportance(outcome),
            Stability = MemoryStability.Stable
        };

        // Store structured data in metadata
        memory.SetMetadata("fingerprint", fingerprint);
        memory.SetMetadata("outcome", outcome);
        memory.SetMetadata("fingerprintHash", fingerprint.Hash ?? "");
        memory.SetMetadata("version", "1.0");

        // Generate embedding for semantic search
        var embeddingResult = await _embedding.GenerateEmbeddingAsync(content, cancellationToken);
        memory.Embedding = embeddingResult;

        await _store.StoreAsync(memory, cancellationToken);

        _logger.LogInformation(
            "Stored dataset pattern: {Hash}, Steps: {StepCount}, Metrics: {MetricCount}",
            fingerprint.Hash,
            outcome.Steps.Count,
            outcome.PerformanceMetrics?.Count ?? 0);
    }

    /// <summary>
    /// Finds similar patterns for a new dataset.
    /// </summary>
    /// <param name="fingerprint">New dataset fingerprint.</param>
    /// <param name="topK">Maximum number of results.</param>
    /// <param name="minScore">Minimum similarity score.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of recommendations from similar patterns.</returns>
    public async Task<List<ProcessingRecommendation>> FindSimilarPatternsAsync(
        DatasetFingerprint fingerprint,
        int topK = 5,
        float minScore = 0.5f,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(fingerprint);

        var queryText = fingerprint.Describe();
        var queryEmbedding = await _embedding.GenerateEmbeddingAsync(queryText, cancellationToken);

        var searchOptions = new MemorySearchOptions
        {
            UserId = PatternUserId,
            Types = [MemoryType.Procedural],
            Limit = topK,
            MinScore = minScore
        };

        var results = await _store.SearchAsync(queryEmbedding, searchOptions, cancellationToken);

        var recommendations = new List<ProcessingRecommendation>();

        foreach (var result in results)
        {
            if (result.Memory.TryGetMetadata<DatasetFingerprint>("fingerprint", out var storedFingerprint) &&
                result.Memory.TryGetMetadata<ProcessingOutcome>("outcome", out var outcome))
            {
                recommendations.Add(new ProcessingRecommendation
                {
                    SimilarFingerprint = storedFingerprint,
                    Outcome = outcome,
                    SimilarityScore = result.Score
                });
            }
        }

        _logger.LogDebug(
            "Found {Count} similar patterns for dataset with {Columns} columns",
            recommendations.Count,
            fingerprint.ColumnNames.Count);

        return recommendations;
    }

    /// <summary>
    /// Gets the best recommendation for a dataset.
    /// </summary>
    /// <param name="fingerprint">New dataset fingerprint.</param>
    /// <param name="minConfidence">Minimum confidence threshold.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Best recommendation or null if none found.</returns>
    public async Task<ProcessingRecommendation?> GetRecommendationAsync(
        DatasetFingerprint fingerprint,
        float minConfidence = 0.7f,
        CancellationToken cancellationToken = default)
    {
        var recommendations = await FindSimilarPatternsAsync(
            fingerprint,
            topK: 1,
            minScore: minConfidence,
            cancellationToken);

        return recommendations.FirstOrDefault();
    }

    /// <summary>
    /// Gets pattern statistics.
    /// </summary>
    public async Task<PatternStatistics> GetStatisticsAsync(CancellationToken cancellationToken = default)
    {
        var typeCounts = await _store.GetTypeCountsAsync(PatternUserId, cancellationToken);
        var proceduralCount = typeCounts.GetValueOrDefault(MemoryType.Procedural, 0);

        return new PatternStatistics
        {
            TotalPatterns = proceduralCount,
            LastUpdated = DateTime.UtcNow
        };
    }

    private static float CalculateImportance(ProcessingOutcome outcome)
    {
        // Higher importance for successful outcomes with good metrics
        var baseScore = 0.5f;

        if (outcome.PerformanceMetrics?.TryGetValue("Accuracy", out var accuracy) == true)
            baseScore += (float)(accuracy * 0.3);
        else if (outcome.PerformanceMetrics?.TryGetValue("RSquared", out var r2) == true)
            baseScore += (float)(r2 * 0.3);

        if (outcome.Steps.Count > 0)
            baseScore += Math.Min(0.2f, outcome.Steps.Count * 0.02f);

        return Math.Clamp(baseScore, 0f, 1f);
    }
}

/// <summary>
/// Statistics about stored patterns.
/// </summary>
public sealed class PatternStatistics
{
    public int TotalPatterns { get; set; }
    public DateTime LastUpdated { get; set; }
}
