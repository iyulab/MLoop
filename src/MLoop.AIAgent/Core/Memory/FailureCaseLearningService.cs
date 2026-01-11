using MemoryIndexer.Interfaces;
using MemoryIndexer.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Core.Memory.Models;

namespace MLoop.AIAgent.Core.Memory;

/// <summary>
/// T5.2: Failure Case Learning Service.
/// Stores failure cases and their resolutions for proactive warning and resolution suggestion.
/// Uses episodic memory to capture specific failure events with temporal context.
/// </summary>
public sealed class FailureCaseLearningService
{
    private const string FailureUserId = "mloop:failures";
    private readonly IMemoryStore _store;
    private readonly IEmbeddingService _embedding;
    private readonly ILogger<FailureCaseLearningService> _logger;

    public FailureCaseLearningService(
        IMemoryStore store,
        IEmbeddingService embedding,
        ILogger<FailureCaseLearningService>? logger = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
        _logger = logger ?? NullLogger<FailureCaseLearningService>.Instance;
    }

    /// <summary>
    /// Stores a failure case with its resolution for future learning.
    /// </summary>
    /// <param name="context">Failure context including error type, message, and dataset state.</param>
    /// <param name="resolution">Resolution that fixed the failure.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StoreFailureAsync(
        FailureContext context,
        Resolution resolution,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(resolution);

        var content = $"{context.Describe()} Resolution: {resolution.Describe()}";

        var memory = new MemoryUnit
        {
            UserId = FailureUserId,
            Type = MemoryType.Episodic, // Episodic for specific failure events
            Scope = Scope.User,
            Tier = Tier.Long,
            Content = content,
            ImportanceScore = CalculateImportance(context, resolution),
            Stability = MemoryStability.Stable
        };

        // Store structured data in metadata
        memory.SetMetadata("context", context);
        memory.SetMetadata("resolution", resolution);
        memory.SetMetadata("errorType", context.ErrorType);
        memory.SetMetadata("phase", context.Phase);
        memory.SetMetadata("version", "1.0");

        // Generate embedding for semantic search
        var embeddingResult = await _embedding.GenerateEmbeddingAsync(content, cancellationToken);
        memory.Embedding = embeddingResult;

        await _store.StoreAsync(memory, cancellationToken);

        _logger.LogInformation(
            "Stored failure case: {ErrorType} in {Phase}, Resolution verified: {Verified}",
            context.ErrorType,
            context.Phase,
            resolution.Verified);
    }

    /// <summary>
    /// Checks for similar past failures that might affect current processing.
    /// Returns proactive warnings based on past failure patterns.
    /// </summary>
    /// <param name="datasetInfo">Current dataset information.</param>
    /// <param name="topK">Maximum number of warnings.</param>
    /// <param name="minScore">Minimum similarity score.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of warnings from similar past failures.</returns>
    public async Task<List<FailureWarning>> CheckForSimilarFailuresAsync(
        DatasetInfo datasetInfo,
        int topK = 3,
        float minScore = 0.6f,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(datasetInfo);

        var queryText = datasetInfo.ToQueryString();
        var queryEmbedding = await _embedding.GenerateEmbeddingAsync(queryText, cancellationToken);

        var searchOptions = new MemorySearchOptions
        {
            UserId = FailureUserId,
            Types = [MemoryType.Episodic],
            Limit = topK,
            MinScore = minScore
        };

        var results = await _store.SearchAsync(queryEmbedding, searchOptions, cancellationToken);

        var warnings = new List<FailureWarning>();

        foreach (var result in results)
        {
            if (result.Memory.TryGetMetadata<FailureContext>("context", out var context) &&
                result.Memory.TryGetMetadata<Resolution>("resolution", out var resolution))
            {
                warnings.Add(new FailureWarning
                {
                    Context = context,
                    Resolution = resolution,
                    SimilarityScore = result.Score
                });
            }
        }

        if (warnings.Count > 0)
        {
            _logger.LogWarning(
                "Found {Count} potential failure warnings for current dataset processing",
                warnings.Count);
        }

        return warnings;
    }

    /// <summary>
    /// Finds a resolution for a specific error based on past failures.
    /// </summary>
    /// <param name="errorType">Type of error.</param>
    /// <param name="errorMessage">Error message.</param>
    /// <param name="datasetContext">Optional dataset context.</param>
    /// <param name="minConfidence">Minimum confidence threshold.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Resolution suggestion or null if none found.</returns>
    public async Task<ResolutionSuggestion?> FindResolutionAsync(
        string errorType,
        string errorMessage,
        DatasetFingerprint? datasetContext = null,
        float minConfidence = 0.7f,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(errorType))
            throw new ArgumentException("Error type is required", nameof(errorType));

        var queryParts = new List<string>
        {
            $"Error: {errorType}",
            $"Message: {errorMessage}"
        };

        if (datasetContext != null)
            queryParts.Add($"Dataset: {datasetContext.Describe()}");

        var queryText = string.Join(". ", queryParts);
        var queryEmbedding = await _embedding.GenerateEmbeddingAsync(queryText, cancellationToken);

        var searchOptions = new MemorySearchOptions
        {
            UserId = FailureUserId,
            Types = [MemoryType.Episodic],
            Limit = 1,
            MinScore = minConfidence
        };

        var results = await _store.SearchAsync(queryEmbedding, searchOptions, cancellationToken);

        if (results.Count == 0)
        {
            _logger.LogDebug("No resolution found for error: {ErrorType}", errorType);
            return null;
        }

        var topResult = results[0];
        if (topResult.Memory.TryGetMetadata<FailureContext>("context", out var context) &&
            topResult.Memory.TryGetMetadata<Resolution>("resolution", out var resolution))
        {
            _logger.LogInformation(
                "Found resolution suggestion for {ErrorType} with confidence {Score:P0}",
                errorType,
                topResult.Score);

            return new ResolutionSuggestion
            {
                OriginalContext = context,
                SuggestedResolution = resolution,
                Confidence = topResult.Score,
                MatchedErrorType = context?.ErrorType == errorType
            };
        }

        return null;
    }

    /// <summary>
    /// Gets statistics about stored failure cases.
    /// </summary>
    public async Task<FailureStatistics> GetStatisticsAsync(CancellationToken cancellationToken = default)
    {
        var typeCounts = await _store.GetTypeCountsAsync(FailureUserId, cancellationToken);
        var episodicCount = typeCounts.GetValueOrDefault(MemoryType.Episodic, 0);

        return new FailureStatistics
        {
            TotalFailures = episodicCount,
            LastUpdated = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Gets recent failures for a specific phase.
    /// </summary>
    /// <param name="phase">Processing phase to filter by.</param>
    /// <param name="limit">Maximum number of results.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<List<FailureCase>> GetRecentFailuresAsync(
        string? phase = null,
        int limit = 10,
        CancellationToken cancellationToken = default)
    {
        // Use a generic query to find recent failures
        var queryText = string.IsNullOrEmpty(phase)
            ? "Recent failure cases"
            : $"Failure in {phase} phase";

        var queryEmbedding = await _embedding.GenerateEmbeddingAsync(queryText, cancellationToken);

        var searchOptions = new MemorySearchOptions
        {
            UserId = FailureUserId,
            Types = [MemoryType.Episodic],
            Limit = limit,
            MinScore = 0.0f // Get all matches
        };

        var results = await _store.SearchAsync(queryEmbedding, searchOptions, cancellationToken);

        var failures = new List<FailureCase>();

        foreach (var result in results)
        {
            if (result.Memory.TryGetMetadata<FailureContext>("context", out var context) &&
                result.Memory.TryGetMetadata<Resolution>("resolution", out var resolution))
            {
                // Filter by phase if specified
                if (!string.IsNullOrEmpty(phase) &&
                    !string.Equals(context?.Phase, phase, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                failures.Add(new FailureCase
                {
                    Context = context,
                    Resolution = resolution
                });
            }
        }

        return failures;
    }

    private static float CalculateImportance(FailureContext context, Resolution resolution)
    {
        var baseScore = 0.5f;

        // Verified resolutions are more important
        if (resolution.Verified)
            baseScore += 0.2f;

        // Failures with clear root cause are more valuable
        if (!string.IsNullOrEmpty(resolution.RootCause))
            baseScore += 0.1f;

        // Failures with prevention advice are more valuable
        if (!string.IsNullOrEmpty(resolution.PreventionAdvice))
            baseScore += 0.1f;

        // Failures with dataset context are more useful for pattern matching
        if (context.DatasetContext != null)
            baseScore += 0.1f;

        return Math.Clamp(baseScore, 0f, 1f);
    }
}

/// <summary>
/// Suggestion for resolving a failure based on past cases.
/// </summary>
public sealed class ResolutionSuggestion
{
    /// <summary>
    /// Original failure context that was resolved.
    /// </summary>
    public FailureContext? OriginalContext { get; set; }

    /// <summary>
    /// Suggested resolution from past case.
    /// </summary>
    public Resolution? SuggestedResolution { get; set; }

    /// <summary>
    /// Confidence in the suggestion (0.0 to 1.0).
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Whether the error type exactly matches.
    /// </summary>
    public bool MatchedErrorType { get; set; }

    /// <summary>
    /// Human-readable suggestion message.
    /// </summary>
    public string Message => $"Based on similar failure ({Confidence:P0} match): " +
                             $"Root cause was '{SuggestedResolution?.RootCause}'. " +
                             $"Suggested fix: {SuggestedResolution?.FixDescription}";
}

/// <summary>
/// Statistics about stored failure cases.
/// </summary>
public sealed class FailureStatistics
{
    public int TotalFailures { get; set; }
    public DateTime LastUpdated { get; set; }
}

/// <summary>
/// A stored failure case with context and resolution.
/// </summary>
public sealed class FailureCase
{
    public FailureContext? Context { get; set; }
    public Resolution? Resolution { get; set; }
}
