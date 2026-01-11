using MemoryIndexer;
using MemoryIndexer.Configuration;
using MemoryIndexer.Interfaces;
using MemoryIndexer.Sdk.Extensions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Configuration;
using MLoop.AIAgent.Core.Memory;
using MLoop.AIAgent.Core.Memory.Models;
using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.Memory;

/// <summary>
/// Collects training metadata for future pattern recognition and learning.
/// Operates silently in the background - never throws, never blocks, no UI output.
/// </summary>
public sealed class TrainingMemoryCollector : IDisposable
{
    private readonly ServiceProvider? _serviceProvider;
    private readonly DatasetPatternMemoryService? _patternService;
    private readonly FailureCaseLearningService? _failureService;
    private readonly ILogger<TrainingMemoryCollector> _logger;
    private readonly bool _initialized;

    public TrainingMemoryCollector(string projectRoot, ILogger<TrainingMemoryCollector>? logger = null)
    {
        _logger = logger ?? NullLogger<TrainingMemoryCollector>.Instance;

        try
        {
            var config = MemoryConfig.CreateDefault(projectRoot);

            if (!config.Enabled)
            {
                _initialized = false;
                return;
            }

            // Ensure storage directory exists
            Directory.CreateDirectory(config.StorageDirectory);

            var services = new ServiceCollection();

            // Add logging
            services.AddLogging(builder => builder.AddDebug());

            // Add memory-indexer with mock embedding (no external dependencies)
            services.AddMemoryIndexer(options =>
            {
                options.Storage.Type = StorageType.SqliteVec;
                options.Storage.ConnectionString = $"Data Source={config.GetDatabasePath()}";
                options.Storage.VectorDimensions = 384;

                // Use mock embedding for zero-config background operation
                options.Embedding.Provider = EmbeddingProvider.Mock;
                options.Embedding.Dimensions = 384;

                options.Search.DefaultLimit = 10;
            });

            _serviceProvider = services.BuildServiceProvider();

            var store = _serviceProvider.GetRequiredService<IMemoryStore>();
            var embedding = _serviceProvider.GetRequiredService<IEmbeddingService>();

            _patternService = new DatasetPatternMemoryService(
                store,
                embedding,
                _serviceProvider.GetService<ILogger<DatasetPatternMemoryService>>());

            _failureService = new FailureCaseLearningService(
                store,
                embedding,
                _serviceProvider.GetService<ILogger<FailureCaseLearningService>>());

            _initialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to initialize training memory collector");
            _initialized = false;
        }
    }

    /// <summary>
    /// Stores successful training outcome for future pattern recognition.
    /// </summary>
    public async Task StoreSuccessfulTrainingAsync(
        TrainingConfig config,
        TrainingResult result,
        CancellationToken cancellationToken = default)
    {
        if (!_initialized || _patternService == null)
            return;

        try
        {
            // Create fingerprint from training context
            var fingerprint = CreateFingerprint(config, result);

            // Create outcome from result
            var outcome = new ProcessingOutcome
            {
                Success = true,
                ProcessingTimeMs = (long)(result.TrainingTimeSeconds * 1000),
                BestTrainer = result.BestTrainer,
                PerformanceMetrics = result.Metrics,
                ProcessedAt = DateTime.UtcNow,
                Steps =
                [
                    new PreprocessingStep
                    {
                        Type = "AutoML Training",
                        Order = 1,
                        Parameters = new Dictionary<string, object>
                        {
                            ["Task"] = config.Task,
                            ["TimeLimit"] = config.TimeLimitSeconds,
                            ["Metric"] = config.Metric
                        }
                    }
                ]
            };

            await _patternService.StorePatternAsync(fingerprint, outcome, cancellationToken);

            _logger.LogDebug(
                "Stored training pattern: {Hash}, Trainer: {Trainer}",
                fingerprint.Hash,
                result.BestTrainer);
        }
        catch (Exception ex)
        {
            // Silent failure - never interrupt training workflow
            _logger.LogDebug(ex, "Failed to store training pattern");
        }
    }

    /// <summary>
    /// Stores training failure for future prevention learning.
    /// </summary>
    public async Task StoreTrainingFailureAsync(
        TrainingConfig config,
        Exception exception,
        CancellationToken cancellationToken = default)
    {
        if (!_initialized || _failureService == null)
            return;

        try
        {
            // Create a minimal fingerprint for error context
            var datasetFingerprint = new DatasetFingerprint
            {
                LabelColumn = config.LabelColumn,
                DetectedTaskType = config.Task,
                ColumnNames = [],
                RowCount = 0,
                SizeCategory = "Unknown"
            };
            datasetFingerprint.Hash = ComputeSimpleHash(config.Task, config.LabelColumn);

            var context = new FailureContext
            {
                ErrorType = exception.GetType().Name,
                ErrorMessage = exception.Message,
                StackTrace = exception.StackTrace,
                Phase = "Training",
                OccurredAt = DateTime.UtcNow,
                DatasetContext = datasetFingerprint,
                Environment = new Dictionary<string, string>
                {
                    ["Task"] = config.Task,
                    ["TimeLimit"] = config.TimeLimitSeconds.ToString(),
                    ["Metric"] = config.Metric,
                    ["ModelName"] = config.ModelName,
                    ["DataFile"] = config.DataFile
                }
            };

            // Create basic resolution (will be enriched if user provides feedback later)
            var resolution = new Resolution
            {
                RootCause = $"Training failed with {exception.GetType().Name}",
                FixDescription = "Review error message and data quality",
                Verified = false,
                ResolvedAt = DateTime.UtcNow
            };

            await _failureService.StoreFailureAsync(context, resolution, cancellationToken);

            _logger.LogDebug(
                "Stored training failure: {ErrorType} in {Phase}",
                context.ErrorType,
                context.Phase);
        }
        catch (Exception ex)
        {
            // Silent failure - never interrupt error handling
            _logger.LogDebug(ex, "Failed to store training failure");
        }
    }

    private static DatasetFingerprint CreateFingerprint(TrainingConfig config, TrainingResult result)
    {
        // Extract column names from result or config
        var columnNames = new List<string>();

        // Parse schema info from result if available
        if (!string.IsNullOrEmpty(result.SchemaInfo))
        {
            columnNames = result.SchemaInfo.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(c => c.Trim())
                .ToList();
        }

        var fingerprint = new DatasetFingerprint
        {
            ColumnNames = columnNames,
            RowCount = result.RowCount,
            LabelColumn = config.LabelColumn,
            DetectedTaskType = config.Task,
            SizeCategory = result.RowCount switch
            {
                < 1_000 => "Small",
                < 100_000 => "Medium",
                < 1_000_000 => "Large",
                _ => "VeryLarge"
            },
            DomainKeywords = ExtractDomainKeywords(columnNames)
        };

        fingerprint.Hash = ComputeSimpleHash(
            string.Join(",", columnNames),
            config.Task,
            config.LabelColumn,
            fingerprint.SizeCategory);
        return fingerprint;
    }

    private static string ComputeSimpleHash(params string?[] values)
    {
        var combined = string.Join("|", values.Where(v => !string.IsNullOrEmpty(v)));
        var bytes = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(combined));
        return Convert.ToHexString(bytes)[..16];
    }

    private static List<string> ExtractDomainKeywords(List<string> columnNames)
    {
        var keywords = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var domainPatterns = new Dictionary<string, string[]>
        {
            ["finance"] = ["price", "cost", "revenue", "profit", "amount", "balance", "credit", "debit"],
            ["customer"] = ["customer", "client", "user", "member", "subscriber"],
            ["sales"] = ["sales", "order", "quantity", "discount", "product"],
            ["time"] = ["date", "time", "year", "month", "day", "hour", "timestamp"],
            ["location"] = ["address", "city", "state", "country", "zip", "region", "location"],
            ["classification"] = ["category", "type", "class", "label", "status", "churn", "fraud"]
        };

        foreach (var column in columnNames)
        {
            var lowerColumn = column.ToLowerInvariant();
            foreach (var (domain, patterns) in domainPatterns)
            {
                if (patterns.Any(p => lowerColumn.Contains(p)))
                {
                    keywords.Add(domain);
                }
            }
        }

        return keywords.ToList();
    }

    public void Dispose()
    {
        _serviceProvider?.Dispose();
    }
}
