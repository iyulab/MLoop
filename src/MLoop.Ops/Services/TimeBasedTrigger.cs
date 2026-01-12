using System.Text.Json;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// Simple time-based retraining trigger.
/// Evaluates whether a model should be retrained based on time elapsed since last training.
/// Only supports TimeBased conditions; other conditions are marked as unsupported.
/// </summary>
public sealed class TimeBasedTrigger : IRetrainingTrigger
{
    private readonly string _projectRoot;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    /// <summary>
    /// Default retraining interval in days.
    /// </summary>
    public const int DefaultRetrainingIntervalDays = 30;

    /// <summary>
    /// Creates a new TimeBasedTrigger for the specified project root.
    /// </summary>
    /// <param name="projectRoot">Absolute path to the MLoop project root</param>
    public TimeBasedTrigger(string projectRoot)
    {
        _projectRoot = projectRoot ?? throw new ArgumentNullException(nameof(projectRoot));
    }

    /// <inheritdoc/>
    public async Task<TriggerEvaluation> EvaluateAsync(
        string modelName,
        IEnumerable<RetrainingCondition> conditions,
        CancellationToken cancellationToken = default)
    {
        var conditionList = conditions.ToList();
        var results = new List<ConditionResult>();
        var evaluatedAt = DateTimeOffset.UtcNow;

        // Get the last training timestamp
        var lastTrainingTime = await GetLastTrainingTimeAsync(modelName, cancellationToken);

        foreach (var condition in conditionList)
        {
            var result = EvaluateCondition(condition, lastTrainingTime, evaluatedAt);
            results.Add(result);
        }

        var shouldRetrain = results.Any(r => r.IsMet);
        var metConditions = results.Where(r => r.IsMet).Select(r => r.Condition.Name).ToList();

        string? recommendedAction = null;
        if (shouldRetrain)
        {
            recommendedAction = metConditions.Count > 0
                ? $"Retrain model '{modelName}' - conditions met: {string.Join(", ", metConditions)}"
                : $"Retrain model '{modelName}'";
        }

        return new TriggerEvaluation(
            ShouldRetrain: shouldRetrain,
            ConditionResults: results,
            RecommendedAction: recommendedAction,
            EvaluatedAt: evaluatedAt
        );
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<RetrainingCondition>> GetDefaultConditionsAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        // Default: retrain if 30 days have passed since last training
        var defaultConditions = new List<RetrainingCondition>
        {
            new(
                Type: ConditionType.TimeBased,
                Name: "Scheduled Retraining",
                Threshold: DefaultRetrainingIntervalDays,
                Description: $"Retrain if more than {DefaultRetrainingIntervalDays} days since last training"
            )
        };

        return Task.FromResult<IReadOnlyList<RetrainingCondition>>(defaultConditions);
    }

    private ConditionResult EvaluateCondition(
        RetrainingCondition condition,
        DateTimeOffset? lastTrainingTime,
        DateTimeOffset evaluatedAt)
    {
        switch (condition.Type)
        {
            case ConditionType.TimeBased:
                return EvaluateTimeBasedCondition(condition, lastTrainingTime, evaluatedAt);

            case ConditionType.AccuracyDrop:
            case ConditionType.DataDrift:
            case ConditionType.FeedbackVolume:
            case ConditionType.PerformanceDegradation:
                // These conditions require additional data sources (feedback, drift detection, etc.)
                // Return as "not supported" in this simple implementation
                return new ConditionResult(
                    Condition: condition,
                    IsMet: false,
                    CurrentValue: 0,
                    Details: $"Condition type '{condition.Type}' is not supported by TimeBasedTrigger. " +
                             "Use a full IRetrainingTrigger implementation with required data sources."
                );

            default:
                return new ConditionResult(
                    Condition: condition,
                    IsMet: false,
                    CurrentValue: 0,
                    Details: $"Unknown condition type: {condition.Type}"
                );
        }
    }

    private static ConditionResult EvaluateTimeBasedCondition(
        RetrainingCondition condition,
        DateTimeOffset? lastTrainingTime,
        DateTimeOffset evaluatedAt)
    {
        if (!lastTrainingTime.HasValue)
        {
            // No training history - recommend training
            return new ConditionResult(
                Condition: condition,
                IsMet: true,
                CurrentValue: double.PositiveInfinity,
                Details: "No training history found - initial training recommended"
            );
        }

        var daysSinceLastTraining = (evaluatedAt - lastTrainingTime.Value).TotalDays;
        var thresholdDays = condition.Threshold;
        var isMet = daysSinceLastTraining >= thresholdDays;

        var details = isMet
            ? $"Last training was {daysSinceLastTraining:F1} days ago (threshold: {thresholdDays} days)"
            : $"Last training was {daysSinceLastTraining:F1} days ago (threshold: {thresholdDays} days) - no retraining needed";

        return new ConditionResult(
            Condition: condition,
            IsMet: isMet,
            CurrentValue: daysSinceLastTraining,
            Details: details
        );
    }

    private async Task<DateTimeOffset?> GetLastTrainingTimeAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var experimentsPath = GetExperimentsPath(modelName);

        if (!Directory.Exists(experimentsPath))
        {
            return null;
        }

        DateTimeOffset? latestTimestamp = null;

        foreach (var expDir in Directory.GetDirectories(experimentsPath))
        {
            var metadataPath = Path.Combine(expDir, "experiment.json");

            if (!File.Exists(metadataPath))
                continue;

            try
            {
                var json = await File.ReadAllTextAsync(metadataPath, cancellationToken);
                var metadata = JsonSerializer.Deserialize<ExperimentMetadata>(json, JsonOptions);

                if (metadata != null &&
                    metadata.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                {
                    var timestamp = new DateTimeOffset(metadata.Timestamp, TimeSpan.Zero);

                    if (!latestTimestamp.HasValue || timestamp > latestTimestamp)
                    {
                        latestTimestamp = timestamp;
                    }
                }
            }
            catch
            {
                // Skip experiments with invalid metadata
                continue;
            }
        }

        return latestTimestamp;
    }

    private string GetExperimentsPath(string modelName)
    {
        return Path.Combine(_projectRoot, "models", SanitizeModelName(modelName), "experiments");
    }

    private static string SanitizeModelName(string modelName)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sanitized = modelName;

        foreach (var c in invalid)
        {
            sanitized = sanitized.Replace(c, '_');
        }

        return sanitized.ToLowerInvariant();
    }

    /// <summary>
    /// Minimal experiment metadata for reading training timestamps.
    /// </summary>
    private sealed class ExperimentMetadata
    {
        public DateTime Timestamp { get; set; }
        public string Status { get; set; } = string.Empty;
    }
}
