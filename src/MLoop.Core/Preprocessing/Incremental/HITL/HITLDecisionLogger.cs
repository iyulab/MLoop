using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Logs HITL decisions to JSON files for audit trail.
/// </summary>
public sealed class HITLDecisionLogger : IHITLDecisionLogger
{
    private readonly string _logDirectory;
    private readonly ILogger<HITLDecisionLogger> _logger;
    private readonly JsonSerializerOptions _jsonOptions;

    public HITLDecisionLogger(
        string baseDirectory,
        ILogger<HITLDecisionLogger> logger)
    {
        _logDirectory = Path.Combine(baseDirectory, "hitl-decisions");
        _logger = logger;

        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            Converters = { new JsonStringEnumConverter() }
        };

        EnsureLogDirectoryExists();
    }

    /// <inheritdoc />
    public async Task LogDecisionAsync(HITLDecisionLog log)
    {
        var fileName = $"{log.Question.Id}_{DateTime.UtcNow:yyyyMMddHHmmss}.json";
        var filePath = Path.Combine(_logDirectory, fileName);

        try
        {
            var json = JsonSerializer.Serialize(log, _jsonOptions);
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation(
                "Logged HITL decision {QuestionId} to {FilePath}",
                log.Question.Id, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to log HITL decision {QuestionId}",
                log.Question.Id);
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByRuleAsync(string ruleId)
    {
        var allLogs = await LoadAllLogsAsync();
        return allLogs
            .Where(log => log.Question.RelatedRule.Id == ruleId)
            .OrderBy(log => log.LoggedAt)
            .ToList();
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByTimeRangeAsync(
        DateTime startTime,
        DateTime endTime)
    {
        var allLogs = await LoadAllLogsAsync();
        return allLogs
            .Where(log => log.LoggedAt >= startTime &&
                          log.LoggedAt <= endTime)
            .OrderBy(log => log.LoggedAt)
            .ToList();
    }

    /// <inheritdoc />
    public async Task<HITLDecisionSummary> GetDecisionSummaryAsync()
    {
        var allLogs = await LoadAllLogsAsync();

        if (allLogs.Count == 0)
        {
            return new HITLDecisionSummary
            {
                TotalDecisions = 0,
                RecommendationsFollowed = 0,
                RecommendationsOverridden = 0,
                AverageDecisionTimeSeconds = 0,
                EarliestDecision = DateTime.MinValue,
                LatestDecision = DateTime.MinValue
            };
        }

        var recommendationsFollowed = allLogs.Count(log =>
            log.Answer.SelectedOption == log.Question.RecommendedOption);

        var recommendationsOverridden = allLogs.Count(log =>
            !string.IsNullOrEmpty(log.Question.RecommendedOption) &&
            log.Answer.SelectedOption != log.Question.RecommendedOption);

        var avgDecisionTime = allLogs.Average(log => log.Answer.TimeToDecide.TotalSeconds);

        var decisionTypeDistribution = allLogs
            .GroupBy(log => log.Question.Type.ToString())
            .ToDictionary(g => g.Key, g => g.Count());

        // Extract action from ApprovedRule's SuggestedAction field
        var actionDistribution = allLogs
            .Select(log =>
            {
                var action = log.Question.Options
                    .FirstOrDefault(o => o.Key == log.Answer.SelectedOption)?.Action
                    ?? ActionType.KeepAsIs;
                return action;
            })
            .GroupBy(action => action)
            .ToDictionary(g => g.Key, g => g.Count());

        return new HITLDecisionSummary
        {
            TotalDecisions = allLogs.Count,
            RecommendationsFollowed = recommendationsFollowed,
            RecommendationsOverridden = recommendationsOverridden,
            AverageDecisionTimeSeconds = avgDecisionTime,
            DecisionTypeDistribution = decisionTypeDistribution,
            ActionDistribution = actionDistribution,
            EarliestDecision = allLogs.Min(log => log.LoggedAt),
            LatestDecision = allLogs.Max(log => log.LoggedAt)
        };
    }

    private async Task<List<HITLDecisionLog>> LoadAllLogsAsync()
    {
        var logs = new List<HITLDecisionLog>();

        if (!Directory.Exists(_logDirectory))
        {
            return logs;
        }

        var jsonFiles = Directory.GetFiles(_logDirectory, "*.json");

        foreach (var file in jsonFiles)
        {
            try
            {
                var json = await File.ReadAllTextAsync(file);
                var log = JsonSerializer.Deserialize<HITLDecisionLog>(json, _jsonOptions);

                if (log != null)
                {
                    logs.Add(log);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Failed to load decision log from {FilePath}. Skipping.",
                    file);
            }
        }

        return logs;
    }

    private void EnsureLogDirectoryExists()
    {
        if (!Directory.Exists(_logDirectory))
        {
            Directory.CreateDirectory(_logDirectory);
            _logger.LogInformation("Created HITL decision log directory: {Directory}", _logDirectory);
        }
    }
}
