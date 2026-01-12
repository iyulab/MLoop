using System.Text.Json;
using FluentAssertions;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class TimeBasedTriggerTests : IDisposable
{
    private readonly string _testDir;
    private readonly TimeBasedTrigger _trigger;

    public TimeBasedTriggerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _trigger = new TimeBasedTrigger(_testDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDir))
            {
                Directory.Delete(_testDir, recursive: true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    [Fact]
    public async Task EvaluateAsync_ReturnsTrue_WhenNoTrainingHistory()
    {
        // Arrange
        var modelName = "new-model";
        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Scheduled", 30, "Every 30 days")
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults.Should().HaveCount(1);
        result.ConditionResults[0].IsMet.Should().BeTrue();
        result.ConditionResults[0].Details.Should().Contain("No training history");
    }

    [Fact]
    public async Task EvaluateAsync_ReturnsTrue_WhenThresholdExceeded()
    {
        // Arrange
        var modelName = "old-model";
        await CreateExperimentWithTimestamp(modelName, "exp-001", DateTime.UtcNow.AddDays(-45));

        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Scheduled", 30, "Every 30 days")
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults[0].IsMet.Should().BeTrue();
        result.ConditionResults[0].CurrentValue.Should().BeGreaterThanOrEqualTo(45);
    }

    [Fact]
    public async Task EvaluateAsync_ReturnsFalse_WhenThresholdNotExceeded()
    {
        // Arrange
        var modelName = "recent-model";
        await CreateExperimentWithTimestamp(modelName, "exp-001", DateTime.UtcNow.AddDays(-5));

        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Scheduled", 30, "Every 30 days")
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().BeLessThan(10);
    }

    [Fact]
    public async Task EvaluateAsync_UsesLatestCompletedExperiment()
    {
        // Arrange
        var modelName = "multi-exp-model";
        await CreateExperimentWithTimestamp(modelName, "exp-001", DateTime.UtcNow.AddDays(-60));
        await CreateExperimentWithTimestamp(modelName, "exp-002", DateTime.UtcNow.AddDays(-10));
        await CreateExperimentWithTimestamp(modelName, "exp-003", DateTime.UtcNow.AddDays(-40), "Failed");

        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Scheduled", 30, "Every 30 days")
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().BeLessThan(15);
    }

    [Fact]
    public async Task EvaluateAsync_ReturnsNotSupported_ForNonTimeBasedConditions()
    {
        // Arrange
        var modelName = "test-model";
        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.AccuracyDrop, "Accuracy Drop", 0.05),
            new(ConditionType.DataDrift, "Data Drift", 0.1),
            new(ConditionType.FeedbackVolume, "Feedback Volume", 100)
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults.Should().HaveCount(3);
        result.ConditionResults.All(r => !r.IsMet).Should().BeTrue();
        result.ConditionResults.All(r => r.Details!.Contains("not supported")).Should().BeTrue();
    }

    [Fact]
    public async Task EvaluateAsync_HandlesMultipleConditions()
    {
        // Arrange
        var modelName = "multi-condition-model";
        await CreateExperimentWithTimestamp(modelName, "exp-001", DateTime.UtcNow.AddDays(-20));

        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Weekly", 7),
            new(ConditionType.TimeBased, "Monthly", 30)
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults.Should().HaveCount(2);
        result.ConditionResults[0].IsMet.Should().BeTrue(); // 20 > 7
        result.ConditionResults[1].IsMet.Should().BeFalse(); // 20 < 30
    }

    [Fact]
    public async Task GetDefaultConditionsAsync_ReturnsTimeBasedCondition()
    {
        // Act
        var conditions = await _trigger.GetDefaultConditionsAsync("any-model");

        // Assert
        conditions.Should().HaveCount(1);
        conditions[0].Type.Should().Be(ConditionType.TimeBased);
        conditions[0].Threshold.Should().Be(TimeBasedTrigger.DefaultRetrainingIntervalDays);
    }

    [Fact]
    public async Task EvaluateAsync_ProvidesRecommendedAction_WhenShouldRetrain()
    {
        // Arrange
        var modelName = "action-model";
        var conditions = new List<RetrainingCondition>
        {
            new(ConditionType.TimeBased, "Scheduled", 30)
        };

        // Act
        var result = await _trigger.EvaluateAsync(modelName, conditions);

        // Assert
        result.RecommendedAction.Should().NotBeNullOrEmpty();
        result.RecommendedAction.Should().Contain("Retrain");
        result.RecommendedAction.Should().Contain(modelName);
    }

    private async Task CreateExperimentWithTimestamp(
        string modelName,
        string experimentId,
        DateTime timestamp,
        string status = "Completed")
    {
        var expPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant(), "experiments", experimentId);
        Directory.CreateDirectory(expPath);

        var metadata = new
        {
            Timestamp = timestamp,
            Status = status,
            ModelName = modelName,
            ExperimentId = experimentId
        };

        var json = JsonSerializer.Serialize(metadata);
        await File.WriteAllTextAsync(Path.Combine(expPath, "experiment.json"), json);
    }
}
