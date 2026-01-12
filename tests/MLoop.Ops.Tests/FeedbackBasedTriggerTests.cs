using System.Text.Json;
using FluentAssertions;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class FeedbackBasedTriggerTests : IDisposable
{
    private readonly string _testDir;
    private readonly FeedbackBasedTrigger _trigger;
    private readonly FileFeedbackCollector _feedbackCollector;
    private readonly FilePredictionLogger _logger;

    public FeedbackBasedTriggerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _feedbackCollector = new FileFeedbackCollector(_testDir);
        _logger = new FilePredictionLogger(_testDir);
        _trigger = new FeedbackBasedTrigger(_feedbackCollector);
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
    public async Task EvaluateAsync_AccuracyDrop_TriggersWhenBelowThreshold()
    {
        // Arrange - 2 correct, 3 wrong = 40% accuracy
        await CreateFeedbackScenario("test-model", correctCount: 2, wrongCount: 3);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.5)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults.Should().HaveCount(1);
        result.ConditionResults[0].IsMet.Should().BeTrue();
        result.ConditionResults[0].CurrentValue.Should().BeApproximately(0.4, 0.01);
        result.RecommendedAction.Should().Contain("accuracy_check");
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_DoesNotTriggerWhenAboveThreshold()
    {
        // Arrange - 4 correct, 1 wrong = 80% accuracy
        await CreateFeedbackScenario("test-model", correctCount: 4, wrongCount: 1);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().BeApproximately(0.8, 0.01);
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_NoFeedback_DoesNotTrigger()
    {
        // Arrange - no predictions or feedback
        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)
        };

        // Act
        var result = await _trigger.EvaluateAsync("empty-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].Details.Should().Contain("No accuracy data");
    }

    [Fact]
    public async Task EvaluateAsync_FeedbackVolume_TriggersWhenAboveThreshold()
    {
        // Arrange - create 10 predictions with feedback
        await CreateFeedbackScenario("test-model", correctCount: 10, wrongCount: 0);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.FeedbackVolume, "feedback_count", 5)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults[0].IsMet.Should().BeTrue();
        result.ConditionResults[0].CurrentValue.Should().Be(10);
    }

    [Fact]
    public async Task EvaluateAsync_FeedbackVolume_DoesNotTriggerWhenBelowThreshold()
    {
        // Arrange - create 3 predictions with feedback
        await CreateFeedbackScenario("test-model", correctCount: 3, wrongCount: 0);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.FeedbackVolume, "feedback_count", 10)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().Be(3);
    }

    [Fact]
    public async Task EvaluateAsync_MultipleConditions_TriggersIfAnyMet()
    {
        // Arrange - 4 correct, 1 wrong = 80% accuracy, 5 total feedback
        await CreateFeedbackScenario("test-model", correctCount: 4, wrongCount: 1);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.5), // not met (80% > 50%)
            new RetrainingCondition(ConditionType.FeedbackVolume, "feedback_count", 3)  // met (5 >= 3)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults.Should().HaveCount(2);
        result.ConditionResults[0].IsMet.Should().BeFalse(); // accuracy not met
        result.ConditionResults[1].IsMet.Should().BeTrue();  // feedback volume met
    }

    [Fact]
    public async Task EvaluateAsync_UnsupportedCondition_ReturnsNotMet()
    {
        // Arrange
        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.DataDrift, "drift_check", 0.5)
        };

        // Act
        var result = await _trigger.EvaluateAsync("test-model", conditions);

        // Assert
        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].Details.Should().Contain("not supported");
    }

    [Fact]
    public async Task GetDefaultConditionsAsync_ReturnsDefaultConditions()
    {
        // Act
        var conditions = await _trigger.GetDefaultConditionsAsync("test-model");

        // Assert
        conditions.Should().HaveCount(2);
        conditions.Should().Contain(c => c.Type == ConditionType.AccuracyDrop);
        conditions.Should().Contain(c => c.Type == ConditionType.FeedbackVolume);
    }

    private async Task CreateFeedbackScenario(string modelName, int correctCount, int wrongCount)
    {
        var predictions = new List<string>();

        // Create predictions
        for (int i = 0; i < correctCount + wrongCount; i++)
        {
            var output = i < correctCount ? "ClassA" : "ClassB";
            var input = new Dictionary<string, object> { ["feature"] = $"value_{i}" };
            await _logger.LogPredictionAsync(modelName, "exp-001", input, output, 0.9);

            // Get the prediction ID
            var logsPath = Path.Combine(_testDir, ".mloop", "logs", modelName);
            var logFiles = Directory.GetFiles(logsPath, "*.jsonl");
            var lastLine = (await File.ReadAllLinesAsync(logFiles[0])).Last();
            using var doc = JsonDocument.Parse(lastLine);
            predictions.Add(doc.RootElement.GetProperty("id").GetString()!);
        }

        // Record feedback - correct for first N, wrong for rest
        for (int i = 0; i < predictions.Count; i++)
        {
            var actualValue = i < correctCount ? "ClassA" : "ClassA"; // All actual are ClassA
            await _feedbackCollector.RecordFeedbackAsync(predictions[i], actualValue);
        }
    }
}
