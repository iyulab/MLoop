using System.Text.Json;
using FluentAssertions;
using MLoop.DataStore.Interfaces;
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

    /// <summary>Logs <paramref name="inside"/> predictions whose actual value fell inside the interval they stated (90 %, [10, 20]) and <paramref name="outside"/> whose actual fell outside, and records the feedback.</summary>
    private async Task StatedIntervalScenario(string model, int inside, int outside, double? weight = null)
    {
        for (var i = 0; i < inside + outside; i++)
        {
            var id = PredictionIds.New();
            await _logger.LogBatchAsync(model, "exp-001",
            [
                new PredictionLogEntry(model, "exp-001", new Dictionary<string, object> { ["x"] = i }, 15.0, null,
                    DateTimeOffset.UtcNow, id, new PredictionInterval(10, 20, 0.9)),
            ]);
            await _feedbackCollector.RecordFeedbackAsync(id, i < inside ? 15.0 : 40.0, weight: weight);
        }
    }

    private static readonly RetrainingCondition Coverage = RetrainingDefaults.Coverage();

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_TriggersWhenTheStatedLevelIsEstablishedBroken()
    {
        // 15 of 40 inside a stated 90 %: 37.5 % — even the best case of that share (about 53 %) is under 85 % (the level less the tolerated 5 points).
        await StatedIntervalScenario("reg", inside: 15, outside: 25);

        var result = await _trigger.EvaluateAsync("reg", [Coverage]);

        result.ShouldRetrain.Should().BeTrue();
        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeTrue();
        r.CurrentValue.Should().BeApproximately(0.375, 1e-9);
        r.Details.Should().Contain("37.5%").And.Contain("90%").And.Contain("95% interval up to");
        result.RecommendedAction.Should().Contain(RetrainingDefaults.CoverageName);
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_DoesNotTriggerOnAShareThatCouldBeTheStatedLevel()
    {
        // 32 of 40 = 80 %: under the 85 % floor on its face, but its 95 % interval reaches past it — not established.
        await StatedIntervalScenario("reg", inside: 32, outside: 8);

        var result = await _trigger.EvaluateAsync("reg", [Coverage]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults.Single().CurrentValue.Should().BeApproximately(0.8, 1e-9);
        result.ConditionResults.Single().Details.Should().Contain("not established");
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_DoesNotTriggerWithinTheTolerance()
    {
        // 35 of 40 = 87.5 % of a stated 90 %: a couple of points under the level is not what a retraining is for.
        await StatedIntervalScenario("reg", inside: 35, outside: 5);

        var result = await _trigger.EvaluateAsync("reg", [Coverage]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults.Single().Details.Should().Contain("at or above 85.0%");
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_NeedsEnoughFeedbackToJudge()
    {
        // Two of ten inside is 20 % — and ten rows are too few to call it.
        await StatedIntervalScenario("reg", inside: 2, outside: 8);

        var result = await _trigger.EvaluateAsync("reg", [Coverage]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults.Single().Details.Should().Contain("too few to judge").And.Contain("30");
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_CountsWeightedFeedbackByItsEffectiveRows()
    {
        // 40 rows all weighted 1/16 stand for the evidence of 40 rows (Kish: equal weights), the same verdict as unweighted.
        await StatedIntervalScenario("reg", inside: 15, outside: 25, weight: 16);

        var result = await _trigger.EvaluateAsync("reg", [Coverage]);

        result.ConditionResults.Single().IsMet.Should().BeTrue();
        result.ConditionResults.Single().Details.Should().Contain("effective rows");
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_IsNotMeasuredWithoutFeedbackThatStatedAnInterval()
    {
        // A classifier's feedback states no interval: the condition is not measured — not met, no value, and says why.
        for (var i = 0; i < 40; i++)
        {
            var id = PredictionIds.New();
            await _logger.LogBatchAsync("clf", "exp-001", [new PredictionLogEntry("clf", "exp-001", new Dictionary<string, object> { ["x"] = i }, "yes", 0.9, DateTimeOffset.UtcNow, id)]);
            await _feedbackCollector.RecordFeedbackAsync(id, "no");
        }

        var result = await _trigger.EvaluateAsync("clf", [Coverage]);

        result.ShouldRetrain.Should().BeFalse();
        var r = result.ConditionResults.Single();
        r.CurrentValue.Should().BeNull();
        r.Details.Should().Contain("no feedback for a prediction that stated an interval");
    }

    [Fact]
    public async Task EvaluateAsync_IntervalCoverage_IsNotMeasuredWithoutAnyFeedback()
    {
        var result = await _trigger.EvaluateAsync("nothing", [Coverage]);

        result.ConditionResults.Single().IsMet.Should().BeFalse();
        result.ConditionResults.Single().CurrentValue.Should().BeNull();
    }

    [Fact]
    public void The_coverage_default_tolerates_a_few_points_and_names_itself()
    {
        var c = RetrainingDefaults.Coverage();

        c.Type.Should().Be(ConditionType.IntervalCoverage);
        c.Name.Should().Be(RetrainingDefaults.CoverageName).And.Be("interval_coverage");
        c.Threshold.Should().Be(RetrainingDefaults.CoverageShortfall).And.BeApproximately(0.05, 1e-12);
        RetrainingDefaults.All.Should().Contain(Coverage);
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_DoesNotJudgeValueAnswersThatStatedAnInterval()
    {
        // 40 regression answers, none "equal" to its actual value: matched as values that is accuracy 0 and a
        // retrain for every regression model. Read as the intervals they stated they are judged by coverage —
        // and the trigger says that, not that no feedback was collected.
        for (var i = 0; i < 40; i++)
        {
            var id = PredictionIds.New();
            await _logger.LogBatchAsync("reg", "exp-001",
            [
                new PredictionLogEntry("reg", "exp-001", new Dictionary<string, object> { ["x"] = i }, 15.0 + i,
                    null, DateTimeOffset.UtcNow, id, new PredictionInterval(10 + i, 20 + i, 0.9)),
            ]);
            await _feedbackCollector.RecordFeedbackAsync(id, 16.3 + i);
        }

        var result = await _trigger.EvaluateAsync("reg", [new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].IsMet.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().BeNull();
        result.ConditionResults[0].Details.Should().Contain("Accuracy does not apply").And.Contain("interval coverage");
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_TriggersWhenEstablishedBelowThreshold()
    {
        // 12 correct of 60 = 20%; even the 95% interval's best case (about 32%) is below 50%.
        await CreateFeedbackScenario("test-model", correctCount: 12, wrongCount: 48);

        var conditions = new[]
        {
            new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.5)
        };

        var result = await _trigger.EvaluateAsync("test-model", conditions);

        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults.Should().HaveCount(1);
        result.ConditionResults[0].IsMet.Should().BeTrue();
        result.ConditionResults[0].CurrentValue.Should().BeApproximately(0.2, 0.01);
        result.ConditionResults[0].Details.Should().Contain("95% interval");
        result.RecommendedAction.Should().Contain("accuracy_check");
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_AHandfulOfWrongAnswersIsNotADrop()
    {
        // 2 correct, 3 wrong = 40% — on five rows, not evidence of anything.
        await CreateFeedbackScenario("test-model", correctCount: 2, wrongCount: 3);

        var result = await _trigger.EvaluateAsync("test-model",
            [new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.5)]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].CurrentValue.Should().BeApproximately(0.4, 0.01);
        result.ConditionResults[0].Details.Should().Contain("too few to judge");
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_BelowOnItsFaceButWithinTheInterval_IsNotEstablished()
    {
        // 26 of 40 = 65% against 70%: the interval reaches about 78%.
        await CreateFeedbackScenario("test-model", correctCount: 26, wrongCount: 14);

        var result = await _trigger.EvaluateAsync("test-model",
            [new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)]);

        result.ShouldRetrain.Should().BeFalse();
        result.ConditionResults[0].Details.Should().Contain("not established");
    }

    [Theory]
    [InlineData(0.5, 100, 0.5962)]
    [InlineData(0.2, 60, 0.3178)]
    [InlineData(1.0, 30, 1.0)]
    [InlineData(0.0, 30, 0.1135)]
    public void AccuracyUpperBound_IsTheWilsonUpperEnd(double accuracy, double n, double expected)
    {
        RetrainingDefaults.AccuracyUpperBound(accuracy, n).Should().BeApproximately(expected, 0.0005);
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
        result.ConditionResults[0].CurrentValue.Should().BeNull("an accuracy nobody measured is not 0");
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
        conditions.Should().HaveCount(3);
        conditions.Should().Contain(c => c.Type == ConditionType.AccuracyDrop);
        conditions.Should().Contain(c => c.Type == ConditionType.FeedbackVolume);
        conditions.Should().Contain(c => c.Type == ConditionType.IntervalCoverage);
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_JudgesProductionsOwnAnswers()
    {
        // exp-001 answered wrong 60 times before exp-002 replaced it; exp-002 has answered 40 times, 38 right.
        await CreateFeedbackScenario("test-model", correctCount: 0, wrongCount: 60, experimentId: "exp-001");
        await CreateFeedbackScenario("test-model", correctCount: 38, wrongCount: 2, experimentId: "exp-002");
        var trigger = new FeedbackBasedTrigger(_feedbackCollector, (_, _) => Task.FromResult<string?>("exp-002"));

        var result = await trigger.EvaluateAsync("test-model",
            [new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)]);

        result.ShouldRetrain.Should().BeFalse("the replaced model's errors are not the production model's");
        result.ConditionResults[0].CurrentValue.Should().BeApproximately(0.95, 0.001);
        result.ConditionResults[0].Details.Should().Contain("exp-002's answers");
    }

    [Fact]
    public async Task EvaluateAsync_AccuracyDrop_WithoutAProductionModel_ReadsEveryAnswer()
    {
        await CreateFeedbackScenario("test-model", correctCount: 0, wrongCount: 60, experimentId: "exp-001");
        var trigger = new FeedbackBasedTrigger(_feedbackCollector, (_, _) => Task.FromResult<string?>(null));

        var result = await trigger.EvaluateAsync("test-model",
            [new RetrainingCondition(ConditionType.AccuracyDrop, "accuracy_check", 0.7)]);

        result.ShouldRetrain.Should().BeTrue();
        result.ConditionResults[0].Details.Should().NotContain("answers;");
    }

    private async Task CreateFeedbackScenario(string modelName, int correctCount, int wrongCount, string experimentId = "exp-001")
    {
        var predictions = new List<string>();

        // Create predictions
        for (int i = 0; i < correctCount + wrongCount; i++)
        {
            var output = i < correctCount ? "ClassA" : "ClassB";
            var input = new Dictionary<string, object> { ["feature"] = $"value_{i}" };
            await _logger.LogPredictionAsync(modelName, experimentId, input, output, 0.9);

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
