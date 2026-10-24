using System.Text.Json;
using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

public class FileFeedbackCollectorTests : IDisposable
{
    private readonly string _testDir;
    private readonly FileFeedbackCollector _collector;
    private readonly FilePredictionLogger _logger;

    public FileFeedbackCollectorTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _collector = new FileFeedbackCollector(_testDir);
        _logger = new FilePredictionLogger(_testDir);
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
    public async Task RecordFeedbackAsync_FindsAPredictionLoggedLongAgo()
    {
        // Feedback after human review can come weeks later; it used to be refused as "not found" past 7 days.
        var predictionId = await CreatePredictionLog("test-model", "positive");
        var logDir = Path.Combine(_testDir, ".mloop", "logs", "test-model");
        var oldDay = DateTimeOffset.UtcNow.AddDays(-30).ToString("yyyy-MM-dd");
        var today = Directory.GetFiles(logDir, "*.jsonl").Single();
        File.Move(today, Path.Combine(logDir, $"{oldDay}.jsonl"));

        await _collector.RecordFeedbackAsync(predictionId, "negative");

        var feedback = await _collector.GetFeedbackAsync("test-model");
        feedback.Should().ContainSingle(f => f.PredictionId == predictionId);
    }

    [Fact]
    public async Task RecordFeedbackAsync_Success_WhenPredictionExists()
    {
        // Arrange
        var predictionId = await CreatePredictionLog("test-model", "positive");

        // Act
        await _collector.RecordFeedbackAsync(predictionId, "positive", "test");

        // Assert
        var feedback = await _collector.GetFeedbackAsync("test-model");
        feedback.Should().HaveCount(1);
        feedback[0].PredictionId.Should().Be(predictionId);
        feedback[0].ActualValue.ToString().Should().Be("positive");
        feedback[0].Source.Should().Be("test");
    }

    [Fact]
    public async Task RecordFeedbackAsync_ThrowsException_WhenPredictionNotFound()
    {
        // Arrange
        var fakePredictionId = "nonexistent123";

        // Act & Assert
        var act = async () => await _collector.RecordFeedbackAsync(fakePredictionId, "value");
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*not found*");
    }

    [Fact]
    public async Task RecordFeedbackAsync_NotFound_IsItsOwnError()
    {
        var act = async () => await _collector.RecordFeedbackAsync("0000000000000000", "value");
        (await act.Should().ThrowAsync<PredictionNotFoundException>())
            .Which.PredictionId.Should().Be("0000000000000000");
    }

    [Fact]
    public async Task RecordFeedbackAsync_AgainCorrectsIt_AndMetricsCountThePredictionOnce()
    {
        var predictionId = await CreatePredictionLog("model", "A");
        await _collector.RecordFeedbackAsync(predictionId, "B");
        await _collector.RecordFeedbackAsync(predictionId, "A");

        var metrics = await _collector.CalculateMetricsAsync("model");

        metrics.TotalFeedback.Should().Be(1, "a correction replaces the earlier actual value");
        metrics.Accuracy.Should().Be(1.0, "the latest actual value stands");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public async Task RecordFeedbackAsync_RequiresAPredictionId(string? predictionId)
    {
        var act = async () => await _collector.RecordFeedbackAsync(predictionId!, "value");
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task GetFeedbackAsync_ReturnsEmpty_WhenNoFeedback()
    {
        // Act
        var feedback = await _collector.GetFeedbackAsync("nonexistent-model");

        // Assert
        feedback.Should().BeEmpty();
    }

    [Fact]
    public async Task GetFeedbackAsync_FiltersBy_ModelName()
    {
        // Arrange
        var id1 = await CreatePredictionLog("model-a", "out1");
        var id2 = await CreatePredictionLog("model-b", "out2");
        var id3 = await CreatePredictionLog("model-a", "out3");

        await _collector.RecordFeedbackAsync(id1, "actual1");
        await _collector.RecordFeedbackAsync(id2, "actual2");
        await _collector.RecordFeedbackAsync(id3, "actual3");

        // Act
        var feedbackA = await _collector.GetFeedbackAsync("model-a");
        var feedbackB = await _collector.GetFeedbackAsync("model-b");

        // Assert
        feedbackA.Should().HaveCount(2);
        feedbackB.Should().HaveCount(1);
    }

    [Fact]
    public async Task GetFeedbackAsync_RespectsLimit()
    {
        // Arrange
        for (int i = 0; i < 10; i++)
        {
            var id = await CreatePredictionLog("model", $"out{i}");
            await _collector.RecordFeedbackAsync(id, $"actual{i}");
        }

        // Act
        var feedback = await _collector.GetFeedbackAsync("model", limit: 5);

        // Assert
        feedback.Should().HaveCount(5);
    }

    [Fact]
    public async Task CalculateMetricsAsync_ReturnsAccuracy_WhenFeedbackExists()
    {
        // Arrange - 3 correct, 2 wrong = 60% accuracy
        var id1 = await CreatePredictionLog("model", "A");
        var id2 = await CreatePredictionLog("model", "B");
        var id3 = await CreatePredictionLog("model", "A");
        var id4 = await CreatePredictionLog("model", "B");
        var id5 = await CreatePredictionLog("model", "A");

        await _collector.RecordFeedbackAsync(id1, "A");  // correct
        await _collector.RecordFeedbackAsync(id2, "B");  // correct
        await _collector.RecordFeedbackAsync(id3, "B");  // wrong
        await _collector.RecordFeedbackAsync(id4, "A");  // wrong
        await _collector.RecordFeedbackAsync(id5, "A");  // correct

        // Act
        var metrics = await _collector.CalculateMetricsAsync("model");

        // Assert
        metrics.TotalFeedback.Should().Be(5);
        metrics.Accuracy.Should().BeApproximately(0.6, 0.01);
        metrics.EffectiveSampleSize.Should().Be(5); // unweighted: one row is one row of evidence
    }

    [Fact]
    public async Task CalculateMetricsAsync_NumericPrediction_StringActual_CountsAsMatch()
    {
        // numeric prediction output (regression / numeric class label) is logged as a JSON
        // number, but the CLI actual value is always a string. Accuracy must register the obvious
        // match (predicted 1 vs actual "1") — it previously reported 0% for all such models.
        var id1 = await CreatePredictionLog("model", 1);
        var id2 = await CreatePredictionLog("model", 2);
        var id3 = await CreatePredictionLog("model", 3);
        await _collector.RecordFeedbackAsync(id1, "1");  // correct
        await _collector.RecordFeedbackAsync(id2, "2");  // correct
        await _collector.RecordFeedbackAsync(id3, "9");  // wrong

        var metrics = await _collector.CalculateMetricsAsync("model");
        metrics.Accuracy.Should().BeApproximately(2.0 / 3.0, 0.01);
    }

    [Fact]
    public async Task CalculateMetricsAsync_ReturnsNullAccuracy_WhenNoFeedback()
    {
        // Act
        var metrics = await _collector.CalculateMetricsAsync("empty-model");

        // Assert
        metrics.TotalFeedback.Should().Be(0);
        metrics.Accuracy.Should().BeNull();
    }

    [Fact]
    public async Task RecordFeedbackAsync_StoresPredictedValue()
    {
        // Arrange
        var id = await CreatePredictionLog("model", "predicted_label");

        // Act
        await _collector.RecordFeedbackAsync(id, "actual_label");

        // Assert
        var feedback = await _collector.GetFeedbackAsync("model");
        feedback.Should().HaveCount(1);
        feedback[0].PredictedValue.ToString().Should().Be("predicted_label");
        feedback[0].ActualValue.ToString().Should().Be("actual_label");
    }

    [Fact]
    public async Task RecordFeedbackAsync_Weight_IsKeptWithTheEntry()
    {
        var weighted = await CreatePredictionLog("model", "a");
        var plain = await CreatePredictionLog("model", "b");
        await _collector.RecordFeedbackAsync(weighted, "a", source: "reviewer", weight: 2.5);
        await _collector.RecordFeedbackAsync(plain, "b");

        var feedback = await _collector.GetFeedbackAsync("model");

        var w = feedback.Single(f => f.PredictionId == weighted);
        w.Weight.Should().Be(2.5);
        w.Source.Should().Be("reviewer");
        feedback.Single(f => f.PredictionId == plain).Weight.Should().BeNull();
        feedback.Single(f => f.PredictionId == plain).EffectiveWeight.Should().Be(1.0);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public async Task RecordFeedbackAsync_WeightThatCannotWeightARow_IsRefused_AndNothingIsWritten(double weight)
    {
        var id = await CreatePredictionLog("model", "a");

        var act = () => _collector.RecordFeedbackAsync(id, "a", weight: weight);

        (await act.Should().ThrowAsync<ArgumentOutOfRangeException>()).Which.ParamName.Should().Be("weight");
        (await _collector.GetFeedbackAsync("model")).Should().BeEmpty();
    }

    [Fact]
    public async Task CalculateMetricsAsync_WithWeights_CountsEachPredictionByItsWeight()
    {
        // One right answer reviewed with probability 1/3 stands for three rows; one wrong answer
        // reviewed with certainty stands for one. Counted alike the accuracy would read 0.5.
        var right = await CreatePredictionLog("model", "yes");
        var wrong = await CreatePredictionLog("model", "yes");
        await _collector.RecordFeedbackAsync(right, "yes", weight: 3);
        await _collector.RecordFeedbackAsync(wrong, "no", weight: 1);

        var metrics = await _collector.CalculateMetricsAsync("model");

        metrics.Weighted.Should().BeTrue();
        metrics.Accuracy.Should().BeApproximately(0.75, 1e-12);
        metrics.TotalFeedback.Should().Be(2);
        // Kish: (3 + 1)² ÷ (3² + 1²) = 1.6 — two unequal rows are worth less than two equal ones.
        metrics.EffectiveSampleSize.Should().BeApproximately(1.6, 1e-12);
    }

    [Fact]
    public async Task CalculateMetricsAsync_AllWeightsZero_HasNoAccuracy()
    {
        var id = await CreatePredictionLog("model", "yes");
        await _collector.RecordFeedbackAsync(id, "yes", weight: 0);

        var metrics = await _collector.CalculateMetricsAsync("model");

        metrics.Accuracy.Should().BeNull();
    }

    /// <summary>
    /// Helper to create a prediction log and return its ID.
    /// </summary>
    private async Task<string> CreatePredictionLog(string modelName, object output)
    {
        var input = new Dictionary<string, object> { ["feature"] = "value" };
        await _logger.LogPredictionAsync(modelName, "exp-001", input, output, 0.9);

        // Read the log file to get the prediction ID
        var logsPath = Path.Combine(_testDir, ".mloop", "logs", modelName);
        var logFiles = Directory.GetFiles(logsPath, "*.jsonl");
        var lastLine = (await File.ReadAllLinesAsync(logFiles[0])).Last();

        using var doc = JsonDocument.Parse(lastLine);
        return doc.RootElement.GetProperty("id").GetString()!;
    }
}
