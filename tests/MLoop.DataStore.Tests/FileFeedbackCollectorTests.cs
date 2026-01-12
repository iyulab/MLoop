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
