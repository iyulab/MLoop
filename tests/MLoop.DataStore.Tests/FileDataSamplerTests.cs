using System.Text.Json;
using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

public class FileDataSamplerTests : IDisposable
{
    private readonly string _testDir;
    private readonly FileDataSampler _sampler;
    private readonly FilePredictionLogger _logger;
    private readonly FileFeedbackCollector _feedbackCollector;

    public FileDataSamplerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _sampler = new FileDataSampler(_testDir);
        _logger = new FilePredictionLogger(_testDir);
        _feedbackCollector = new FileFeedbackCollector(_testDir);
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
    public async Task SampleAsync_ThrowsException_WhenNoPredictions()
    {
        // Arrange
        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act & Assert
        var act = async () => await _sampler.SampleAsync(
            "nonexistent-model",
            100,
            SamplingStrategy.Random,
            outputPath);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*No predictions*");
    }

    [Fact]
    public async Task SampleAsync_CreatesCsvFile_WithRandomStrategy()
    {
        // Arrange
        await CreatePredictions("test-model", 10);
        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        var result = await _sampler.SampleAsync(
            "test-model",
            5,
            SamplingStrategy.Random,
            outputPath);

        // Assert
        result.SampledCount.Should().Be(5);
        result.TotalAvailable.Should().Be(10);
        result.StrategyUsed.Should().Be(SamplingStrategy.Random);
        File.Exists(outputPath).Should().BeTrue();

        var lines = await File.ReadAllLinesAsync(outputPath);
        lines.Length.Should().Be(6); // header + 5 data rows
    }

    [Fact]
    public async Task SampleAsync_ReturnsAllAvailable_WhenSampleSizeExceeds()
    {
        // Arrange
        await CreatePredictions("test-model", 5);
        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        var result = await _sampler.SampleAsync(
            "test-model",
            100, // request more than available
            SamplingStrategy.Random,
            outputPath);

        // Assert
        result.SampledCount.Should().Be(5);
        result.TotalAvailable.Should().Be(5);
    }

    [Fact]
    public async Task SampleAsync_RecentStrategy_ReturnsMostRecent()
    {
        // Arrange - create predictions with slight delay
        for (int i = 0; i < 5; i++)
        {
            await _logger.LogPredictionAsync(
                "test-model",
                "exp-001",
                new Dictionary<string, object> { ["index"] = i },
                $"output_{i}",
                0.9);
            await Task.Delay(10); // small delay to ensure order
        }

        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        var result = await _sampler.SampleAsync(
            "test-model",
            2,
            SamplingStrategy.Recent,
            outputPath);

        // Assert
        result.SampledCount.Should().Be(2);
        result.StrategyUsed.Should().Be(SamplingStrategy.Recent);

        var content = await File.ReadAllTextAsync(outputPath);
        // Most recent entries should have higher indices
        content.Should().Contain("output_4");
        content.Should().Contain("output_3");
    }

    [Fact]
    public async Task SampleAsync_FeedbackPriority_PrioritizesFeedback()
    {
        // Arrange - create 5 predictions, add feedback to 2
        var predictionIds = new List<string>();
        for (int i = 0; i < 5; i++)
        {
            var id = await CreatePredictionAndGetId("test-model", $"output_{i}");
            predictionIds.Add(id);
        }

        // Add feedback to first 2 predictions
        await _feedbackCollector.RecordFeedbackAsync(predictionIds[0], "actual_0");
        await _feedbackCollector.RecordFeedbackAsync(predictionIds[1], "actual_1");

        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        var result = await _sampler.SampleAsync(
            "test-model",
            3,
            SamplingStrategy.FeedbackPriority,
            outputPath);

        // Assert
        result.SampledCount.Should().Be(3);
        result.StrategyUsed.Should().Be(SamplingStrategy.FeedbackPriority);

        var content = await File.ReadAllTextAsync(outputPath);
        // Should prioritize entries with feedback
        content.Should().Contain("actual_0");
        content.Should().Contain("actual_1");
    }

    [Fact]
    public async Task SampleAsync_IncludesActualValueColumn_WhenFeedbackExists()
    {
        // Arrange
        var id = await CreatePredictionAndGetId("test-model", "predicted");
        await _feedbackCollector.RecordFeedbackAsync(id, "actual");

        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        await _sampler.SampleAsync(
            "test-model",
            10,
            SamplingStrategy.Random,
            outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        lines[0].Should().Contain("ActualValue"); // header should have ActualValue
        lines[1].Should().Contain("actual");
    }

    [Fact]
    public async Task GetStatisticsAsync_ReturnsZeroes_WhenNoPredictions()
    {
        // Act
        var stats = await _sampler.GetStatisticsAsync("nonexistent-model");

        // Assert
        stats.TotalPredictions.Should().Be(0);
        stats.PredictionsWithFeedback.Should().Be(0);
        stats.LowConfidenceCount.Should().Be(0);
    }

    [Fact]
    public async Task GetStatisticsAsync_ReturnsCorrectStats()
    {
        // Arrange - 5 predictions, 2 with feedback, 2 low confidence
        for (int i = 0; i < 5; i++)
        {
            var confidence = i < 2 ? 0.5 : 0.9; // first 2 are low confidence
            await _logger.LogPredictionAsync(
                "test-model",
                "exp-001",
                new Dictionary<string, object> { ["feature"] = i },
                $"output_{i}",
                confidence);
        }

        // Add feedback to 2 predictions
        var logsPath = Path.Combine(_testDir, ".mloop", "logs", "test-model");
        var logFiles = Directory.GetFiles(logsPath, "*.jsonl");
        var lines = await File.ReadAllLinesAsync(logFiles[0]);

        for (int i = 0; i < 2 && i < lines.Length; i++)
        {
            using var doc = JsonDocument.Parse(lines[i]);
            var id = doc.RootElement.GetProperty("id").GetString()!;
            await _feedbackCollector.RecordFeedbackAsync(id, $"actual_{i}");
        }

        // Act
        var stats = await _sampler.GetStatisticsAsync("test-model");

        // Assert
        stats.ModelName.Should().Be("test-model");
        stats.TotalPredictions.Should().Be(5);
        stats.PredictionsWithFeedback.Should().Be(2);
        stats.LowConfidenceCount.Should().Be(2);
        stats.OldestEntry.Should().NotBe(DateTimeOffset.MinValue);
        stats.NewestEntry.Should().NotBe(DateTimeOffset.MinValue);
    }

    private async Task CreatePredictions(string modelName, int count)
    {
        for (int i = 0; i < count; i++)
        {
            await _logger.LogPredictionAsync(
                modelName,
                "exp-001",
                new Dictionary<string, object> { ["feature"] = $"value_{i}" },
                $"output_{i}",
                0.9);
        }
    }

    private async Task<string> CreatePredictionAndGetId(string modelName, object output)
    {
        var input = new Dictionary<string, object> { ["feature"] = "value" };
        await _logger.LogPredictionAsync(modelName, "exp-001", input, output, 0.9);

        var logsPath = Path.Combine(_testDir, ".mloop", "logs", modelName);
        var logFiles = Directory.GetFiles(logsPath, "*.jsonl");
        var lastLine = (await File.ReadAllLinesAsync(logFiles[0])).Last();

        using var doc = JsonDocument.Parse(lastLine);
        return doc.RootElement.GetProperty("id").GetString()!;
    }
}
