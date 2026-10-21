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

    // The model's label column — the feedback's actual value is written under it.
    private static readonly RetrainingColumns Columns = new("label");

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
            outputPath, Columns);

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
            outputPath, Columns);

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
            outputPath, Columns);

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
            outputPath, Columns);

        // Assert
        result.SampledCount.Should().Be(2);
        result.StrategyUsed.Should().Be(SamplingStrategy.Recent);

        // Most recent entries have the higher indices (no feedback: empty label)
        (await File.ReadAllLinesAsync(outputPath)).Skip(1).Should().BeEquivalentTo("4,", "3,");
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
            outputPath, Columns);

        // Assert
        result.SampledCount.Should().Be(3);
        result.StrategyUsed.Should().Be(SamplingStrategy.FeedbackPriority);

        var content = await File.ReadAllTextAsync(outputPath);
        // Should prioritize entries with feedback
        content.Should().Contain("actual_0");
        content.Should().Contain("actual_1");
    }

    [Fact]
    public async Task SampleAsync_DuplicateFeedbackForSamePrediction_DoesNotCrash_UsesLatest()
    {
        // Feedback can be recorded more than once for the same prediction (RecordFeedbackAsync just
        // appends — a user correcting a label). Sampling must not crash, and should use the most
        // recently recorded value.
        var id = await CreatePredictionAndGetId("test-model", "predicted");
        await _feedbackCollector.RecordFeedbackAsync(id, "first");
        await _feedbackCollector.RecordFeedbackAsync(id, "corrected");

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.FeedbackPriority, outputPath, Columns);

        result.SampledCount.Should().Be(1);
        var content = await File.ReadAllTextAsync(outputPath);
        content.Should().Contain("corrected");
        content.Should().NotContain("first");
    }

    [Fact]
    public async Task SampleAsync_WritesFeedbackUnderTheModelsLabel_AndNotTheModelsAnswer()
    {
        // The file is the next training's input: every column in it is a feature, so the model's own
        // answer must not be one (it would learn to copy itself), and the actual value is the label.
        var id = await CreatePredictionAndGetId("test-model", "predicted");
        await _feedbackCollector.RecordFeedbackAsync(id, "actual", source: "reviewer");

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Random, outputPath, Columns);

        var lines = await File.ReadAllLinesAsync(outputPath);
        lines[0].Should().Be("feature,label");
        lines[1].Should().Be("value,actual");
        result.LabeledCount.Should().Be(1);
        result.LabelColumn.Should().Be("label");
        result.WeightColumn.Should().BeNull();
    }

    [Fact]
    public async Task SampleAsync_RowWithoutFeedback_HasAnEmptyLabel()
    {
        await CreatePredictionAndGetId("test-model", "predicted");

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Random, outputPath, Columns);

        (await File.ReadAllLinesAsync(outputPath)).Should().Equal("feature,label", "value,");
        result.LabeledCount.Should().Be(0);
    }

    [Fact]
    public async Task SampleAsync_FeedbackWeights_AreWrittenAsAWeightColumn_UnweightedRowsCountOne()
    {
        var weighted = await CreatePredictionAndGetId("test-model", "a");
        var plain = await CreatePredictionAndGetId("test-model", "b");
        await CreatePredictionAndGetId("test-model", "c");
        await _feedbackCollector.RecordFeedbackAsync(weighted, "x", weight: 4.0);
        await _feedbackCollector.RecordFeedbackAsync(plain, "y");

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Recent, outputPath, Columns);

        result.WeightColumn.Should().Be(RetrainingColumns.DefaultWeight);
        var lines = await File.ReadAllLinesAsync(outputPath);
        lines[0].Should().Be("feature,label,weight");
        lines.Skip(1).Should().BeEquivalentTo("value,x,4", "value,y,1", "value,,1");
    }

    [Fact]
    public async Task SampleAsync_ModelThatTrainsWeighted_GetsItsWeightColumn_EvenWithoutFeedbackWeights()
    {
        // A model declared with `weight: w` needs the column in every file it retrains on.
        var id = await CreatePredictionAndGetId("test-model", "a");
        await _feedbackCollector.RecordFeedbackAsync(id, "x");

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Random, outputPath,
            new RetrainingColumns("label", "w"));

        result.WeightColumn.Should().Be("w");
        (await File.ReadAllLinesAsync(outputPath)).Should().Equal("feature,label,w", "value,x,1");
    }

    [Fact]
    public async Task SampleAsync_DefaultWeightName_StepsAsideForALoggedInputOfThatName()
    {
        await _logger.LogPredictionAsync("test-model", "exp-001",
            new Dictionary<string, object> { ["weight"] = 70 }, "a", 0.9);
        var logFile = Directory.GetFiles(Path.Combine(_testDir, ".mloop", "logs", "test-model"), "*.jsonl")[0];
        using var doc = JsonDocument.Parse((await File.ReadAllLinesAsync(logFile)).Last());
        await _feedbackCollector.RecordFeedbackAsync(doc.RootElement.GetProperty("id").GetString()!, "x", weight: 2);

        var outputPath = Path.Combine(_testDir, "output.csv");
        var result = await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Random, outputPath, Columns);

        result.WeightColumn.Should().Be("feedback_weight");
        (await File.ReadAllLinesAsync(outputPath)).Should().Equal("weight,label,feedback_weight", "70,x,2");
    }

    [Fact]
    public async Task SampleAsync_LoggedLabelColumn_IsReplacedByFeedback_NotDuplicated()
    {
        await _logger.LogPredictionAsync("test-model", "exp-001",
            new Dictionary<string, object> { ["feature"] = "v", ["label"] = "stale" }, "a", 0.9);
        var logFile = Directory.GetFiles(Path.Combine(_testDir, ".mloop", "logs", "test-model"), "*.jsonl")[0];
        using var doc = JsonDocument.Parse((await File.ReadAllLinesAsync(logFile)).Last());
        await _feedbackCollector.RecordFeedbackAsync(doc.RootElement.GetProperty("id").GetString()!, "fresh");

        var outputPath = Path.Combine(_testDir, "output.csv");
        await _sampler.SampleAsync("test-model", 10, SamplingStrategy.Random, outputPath, Columns);

        (await File.ReadAllLinesAsync(outputPath)).Should().Equal("feature,label", "v,fresh");
    }

    [Fact]
    public async Task SampleAsync_StratifiedStrategy_MaintainsClassDistribution()
    {
        // Arrange - create predictions with different classes (60% A, 30% B, 10% C)
        for (int i = 0; i < 6; i++)
        {
            await _logger.LogPredictionAsync(
                "test-model",
                "exp-001",
                new Dictionary<string, object> { ["index"] = i },
                "class_A",
                0.9);
        }
        for (int i = 0; i < 3; i++)
        {
            await _logger.LogPredictionAsync(
                "test-model",
                "exp-001",
                new Dictionary<string, object> { ["index"] = i + 6 },
                "class_B",
                0.8);
        }
        await _logger.LogPredictionAsync(
            "test-model",
            "exp-001",
            new Dictionary<string, object> { ["index"] = 9 },
            "class_C",
            0.7);

        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act - sample 5 out of 10
        var result = await _sampler.SampleAsync(
            "test-model",
            5,
            SamplingStrategy.Stratified,
            outputPath, Columns);

        // Assert
        result.SampledCount.Should().Be(5);
        result.StrategyUsed.Should().Be(SamplingStrategy.Stratified);

        // Strata are the predicted classes: A is indices 0-5, B 6-8. At minimum, at least one row
        // of each class that has enough samples.
        var indices = (await File.ReadAllLinesAsync(outputPath)).Skip(1)
            .Select(line => int.Parse(line.Split(',')[0])).ToList();
        indices.Should().Contain(i => i <= 5);
        indices.Should().Contain(i => i >= 6 && i <= 8);
        // class_C might be included (10% of 5 = 0.5, rounds to 1)
    }

    [Fact]
    public async Task SampleAsync_LowConfidenceStrategy_PrioritizesLowConfidence()
    {
        // Arrange - create predictions with varying confidence
        var confidences = new[] { 0.95, 0.85, 0.75, 0.45, 0.35 };
        for (int i = 0; i < confidences.Length; i++)
        {
            await _logger.LogPredictionAsync(
                "test-model",
                "exp-001",
                new Dictionary<string, object> { ["index"] = i },
                $"output_{i}",
                confidences[i]);
        }

        var outputPath = Path.Combine(_testDir, "output.csv");

        // Act
        var result = await _sampler.SampleAsync(
            "test-model",
            2,
            SamplingStrategy.LowConfidence,
            outputPath, Columns);

        // Assert
        result.SampledCount.Should().Be(2);
        result.StrategyUsed.Should().Be(SamplingStrategy.LowConfidence);

        // The lowest confidence predictions: index 4 (0.35) and 3 (0.45)
        (await File.ReadAllLinesAsync(outputPath)).Skip(1).Should().BeEquivalentTo("4,", "3,");
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
