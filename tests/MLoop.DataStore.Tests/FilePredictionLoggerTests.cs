using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

public class FilePredictionLoggerTests : IDisposable
{
    private readonly string _testDir;
    private readonly FilePredictionLogger _logger;

    public FilePredictionLoggerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
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
    public async Task LogPredictionAsync_CreatesSingleEntry()
    {
        // Arrange
        var input = new Dictionary<string, object>
        {
            ["feature1"] = 1.0,
            ["feature2"] = "test"
        };

        // Act
        await _logger.LogPredictionAsync(
            "test-model",
            "exp-001",
            input,
            "positive",
            0.95);

        // Assert
        var logs = await _logger.GetLogsAsync("test-model");
        logs.Should().HaveCount(1);
        logs[0].ModelName.Should().Be("test-model");
        logs[0].ExperimentId.Should().Be("exp-001");
        logs[0].Output.ToString().Should().Be("positive");
        logs[0].Confidence.Should().Be(0.95);
    }

    [Fact]
    public async Task LogBatchAsync_CreatesMultipleEntries()
    {
        // Arrange
        var entries = new List<PredictionLogEntry>
        {
            new("model1", "exp1", new Dictionary<string, object> { ["x"] = 1 }, "A", 0.9, DateTimeOffset.UtcNow),
            new("model1", "exp1", new Dictionary<string, object> { ["x"] = 2 }, "B", 0.8, DateTimeOffset.UtcNow),
            new("model1", "exp1", new Dictionary<string, object> { ["x"] = 3 }, "C", 0.7, DateTimeOffset.UtcNow)
        };

        // Act
        await _logger.LogBatchAsync("model1", "exp1", entries);

        // Assert
        var logs = await _logger.GetLogsAsync("model1");
        logs.Should().HaveCount(3);
    }

    [Fact]
    public async Task GetLogsAsync_ReturnsEmptyForNoLogs()
    {
        // Act
        var logs = await _logger.GetLogsAsync("nonexistent-model");

        // Assert
        logs.Should().BeEmpty();
    }

    [Fact]
    public async Task GetLogsAsync_FiltersBy_ModelName()
    {
        // Arrange
        await _logger.LogPredictionAsync("model-a", "exp1", new Dictionary<string, object>(), "out1");
        await _logger.LogPredictionAsync("model-b", "exp1", new Dictionary<string, object>(), "out2");
        await _logger.LogPredictionAsync("model-a", "exp2", new Dictionary<string, object>(), "out3");

        // Act
        var logsA = await _logger.GetLogsAsync("model-a");
        var logsB = await _logger.GetLogsAsync("model-b");

        // Assert
        logsA.Should().HaveCount(2);
        logsB.Should().HaveCount(1);
    }

    [Fact]
    public async Task GetLogsAsync_RespectsLimit()
    {
        // Arrange
        for (int i = 0; i < 10; i++)
        {
            await _logger.LogPredictionAsync("model", "exp", new Dictionary<string, object>(), $"out{i}");
        }

        // Act
        var logs = await _logger.GetLogsAsync("model", limit: 5);

        // Assert
        logs.Should().HaveCount(5);
    }

    [Fact]
    public async Task GetLogsAsync_FiltersBy_DateRange()
    {
        // Arrange
        var now = DateTimeOffset.UtcNow;
        var entry1 = new PredictionLogEntry("model", "exp", new Dictionary<string, object>(), "old", null, now.AddDays(-5));
        var entry2 = new PredictionLogEntry("model", "exp", new Dictionary<string, object>(), "recent", null, now.AddDays(-1));
        var entry3 = new PredictionLogEntry("model", "exp", new Dictionary<string, object>(), "today", null, now);

        await _logger.LogBatchAsync("model", "exp", new[] { entry1, entry2, entry3 });

        // Act
        var logs = await _logger.GetLogsAsync("model", from: now.AddDays(-2), to: now.AddDays(1));

        // Assert
        logs.Should().HaveCount(2);
        logs.Should().Contain(l => l.Output!.ToString() == "recent");
        logs.Should().Contain(l => l.Output!.ToString() == "today");
    }

    [Fact]
    public async Task GetLogsAsync_WithoutModelName_ReturnsAllModels()
    {
        // Arrange
        await _logger.LogPredictionAsync("model-x", "exp", new Dictionary<string, object>(), "x");
        await _logger.LogPredictionAsync("model-y", "exp", new Dictionary<string, object>(), "y");

        // Act
        var logs = await _logger.GetLogsAsync();

        // Assert
        logs.Should().HaveCount(2);
        logs.Select(l => l.ModelName).Should().Contain(new[] { "model-x", "model-y" });
    }

    [Fact]
    public async Task LogPredictionAsync_HandlesSpecialCharactersInModelName()
    {
        // Arrange
        var modelName = "my:model/test";
        var input = new Dictionary<string, object> { ["x"] = 1 };

        // Act
        await _logger.LogPredictionAsync(modelName, "exp", input, "result");

        // Assert - should not throw, and sanitized name should work
        var logs = await _logger.GetLogsAsync(modelName);
        logs.Should().HaveCount(1);
    }
}
