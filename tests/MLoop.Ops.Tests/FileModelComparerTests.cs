using System.Text.Json;
using FluentAssertions;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class FileModelComparerTests : IDisposable
{
    private readonly string _testDir;
    private readonly FileModelComparer _comparer;

    public FileModelComparerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _comparer = new FileModelComparer(_testDir);
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
    public async Task CompareAsync_ReturnsCorrectComparison_WhenCandidateIsBetter()
    {
        // Arrange
        var modelName = "test-model";
        await CreateExperimentWithMetrics(modelName, "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85,
            ["F1Score"] = 0.82
        });
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92,
            ["F1Score"] = 0.90
        });

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeTrue();
        result.CandidateExpId.Should().Be("exp-002");
        result.BaselineExpId.Should().Be("exp-001");
        result.Improvement.Should().BeGreaterThan(0);
        result.MetricDetails.Should().ContainKey("Accuracy");
        result.MetricDetails["Accuracy"].IsBetter.Should().BeTrue();
    }

    [Fact]
    public async Task CompareAsync_ReturnsCorrectComparison_WhenBaselineIsBetter()
    {
        // Arrange
        var modelName = "test-model";
        await CreateExperimentWithMetrics(modelName, "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95
        });
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeFalse();
        result.MetricDetails["Accuracy"].IsBetter.Should().BeFalse();
    }

    [Fact]
    public async Task CompareAsync_HandlesLowerIsBetterMetrics()
    {
        // Arrange
        var modelName = "test-model";
        await CreateExperimentWithMetrics(modelName, "exp-001", new Dictionary<string, double>
        {
            ["MeanAbsoluteError"] = 0.15
        });
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["MeanAbsoluteError"] = 0.08
        });

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeTrue();
        result.MetricDetails["MeanAbsoluteError"].IsBetter.Should().BeTrue();
    }

    [Fact]
    public async Task CompareAsync_ThrowsFileNotFoundException_WhenMetricsMissing()
    {
        // Arrange
        var modelName = "test-model";
        var expPath = Path.Combine(_testDir, "models", modelName, "staging", "exp-001");
        Directory.CreateDirectory(expPath);
        // No metrics.json file created

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _comparer.CompareAsync(modelName, "exp-001", "exp-002"));
    }

    private async Task CreateExperimentWithMetrics(
        string modelName,
        string experimentId,
        Dictionary<string, double> metrics)
    {
        // Mirror the real ExperimentStore layout: models/{name}/staging/{expId}/metrics.json.
        var expPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant(), "staging", experimentId);
        Directory.CreateDirectory(expPath);

        var metricsJson = JsonSerializer.Serialize(metrics);
        await File.WriteAllTextAsync(Path.Combine(expPath, "metrics.json"), metricsJson);
    }
}
