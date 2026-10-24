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
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.82
        });
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["accuracy"] = 0.92,
            ["f1_score"] = 0.90
        });

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeTrue();
        result.CandidateExpId.Should().Be("exp-002");
        result.BaselineExpId.Should().Be("exp-001");
        result.Improvement.Should().BeGreaterThan(0);
        result.Metric.Should().Be("accuracy");   // "auto": the binary primary metric
        result.MetricDetails.Should().ContainKey("accuracy");
        result.MetricDetails["accuracy"].IsBetter.Should().BeTrue();
    }

    [Fact]
    public async Task CompareAsync_ReturnsCorrectComparison_WhenBaselineIsBetter()
    {
        // Arrange
        var modelName = "test-model";
        await CreateExperimentWithMetrics(modelName, "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.95
        });
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["accuracy"] = 0.80
        });

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeFalse();
        result.MetricDetails["accuracy"].IsBetter.Should().BeFalse();
    }

    [Fact]
    public async Task CompareAsync_HandlesLowerIsBetterMetrics()
    {
        // Arrange
        var modelName = "test-model";
        await CreateExperimentWithMetrics(modelName, "exp-001", new Dictionary<string, double>
        {
            ["mae"] = 0.15
        }, task: "regression", metric: "mae");
        await CreateExperimentWithMetrics(modelName, "exp-002", new Dictionary<string, double>
        {
            ["mae"] = 0.08
        }, task: "regression", metric: "mae");

        // Act
        var result = await _comparer.CompareAsync(modelName, "exp-002", "exp-001");

        // Assert
        result.CandidateIsBetter.Should().BeTrue();
        result.MetricDetails["mae"].IsBetter.Should().BeTrue();
        result.Improvement.Should().BeApproximately(46.67, 0.01);   // positive: lower is better here
    }

    [Fact]
    public async Task OneMetricDecides_NotTheMajorityOrTheMeanOfMixedScales()
    {
        // Better on two of three metrics, worse on the one the run optimized: the run's metric decides.
        await CreateExperimentWithMetrics("m", "exp-001", new() { ["r_squared"] = 0.90, ["rmse"] = 12.0, ["mae"] = 9.0 }, task: "regression", metric: "r_squared");
        await CreateExperimentWithMetrics("m", "exp-002", new() { ["r_squared"] = 0.80, ["rmse"] = 11.0, ["mae"] = 8.0 }, task: "regression", metric: "r_squared");

        var result = await _comparer.CompareAsync("m", "exp-002", "exp-001");

        result.Metric.Should().Be("r_squared");
        result.CandidateIsBetter.Should().BeFalse();
        result.CandidateScore.Should().Be(0.80);
        result.BaselineScore.Should().Be(0.90);
        result.Recommendation.Should().Contain("r_squared");
    }

    [Fact]
    public async Task AMetricOneSideLacks_IsLeftOut_NotReadAsZero()
    {
        // Read as 0, a missing rmse used to be a win for whichever side lacked it.
        await CreateExperimentWithMetrics("m", "exp-001", new() { ["r_squared"] = 0.90, ["rmse"] = 12.0 }, task: "regression");
        await CreateExperimentWithMetrics("m", "exp-002", new() { ["r_squared"] = 0.80 }, task: "regression");

        var result = await _comparer.CompareAsync("m", "exp-002", "exp-001");

        result.MetricDetails.Keys.Should().BeEquivalentTo(["r_squared"]);
        result.Metric.Should().Be("r_squared");
        result.CandidateIsBetter.Should().BeFalse();
    }

    [Fact]
    public async Task TwoExperimentsSharingNoMetric_AreNotJudged()
    {
        await CreateExperimentWithMetrics("m", "exp-001", new() { ["rmse"] = 12.0 }, task: "regression", metric: "rmse");
        await CreateExperimentWithMetrics("m", "exp-002", new() { ["mae"] = 8.0 }, task: "regression", metric: "rmse");

        var result = await _comparer.CompareAsync("m", "exp-002", "exp-001");

        result.Metric.Should().BeNull();
        result.CandidateIsBetter.Should().BeFalse();
        result.Recommendation.Should().Contain("share no metric");
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
        Dictionary<string, double> metrics,
        string task = "binary-classification",
        string? metric = null)
    {
        // Mirror the real ExperimentStore layout: models/{name}/staging/{expId}/metrics.json.
        var expPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant(), "staging", experimentId);
        Directory.CreateDirectory(expPath);

        var metricsJson = JsonSerializer.Serialize(metrics);
        await File.WriteAllTextAsync(Path.Combine(expPath, "metrics.json"), metricsJson);
        // The task and the metric the run optimized, where ExperimentStore writes them.
        await File.WriteAllTextAsync(Path.Combine(expPath, "metadata.json"),
            JsonSerializer.Serialize(new { task, config = new { metric = metric ?? "auto" } }));
    }
}
