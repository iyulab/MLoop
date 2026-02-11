using System.Text.Json;
using FluentAssertions;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class FilePromotionManagerTests : IDisposable
{
    private readonly string _testDir;
    private readonly TestModelComparer _comparer;
    private readonly FilePromotionManager _manager;

    public FilePromotionManagerTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
        _comparer = new TestModelComparer(_testDir);
        _manager = new FilePromotionManager(_testDir, _comparer);
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

    #region EvaluatePromotionAsync Tests

    [Fact]
    public async Task EvaluatePromotionAsync_Fails_WhenCandidateHasNoMetrics()
    {
        var policy = new PromotionPolicy(MinimumImprovement: 1.0);

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-001", policy);

        result.ShouldPromote.Should().BeFalse();
        result.ChecksFailed.Should().Contain(s => s.Contains("no metrics"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Passes_WhenCandidateHasMetrics_NoProduction()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        var policy = new PromotionPolicy(MinimumImprovement: 1.0);

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-001", policy);

        result.ShouldPromote.Should().BeTrue();
        result.ChecksPassed.Should().Contain(s => s.Contains("No existing production"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Passes_WhenCandidateIsBetterThanProduction()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92
        });
        await SetProductionExperiment("test-model", "exp-001");

        _comparer.SetResult(candidateIsBetter: true, improvement: 8.24);
        var policy = new PromotionPolicy(MinimumImprovement: 1.0);

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-002", policy);

        result.ShouldPromote.Should().BeTrue();
        result.ChecksPassed.Should().Contain(s => s.Contains("Better than production"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Fails_WhenCandidateIsWorseThanProduction()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.95
        });
        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });
        await SetProductionExperiment("test-model", "exp-001");

        _comparer.SetResult(candidateIsBetter: false, improvement: -15.79);
        var policy = new PromotionPolicy(MinimumImprovement: 1.0);

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-002", policy);

        result.ShouldPromote.Should().BeFalse();
        result.ChecksFailed.Should().Contain(s => s.Contains("not better"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Fails_WhenImprovementBelowMinimum()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.91
        });
        await SetProductionExperiment("test-model", "exp-001");

        _comparer.SetResult(candidateIsBetter: true, improvement: 1.11);
        var policy = new PromotionPolicy(MinimumImprovement: 5.0);

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-002", policy);

        result.ShouldPromote.Should().BeFalse();
        result.ChecksFailed.Should().Contain(s => s.Contains("below minimum"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Fails_WhenRequiredMetricsMissing()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        var policy = new PromotionPolicy(
            MinimumImprovement: 0,
            RequireComparisonWithProduction: false,
            RequiredMetrics: new List<string> { "Accuracy", "F1Score" });

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-001", policy);

        result.ShouldPromote.Should().BeFalse();
        result.ChecksFailed.Should().Contain(s => s.Contains("F1Score") && s.Contains("missing"));
        result.ChecksPassed.Should().Contain(s => s.Contains("Accuracy") && s.Contains("present"));
    }

    [Fact]
    public async Task EvaluatePromotionAsync_Passes_WhenAllRequiredMetricsPresent()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90,
            ["F1Score"] = 0.88
        });
        var policy = new PromotionPolicy(
            MinimumImprovement: 0,
            RequireComparisonWithProduction: false,
            RequiredMetrics: new List<string> { "Accuracy", "F1Score" });

        var result = await _manager.EvaluatePromotionAsync("test-model", "exp-001", policy);

        result.ShouldPromote.Should().BeTrue();
        result.ChecksPassed.Should().HaveCountGreaterThanOrEqualTo(2);
    }

    #endregion

    #region PromoteAsync Tests

    [Fact]
    public async Task PromoteAsync_CopiesExperimentToProduction()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });

        var result = await _manager.PromoteAsync("test-model", "exp-001");

        result.Success.Should().BeTrue();
        result.PromotedExpId.Should().Be("exp-001");
        result.PreviousExpId.Should().BeNull();

        var productionPath = Path.Combine(_testDir, "models", "test-model", "production");
        Directory.Exists(productionPath).Should().BeTrue();
        File.Exists(Path.Combine(productionPath, "metrics.json")).Should().BeTrue();
    }

    [Fact]
    public async Task PromoteAsync_CreatesBackupOfPreviousProduction()
    {
        // First promotion
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        // Second promotion
        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92
        });
        var result = await _manager.PromoteAsync("test-model", "exp-002");

        result.Success.Should().BeTrue();
        result.PreviousExpId.Should().Be("exp-001");
        result.BackupPath.Should().NotBeNull();
        Directory.Exists(result.BackupPath!).Should().BeTrue();
    }

    [Fact]
    public async Task PromoteAsync_SkipsBackup_WhenDisabled()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92
        });
        var result = await _manager.PromoteAsync("test-model", "exp-002", createBackup: false);

        result.Success.Should().BeTrue();
        result.BackupPath.Should().BeNull();
    }

    [Fact]
    public async Task PromoteAsync_UpdatesRegistry()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        var registryPath = Path.Combine(_testDir, "models", "test-model", "model-registry.json");
        File.Exists(registryPath).Should().BeTrue();

        var json = await File.ReadAllTextAsync(registryPath);
        json.Should().Contain("exp-001");
    }

    [Fact]
    public async Task PromoteAsync_RecordsHistory()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        var history = await _manager.GetHistoryAsync("test-model");

        history.Should().HaveCount(1);
        history[0].ExperimentId.Should().Be("exp-001");
        history[0].Action.Should().Be("promote");
    }

    [Fact]
    public async Task PromoteAsync_Fails_WhenExperimentNotFound()
    {
        var result = await _manager.PromoteAsync("test-model", "nonexistent");

        result.Success.Should().BeFalse();
    }

    #endregion

    #region RollbackAsync Tests

    [Fact]
    public async Task RollbackAsync_RestoresPreviousProduction()
    {
        // Promote exp-001 then exp-002
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92
        });
        await _manager.PromoteAsync("test-model", "exp-002");

        // Rollback
        var result = await _manager.RollbackAsync("test-model");

        result.Success.Should().BeTrue();
        result.RolledBackToExpId.Should().Be("exp-001");
        result.RolledBackFromExpId.Should().Be("exp-002");
    }

    [Fact]
    public async Task RollbackAsync_RollsBackToSpecificExperiment()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-002");

        await CreateExperimentWithMetrics("test-model", "exp-003", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await _manager.PromoteAsync("test-model", "exp-003");

        // Rollback to exp-001 (skipping exp-002)
        var result = await _manager.RollbackAsync("test-model", "exp-001");

        result.Success.Should().BeTrue();
        result.RolledBackToExpId.Should().Be("exp-001");
        result.RolledBackFromExpId.Should().Be("exp-003");
    }

    [Fact]
    public async Task RollbackAsync_Fails_WhenNoProductionModel()
    {
        var result = await _manager.RollbackAsync("test-model");

        result.Success.Should().BeFalse();
    }

    [Fact]
    public async Task RollbackAsync_Fails_WhenTargetExperimentNotFound()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        var result = await _manager.RollbackAsync("test-model", "nonexistent");

        result.Success.Should().BeFalse();
    }

    [Fact]
    public async Task RollbackAsync_RecordsHistory()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.92
        });
        await _manager.PromoteAsync("test-model", "exp-002");

        await _manager.RollbackAsync("test-model");

        var history = await _manager.GetHistoryAsync("test-model");

        history.Should().HaveCount(3); // promote exp-001, promote exp-002, rollback
        history[0].Action.Should().Be("rollback");
    }

    #endregion

    #region GetHistoryAsync Tests

    [Fact]
    public async Task GetHistoryAsync_ReturnsEmpty_WhenNoHistory()
    {
        var history = await _manager.GetHistoryAsync("test-model");

        history.Should().BeEmpty();
    }

    [Fact]
    public async Task GetHistoryAsync_RespectsLimit()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-002");

        await CreateExperimentWithMetrics("test-model", "exp-003", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.90
        });
        await _manager.PromoteAsync("test-model", "exp-003");

        var history = await _manager.GetHistoryAsync("test-model", limit: 2);

        history.Should().HaveCount(2);
        // Should be most recent first
        history[0].ExperimentId.Should().Be("exp-003");
    }

    [Fact]
    public async Task GetHistoryAsync_OrdersByTimestampDescending()
    {
        await CreateExperimentWithMetrics("test-model", "exp-001", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.80
        });
        await _manager.PromoteAsync("test-model", "exp-001");

        await CreateExperimentWithMetrics("test-model", "exp-002", new Dictionary<string, double>
        {
            ["Accuracy"] = 0.85
        });
        await _manager.PromoteAsync("test-model", "exp-002");

        var history = await _manager.GetHistoryAsync("test-model");

        history[0].Timestamp.Should().BeOnOrAfter(history[1].Timestamp);
    }

    #endregion

    #region Helpers

    private async Task CreateExperimentWithMetrics(
        string modelName,
        string experimentId,
        Dictionary<string, double> metrics)
    {
        var expPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant(), "experiments", experimentId);
        Directory.CreateDirectory(expPath);

        var metricsJson = JsonSerializer.Serialize(metrics);
        await File.WriteAllTextAsync(Path.Combine(expPath, "metrics.json"), metricsJson);
    }

    private async Task SetProductionExperiment(string modelName, string experimentId)
    {
        var modelPath = Path.Combine(_testDir, "models", modelName.ToLowerInvariant());
        Directory.CreateDirectory(modelPath);

        var registry = new Dictionary<string, object>
        {
            ["production"] = new Dictionary<string, object>
            {
                ["experimentId"] = experimentId
            }
        };

        var registryJson = JsonSerializer.Serialize(registry);
        await File.WriteAllTextAsync(Path.Combine(modelPath, "model-registry.json"), registryJson);
    }

    #endregion

    /// <summary>
    /// Simple test stub for IModelComparer. Returns configurable comparison results.
    /// </summary>
    private sealed class TestModelComparer : IModelComparer
    {
        private readonly string _projectRoot;
        private bool _candidateIsBetter = true;
        private double _improvement = 5.0;

        public TestModelComparer(string projectRoot)
        {
            _projectRoot = projectRoot;
        }

        public void SetResult(bool candidateIsBetter, double improvement)
        {
            _candidateIsBetter = candidateIsBetter;
            _improvement = improvement;
        }

        public Task<ComparisonResult> CompareAsync(
            string modelName,
            string candidateExpId,
            string baselineExpId,
            CancellationToken cancellationToken = default)
        {
            var result = new ComparisonResult(
                candidateExpId,
                baselineExpId,
                _candidateIsBetter,
                CandidateScore: 0.92,
                BaselineScore: 0.85,
                _improvement,
                new Dictionary<string, MetricComparison>(),
                _candidateIsBetter ? "Candidate is better" : "Baseline is better");

            return Task.FromResult(result);
        }

        public Task<ComparisonResult> CompareWithProductionAsync(
            string modelName,
            string candidateExpId,
            CancellationToken cancellationToken = default)
        {
            return CompareAsync(modelName, candidateExpId, "(production)", cancellationToken);
        }

        public Task<string?> FindBestExperimentAsync(
            string modelName,
            ComparisonCriteria criteria,
            CancellationToken cancellationToken = default)
        {
            return Task.FromResult<string?>(null);
        }
    }
}
