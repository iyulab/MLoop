using FluentAssertions;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;

namespace MLoop.API.Tests;

/// <summary>
/// A training job promotes on the metric it optimized, as <c>mloop train</c> does — not on whatever
/// key happens to come first in the result's metric dictionary (for binary classification that is
/// always <c>accuracy</c>).
/// </summary>
public class TrainingJobPromotionTests
{
    [Theory]
    [InlineData("binary-classification", "auc", "auc")]
    [InlineData("binary-classification", "F1Score", "f1_score")]
    [InlineData("binary-classification", "auto", "accuracy")]
    [InlineData("regression", "rmse", "rmse")]
    public async Task Promotion_is_judged_on_the_optimized_metric(string task, string metric, string expected)
    {
        var registry = new RecordingRegistry();
        var config = new TrainingConfig
        {
            ModelName = "default",
            DataFile = "train.csv",
            LabelColumn = "y",
            Task = task,
            Metric = metric
        };
        var result = new TrainingResult
        {
            ExperimentId = "exp-002",
            Trainer = new TrainerDescriptor { Name = "FastTree" },
            // Insertion order the runner produces: accuracy / r_squared first.
            Metrics = task == "regression"
                ? new() { ["r_squared"] = 0.9, ["rmse"] = 1.2, ["mae"] = 0.8 }
                : new() { ["accuracy"] = 0.8, ["f1_score"] = 0.7, ["auc"] = 0.85 },
            TrainingTimeSeconds = 1,
            ModelPath = "model.zip"
        };

        await TrainingJobRunner.PromoteIfBetterAsync(registry, config, result, CancellationToken.None);

        registry.PrimaryMetric.Should().Be(expected);
        registry.ExperimentId.Should().Be("exp-002");
    }

    private sealed class RecordingRegistry : IModelRegistry
    {
        public string? PrimaryMetric { get; private set; }
        public string? ExperimentId { get; private set; }

        public Task<bool> AutoPromoteAsync(string modelName, string experimentId, string primaryMetric, CancellationToken cancellationToken = default)
        {
            ExperimentId = experimentId;
            PrimaryMetric = primaryMetric;
            return Task.FromResult(true);
        }

        public Task PromoteAsync(string modelName, string experimentId, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<ModelInfo?> GetProductionAsync(string modelName, CancellationToken cancellationToken = default) => Task.FromResult<ModelInfo?>(null);
        public Task<IEnumerable<ModelInfo>> ListAsync(string? modelName = null, CancellationToken cancellationToken = default) => Task.FromResult(Enumerable.Empty<ModelInfo>());
        public string GetProductionPath(string modelName) => "";
        public bool HasProduction(string modelName) => false;
        public string GetProductionModelFile(string modelName) => "";
        public Task<bool> ShouldPromoteAsync(string modelName, string experimentId, string primaryMetric, CancellationToken cancellationToken = default) => Task.FromResult(true);
    }
}
