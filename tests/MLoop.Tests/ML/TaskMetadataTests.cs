using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

/// <summary>
/// Locks the converged task→primary-metric source of truth (TD-06). Previously this mapping was
/// duplicated across InitCommand, ModelRegistry, and TrainingEngine and drifted, producing
/// BUG-46 and F-17. These tests assert the single canonical answer per task.
/// </summary>
public class TaskMetadataTests
{
    [Theory]
    [InlineData("binary-classification", "accuracy")]
    [InlineData("multiclass-classification", "macro_accuracy")]
    [InlineData("image-classification", "micro_accuracy")]
    [InlineData("text-classification", "micro_accuracy")]
    [InlineData("regression", "r_squared")]
    [InlineData("anomaly-detection", "auc")]
    [InlineData("clustering", "average_distance")]
    [InlineData("ranking", "ndcg")]
    [InlineData("forecasting", "mae")]
    [InlineData("time-series-anomaly", "detection_rate")]
    [InlineData("recommendation", "rmse")]
    public void PrimaryMetric_ReturnsCanonicalMetricPerTask(string task, string expected)
    {
        Assert.Equal(expected, TaskMetadata.PrimaryMetric(task));
    }

    [Theory]
    [InlineData("object-detection")]  // mAP has no universal threshold → no scalar primary
    [InlineData("unknown-task")]
    [InlineData(null)]
    public void PrimaryMetric_NoCanonicalMetric_ReturnsNull(string? task)
    {
        Assert.Null(TaskMetadata.PrimaryMetric(task));
    }

    [Theory]
    [InlineData("  regression  ", "r_squared")]   // trimmed
    [InlineData("Binary-Classification", "accuracy")]  // case-insensitive
    public void PrimaryMetric_NormalizesInput(string task, string expected)
    {
        Assert.Equal(expected, TaskMetadata.PrimaryMetric(task));
    }

    [Fact]
    public void PrimaryMetricOrAuto_FallsBackToAuto_WhenNoCanonicalMetric()
    {
        Assert.Equal("auto", TaskMetadata.PrimaryMetricOrAuto("object-detection"));
        Assert.Equal("auto", TaskMetadata.PrimaryMetricOrAuto(null));
        Assert.Equal("r_squared", TaskMetadata.PrimaryMetricOrAuto("regression"));
    }

    [Fact]
    public void AllPrimaryMetrics_ContainsEveryCanonicalMetric_Distinct()
    {
        var all = TaskMetadata.AllPrimaryMetrics.ToList();

        // image and text both map to micro_accuracy → must appear once.
        Assert.Equal(all.Count, all.Distinct().Count());
        Assert.Contains("micro_accuracy", all);
        Assert.Contains("macro_accuracy", all);
        Assert.Contains("r_squared", all);
        Assert.Contains("detection_rate", all);
        Assert.Contains("ndcg", all);
        Assert.Contains("average_distance", all);
    }
}
