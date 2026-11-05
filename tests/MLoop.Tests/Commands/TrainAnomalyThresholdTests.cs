using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>anomaly_threshold</c> under a model in <c>mloop.yaml</c> — the score above which a time-series anomaly model flags a point. A
/// value that is no score, or one on a task that has no detector score, is refused instead of silently meaning nothing.
/// </summary>
public sealed class TrainAnomalyThresholdTests
{
    private static ModelDefinition Definition(string task, double? threshold)
        => new() { Task = task, Label = "value", AnomalyThreshold = threshold };

    [Theory]
    [InlineData(0.05)]
    [InlineData(0.3)]
    [InlineData(0.99)]
    public void AScoreOnATimeSeriesAnomalyModel_HasNoConflict_AndReachesTheTrainingConfig(double threshold)
    {
        var definition = Definition("time-series-anomaly", threshold);

        Assert.Null(ModelTrainingConfig.AnomalyThresholdConflict(definition));
        Assert.Equal(threshold, ModelTrainingConfig.From(definition, "default", "series.csv", 60, 0.2).AnomalyThreshold);
    }

    [Fact]
    public void NoThreshold_LeavesTheDetectorsDefault()
    {
        var definition = Definition("time-series-anomaly", null);

        Assert.Null(ModelTrainingConfig.AnomalyThresholdConflict(definition));
        Assert.Null(ModelTrainingConfig.From(definition, "default", "series.csv", 60, 0.2).AnomalyThreshold);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-0.5)]
    [InlineData(3.0)]
    [InlineData(double.NaN)]
    public void AValueThatIsNoScore_IsRefused(double threshold)
        => Assert.Contains("not a score threshold", ModelTrainingConfig.AnomalyThresholdConflict(Definition("time-series-anomaly", threshold)), StringComparison.Ordinal);

    [Theory]
    [InlineData("regression")]
    [InlineData("anomaly-detection")]
    [InlineData("forecasting")]
    [InlineData("clustering")]
    public void AThresholdOnATaskWithoutADetectorScore_IsRefusedAndNamesTheTask(string task)
    {
        var message = ModelTrainingConfig.AnomalyThresholdConflict(Definition(task, 0.2));

        Assert.NotNull(message);
        Assert.Contains($"'{task}'", message, StringComparison.Ordinal);
        Assert.Contains("time-series-anomaly", message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheYamlKey_IsReadFromTheFile_AndTheAuditKnowsIt()
    {
        const string Yaml = "models:\n  default:\n    task: time-series-anomaly\n    label: value\n    anomaly_threshold: 0.2\n";

        Assert.Empty(ConfigKeyAudit.Find<MLoopConfig>(Yaml));
        Assert.Single(ConfigKeyAudit.Find<MLoopConfig>(Yaml.Replace("anomaly_threshold", "anomaly_treshold")));
    }

    [Fact]
    public void ThePartnersOfTheKey_ReachTheEffectiveDefinition()
    {
        var config = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = Definition("time-series-anomaly", 0.2) } };

        Assert.Equal(0.2, new ConfigMerger().GetEffectiveModelDefinition(config, "default").AnomalyThreshold);
    }
}
